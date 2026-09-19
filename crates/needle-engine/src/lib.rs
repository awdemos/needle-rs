//! The Needle 3 engine: turns a rendered prompt into the JSON response
//! envelope — free-runs the think block, constrains the tool-call JSON with a
//! schema grammar, computes calibrated confidence, and applies the
//! suppression floor. Ported from the behavior contract in `needle/llms.txt`
//! and `needle/__init__.py`.

pub mod grammar;
pub mod prompt;

use grammar::Grammar;
use needle_model::{Cache, Cells, HeadKind, Model};
use needle_tokenizer::Tokenizer;
use rayon::prelude::*;
use serde_json::{json, Value};

pub use needle_model;

/// One parsed tool call.
#[derive(Debug, Clone, serde::Serialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: Value,
}

/// The response envelope (mirrors the Python package's contract).
#[derive(Debug, Clone)]
pub struct Response {
    pub r#type: String,
    pub success: bool,
    pub error: Option<String>,
    pub error_code: Option<String>,
    pub function_calls: Vec<FunctionCall>,
    pub suppressed_calls: Vec<FunctionCall>,
    pub reasoning: String,
    pub confidence: Option<f64>,
    pub prefill_tps: f64,
    pub decode_tps: f64,
    /// executed tool results, attached by the agent loop
    pub results: Option<Value>,
    /// grounding annotations (`{"ungrounded": ["tool.field"]}`)
    pub validation: Option<Value>,
}

impl Response {
    pub fn to_json(&self) -> Value {
        json!({
            "type": self.r#type,
            "success": self.success,
            "error": self.error,
            "error_code": self.error_code,
            "function_calls": self.function_calls,
            "suppressed_calls": self.suppressed_calls,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "prefill_tps": self.prefill_tps,
            "decode_tps": self.decode_tps,
        })
    }
}

/// Confidence below this is withheld into `suppressed_calls`.
pub const CONFIDENCE_FLOOR: f64 = 0.1;

fn softmax_in_place(x: &mut [f64]) {
    let m = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut s = 0.0;
    for v in x.iter_mut() {
        *v = (*v - m).exp();
        s += *v;
    }
    for v in x.iter_mut() {
        *v /= s;
    }
}

/// Runs the engine over a model + tokenizer with a fixed tool set.
pub struct Engine<'m> {
    pub model: &'m Model,
    pub tokenizer: Tokenizer,
    tools: Vec<Value>,
    tools_json: String,
}

impl<'m> Engine<'m> {
    pub fn new(model: &'m Model, tokenizer: Tokenizer, tools: Vec<Value>) -> Engine<'m> {
        let tools_json = serde_json::to_string(&tools).unwrap_or_else(|_| "[]".to_string());
        Engine {
            model,
            tokenizer,
            tools,
            tools_json,
        }
    }

    pub fn tools(&self) -> &[Value] {
        &self.tools
    }

    pub fn tools_json(&self) -> &str {
        &self.tools_json
    }

    /// Greedy next token (argmax over logits).
    fn argmax(logits: &[f32]) -> usize {
        logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Compute the calibrated confidence for the turn.
    pub fn confidence(&self, token_ids: &[u32], call_logprobs: &[f64]) -> Option<f64> {
        confidence(self.model, token_ids, call_logprobs)
    }

    /// (combined, head, decode) — exposed for diagnostics.
    pub fn confidence_parts(&self, token_ids: &[u32], call_logprobs: &[f64]) -> Option<(f64, f64, f64)> {
        confidence_parts(self.model, token_ids, call_logprobs)
    }

    /// Sentence embedding: normalized per-position embeddings, masked-mean
    /// pooled. `None` when the archive ships no embedding head.
    pub fn embed(&self, token_ids: &[u32]) -> Option<Vec<f32>> {
        let head = self.model.head(HeadKind::Embedding)?;
        let cells = self.model.cells_for(token_ids).ok()?;
        let p = probe_head_rows(self.model, head, &cells, true)?;
        let denom: f32 = p.iter().map(|v| v * v).sum::<f32>().sqrt() + 1e-12;
        Some(p.iter().map(|v| v / denom).collect())
    }

    /// Decode one user turn. `cache` carries the conversation; `history`
    /// is the full token list (for cells at the end).
    pub fn complete_turn(
        &self,
        query: &str,
        cache: &mut Cache,
        history: &mut Vec<u32>,
        system: Option<&str>,
        max_new_tokens: usize,
    ) -> Result<Response, String> {
        complete_turn(
            self.model,
            &self.tokenizer,
            &self.tools_json,
            &self.tools,
            query,
            cache,
            history,
            system,
            max_new_tokens,
        )
    }
}

/// Decode one user turn over borrowed parts (free-function form for agents
/// that own the pieces separately).
#[allow(clippy::too_many_arguments)]
pub fn complete_turn(
    model: &Model,
    tokenizer: &Tokenizer,
    tools_json: &str,
    tools: &[Value],
    query: &str,
    cache: &mut Cache,
    history: &mut Vec<u32>,
    system: Option<&str>,
    max_new_tokens: usize,
) -> Result<Response, String> {
        let prompt = prompt::render(tools_json, query, system);
        let mut ids = vec![needle_tokenizer::BOS_ID];
        ids.extend(tokenizer.encode(&prompt));
        history.extend_from_slice(&ids);
        let t0 = std::time::Instant::now();
        let mut logits = model
            .forward(&ids, cache, false)
            .map_err(|e| e.to_string())?
            .0;
        let prefill_tps = ids.len() as f64 / t0.elapsed().as_secs_f64().max(1e-9);

        // ---- phase 1: free-run until <tool_call> ----
        let t_decode_start = std::time::Instant::now();
        let mut generated: Vec<u32> = Vec::new();
        let call_start_id = tokenizer.piece_id(needle_tokenizer::TOOL_CALL_START);
        let call_end_id = tokenizer.piece_id(needle_tokenizer::TOOL_CALL_END);
        let im_end_id = tokenizer.piece_id(needle_tokenizer::IM_END);
        let mut call_region: Vec<usize> = Vec::new();
        let mut call_logprobs: Vec<f64> = Vec::new();
        let mut phase1_tokens = 0usize;
        loop {
            let next = Engine::argmax(&logits);
            if next == needle_tokenizer::EOS_ID as usize || Some(next as u32) == im_end_id {
                let free_text = tokenizer.decode(&generated);
                let reasoning = extract_think(&free_text);
                let decode_tps = phase1_tokens as f64 / t_decode_start.elapsed().as_secs_f64().max(1e-9);
                return Ok(Response {
                    r#type: "respond".into(),
                    success: true,
                    error: None,
                    error_code: None,
                    function_calls: Vec::new(),
                    suppressed_calls: Vec::new(),
                    reasoning,
                    confidence: confidence(model, history, &[]),
                    prefill_tps,
                    decode_tps,
                    results: None,
                    validation: None,
                });
            }
            generated.push(next as u32);
            history.push(next as u32);
            phase1_tokens += 1;
            logits = model.step(next as u32, cache.len, cache, None)
                .map_err(|e| e.to_string())?;
            if Some(next as u32) == call_start_id {
                break;
            }
            if phase1_tokens >= max_new_tokens {
                return Err("max_new_tokens exceeded before <tool_call>".into());
            }
        }
        let free_text = tokenizer.decode(&generated);
        let reasoning = extract_think(&free_text);

        // ---- phase 2: grammar-constrained JSON ----
        let mut grammar = Grammar::compile(tools);
        let vocab = tokenizer.vocab_size();
        let mut json_tokens: Vec<u32> = Vec::new();
        let mut decode_steps = 0usize;
        loop {
            if grammar.is_done() {
                break;
            }
            if decode_steps >= max_new_tokens {
                return Err("max_new_tokens exceeded inside tool_call".into());
            }
            // allow tokens whose every byte the grammar accepts
            let pick = (0..vocab)
                .into_par_iter()
                .filter_map(|tid| {
                    let piece = tokenizer.piece(tid as u32);
                    if piece.is_empty() {
                        return None;
                    }
                    let mut g = grammar.clone();
                    if piece.bytes().all(|b| g.step(b)) {
                        Some((tid, logits[tid]))
                    } else {
                        None
                    }
                })
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|(tid, _)| tid);
            // note: the grammar is mid-string; opening quote handled per-frame —
            // the filter above starts each candidate from the CURRENT state.
            let next = match pick {
                Some(t) => t,
                None => return Err("grammar dead-end: no valid token".into()),
            };
            let piece = tokenizer.piece(next as u32);
            let mut g = grammar.clone();
            let ok = piece.bytes().all(|b| g.step(b));
            if !ok {
                return Err("selected token failed grammar step".into());
            }
            grammar = g;
            // record the decode probability of the chosen token
            let mut probs: Vec<f64> = logits.iter().map(|&v| v as f64).collect();
            softmax_in_place(&mut probs);
            let p = probs.get(next).copied().unwrap_or(0.0).max(1e-12);
            call_logprobs.push(p.ln());
            call_region.push(next);
            json_tokens.push(next as u32);
            generated.push(next as u32);
            history.push(next as u32);
            logits = model.step(next as u32, cache.len, cache, None)
                .map_err(|e| e.to_string())?;
            decode_steps += 1;
        }
        // ---- phase 3: closing </tool_call> (+ optional <|im_end|>) ----
        let mut closing: Vec<u32> = Vec::new();
        for _ in 0..4 {
            let next = Engine::argmax(&logits);
            closing.push(next as u32);
            generated.push(next as u32);
            history.push(next as u32);
            logits = model.step(next as u32, cache.len, cache, None)
                .map_err(|e| e.to_string())?;
            if Some(next as u32) == call_end_id || Some(next as u32) == im_end_id {
                break;
            }
        }
        let _ = call_region;

        // ---- parse the JSON region ----
        let json_text = tokenizer.decode(&json_tokens);
        let calls: Vec<FunctionCall> = match serde_json::from_str::<Value>(&json_text) {
            Ok(Value::Array(arr)) => arr
                .into_iter()
                .filter_map(|c| {
                    let name = c.get("name")?.as_str()?.to_string();
                    let arguments = c.get("arguments").cloned().unwrap_or(json!({}));
                    Some(FunctionCall { name, arguments })
                })
                .collect(),
            _ => Vec::new(),
        };

        let confidence = confidence(model, history, &call_logprobs);
        let mut response = Response {
            // the engine reports "call" for both calls and refusals (empty
            // function_calls); "respond" is reserved for turns that never
            // reached <tool_call>
            r#type: "call".into(),
            success: true,
            error: None,
            error_code: None,
            function_calls: calls,
            suppressed_calls: Vec::new(),
            reasoning,
            confidence,
            prefill_tps,
            decode_tps: (phase1_tokens + decode_steps) as f64
                / t_decode_start.elapsed().as_secs_f64().max(1e-9),
            results: None,
            validation: None,
        };
        // suppression floor
        if let Some(c) = response.confidence {
            if c < CONFIDENCE_FLOOR && !response.function_calls.is_empty() {
                response.suppressed_calls = std::mem::take(&mut response.function_calls);
                response.r#type = "respond".into();
            }
        }
        Ok(response)
}

/// Calibrated confidence: min of the confidence head (sigmoid of its logit)
/// and the mean decode probability of the tool-call region. 1.0 when the
/// archive carries no head.
pub fn confidence(model: &Model, token_ids: &[u32], call_logprobs: &[f64]) -> Option<f64> {
    confidence_parts(model, token_ids, call_logprobs).map(|(c, _, _)| c)
}

/// (combined, head, decode).
pub fn confidence_parts(model: &Model, token_ids: &[u32], call_logprobs: &[f64]) -> Option<(f64, f64, f64)> {
    let decode_prob = if call_logprobs.is_empty() {
        1.0
    } else {
        let mean_lp: f64 = call_logprobs.iter().sum::<f64>() / call_logprobs.len() as f64;
        mean_lp.exp().clamp(0.0, 1.0)
    };
    let calibrated = confidence_head(model, token_ids).unwrap_or(1.0);
    Some((calibrated.min(decode_prob).clamp(0.0, 1.0), calibrated, decode_prob))
}

/// The confidence head over the full sequence: sigmoid(logit at the last
/// position). Probe-head pooling ported from `architecture.py::probe_pool`.
pub fn confidence_head(model: &Model, token_ids: &[u32]) -> Option<f64> {
    let head = model.head(HeadKind::Confidence)?;
    let cells = model.cells_for(token_ids).ok()?;
    let logit = probe_head_forward(model, head, &cells, true)?[0];
    Some((1.0 / (1.0 + (-logit as f64).exp())).clamp(0.0, 1.0))
}

fn extract_think(text: &str) -> String {
    let start = text.find(needle_tokenizer::THINK_START).map(|i| i + needle_tokenizer::THINK_START.len());
    let end = text.find(needle_tokenizer::THINK_END);
    match (start, end) {
        (Some(s), Some(e)) if e >= s => text[s..e].trim().to_string(),
        _ => String::new(),
    }
}

/// `probe_pool` from `architecture.py`: pooled per-row cells → `[q*d]`.
pub fn probe_head_rows(
    _model: &Model,
    head: &needle_model::HeadWeights,
    cells: &Cells,
    with_proj: bool,
) -> Option<Vec<f32>> {
    let d = cells.d;
    let rows = cells.rows;
    let k = head.k;
    let q = head.q;
    let scale = 1.0 / (d as f32).sqrt();
    // scores[l,k,t] = dot(cells[t,l], probes[l,k]) * scale  (keep = all ones)
    let mut scores = vec![vec![vec![0.0f32; cells.t]; k]; rows];
    for l in 0..rows {
        for kk in 0..k {
            let probe = &head.probes[(l * k + kk) * d..(l * k + kk + 1) * d];
            for t in 0..cells.t {
                let cell = &cells.data[(t * rows + l) * d..(t * rows + l + 1) * d];
                let mut acc = 0.0f32;
                for i in 0..d {
                    acc += cell[i] * probe[i];
                }
                scores[l][kk][t] = acc * scale;
            }
        }
    }
    // r[l,k] = softmax_t(scores) . cells[:,l]; r = rms_unit(r) * gain
    let mut r = vec![0.0f32; rows * k * d];
    for l in 0..rows {
        for kk in 0..k {
            let sc = &scores[l][kk];
            let mut w: Vec<f64> = sc.iter().map(|&v| v as f64).collect();
            softmax_in_place(&mut w);
            let mut vec_r = vec![0.0f32; d];
            for t in 0..cells.t {
                let cell = &cells.data[(t * rows + l) * d..(t * rows + l + 1) * d];
                let wt = w[t] as f32;
                for i in 0..d {
                    vec_r[i] += wt * cell[i];
                }
            }
            // rms_unit
            let ss: f32 = vec_r.iter().map(|v| v * v).sum();
            let inv = 1.0 / (ss / d as f32 + 1e-6).sqrt();
            let gain = head.gain[l * k + kk];
            for i in 0..d {
                r[(l * k + kk) * d + i] = vec_r[i] * inv * gain;
            }
        }
    }
    // u[q,l,k] = dot(r[l,k], query[q]) * scale + row_bias; softmax over (l,k)
    let mut out = vec![0.0f32; q * d];
    for qi in 0..q {
        let mut u = vec![0.0f64; rows * k];
        for l in 0..rows {
            for kk in 0..k {
                let rv = &r[(l * k + kk) * d..(l * k + kk + 1) * d];
                let query = &head.query[qi * d..(qi + 1) * d];
                let mut acc = 0.0f32;
                for i in 0..d {
                    acc += rv[i] * query[i];
                }
                u[l * k + kk] = acc as f64 * scale as f64 + head.row_bias[qi * rows * k + l * k + kk] as f64;
            }
        }
        softmax_in_place(&mut u);
        for l in 0..rows {
            for kk in 0..k {
                let w = u[l * k + kk] as f32;
                let rv = &r[(l * k + kk) * d..(l * k + kk + 1) * d];
                for i in 0..d {
                    out[qi * d + i] += w * rv[i];
                }
            }
        }
    }
    if with_proj {
        // apply proj [out_dim, q*d] + bias
        let out_dim = head.proj.len() / (q * d);
        let mut y = vec![0.0f32; out_dim];
        for o in 0..out_dim {
            let row = &head.proj[o * q * d..(o + 1) * q * d];
            let mut acc = head.bias.get(o).copied().unwrap_or(0.0);
            for i in 0..q * d {
                acc += row[i] * out[i];
            }
            y[o] = acc;
        }
        Some(y)
    } else {
        Some(out)
    }
}

fn probe_head_forward(model: &Model, head: &needle_model::HeadWeights, cells: &Cells, with_proj: bool) -> Option<Vec<f32>> {
    probe_head_rows(model, head, cells, with_proj)
}
