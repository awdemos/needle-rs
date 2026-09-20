//! The `Needle` agent API — a Rust analogue of `needle.Needle`.
//!
//! Tools are JSON Schemas (the same shape the Python package consumes when you
//! pass raw schemas). `complete` drives one conversation turn; `run` executes
//! calls through your callback and feeds results back until the model stops
//! calling; `extract` declares a record schema as the only tool.

mod grounding;

pub use grounding::{annotate_ungrounded, validate_extraction, ExtractionError};
pub use needle_engine::{Response, CONFIDENCE_FLOOR};
pub use needle_model;

use chrono::{Datelike, Timelike};
use needle_engine::needle_model::{Cache, Model};
use needle_engine::Engine;
use needle_format::read_archive;
use needle_tokenizer::Tokenizer;
use serde_json::{json, Value};
use std::path::Path;

/// Executes tool calls for [`Needle::run`]. Return the tool's result as JSON.
pub trait Executor {
    fn call(&mut self, name: &str, arguments: &Value) -> Result<Value, String>;
}

impl<F: FnMut(&str, &Value) -> Result<Value, String>> Executor for F {
    fn call(&mut self, name: &str, arguments: &Value) -> Result<Value, String> {
        self(name, arguments)
    }
}

/// A `Needle` agent bound to one toolset.
pub struct Needle {
    pub model: Model,
    pub tokenizer: Tokenizer,
    tools: Vec<Value>,
    system: Option<String>,
    cache: Cache,
    history: Vec<u32>,
    tool_schemas: Vec<Value>,
    seen_years: std::collections::BTreeSet<u32>,
}

impl Needle {
    /// Create an agent. `weights` is a `.cact` archive (e.g. the published
    /// `needle3.cact`); `tools` are raw JSON Schemas; `system` is the optional
    /// environment-facts string (a `date:` fact is always prepended unless
    /// present, mirroring Python's `auto_date=True`).
    pub fn new(weights: &Path, tools: Vec<Value>, system: Option<String>) -> Result<Needle, String> {
        let archive = read_archive(weights).map_err(|e| e.to_string())?;
        let model = Model::from_archive(&archive).map_err(|e| e.to_string())?;
        let tokenizer = Tokenizer::from_blob(archive.tokenizer_blob().ok_or("no tokenizer in archive")?)
            .map_err(|e| e.to_string())?;
        let tool_schemas = tools.clone();
        let system = dated_system(system);
        Ok(Needle {
            cache: model.new_cache(),
            model,
            tokenizer,
            tools,
            system,
            history: Vec::new(),
            tool_schemas,
            seen_years: Default::default(),
        })
    }

    pub fn engine(&self) -> Engine<'_> {
        Engine::new(&self.model, self.tokenizer.clone(), self.tools.clone())
    }

    /// One conversation turn; execute any calls yourself and feed results back
    /// via the next `complete`.
    pub fn complete(&mut self, text: &str) -> Result<Response, String> {
        self.complete_limited(text, 512)
    }

    pub fn complete_limited(&mut self, text: &str, max_new_tokens: usize) -> Result<Response, String> {
        self.seen_years.extend(grounding::source_years(text));
        let tools_json = serde_json::to_string(&self.tools).unwrap_or_else(|_| "[]".into());
        let mut response = needle_engine::complete_turn(
            &self.model,
            &self.tokenizer,
            &tools_json,
            &self.tools,
            text,
            &mut self.cache,
            &mut self.history,
            self.system.as_deref(),
            max_new_tokens,
        )?;
        grounding::annotate_ungrounded(&mut response, &self.tool_schemas, &self.seen_years, self.system.as_deref(), Some(text));
        Ok(response)
    }

    /// The full agentic loop: execute calls through `exec`, feed results back,
    /// and attach the executed results to the final response.
    pub fn run(&mut self, query: &str, exec: &mut dyn Executor) -> Result<Response, String> {
        self.run_limited(query, 8, 512, true, exec)
    }

    pub fn run_limited(
        &mut self,
        query: &str,
        max_steps: usize,
        max_new_tokens: usize,
        strict: bool,
        exec: &mut dyn Executor,
    ) -> Result<Response, String> {
        let mut response = self.complete_limited(query, max_new_tokens)?;
        let mut executed: Vec<Value> = Vec::new();
        for _ in 0..max_steps {
            if response.r#type != "call" || response.function_calls.is_empty() {
                break;
            }
            let ungrounded = grounding::ungrounded_paths(&response);
            let mut results: Vec<Value> = Vec::new();
            for call in &response.function_calls {
                let name = call.name.clone();
                let fabricated: std::collections::BTreeSet<String> =
                    ungrounded.get(&name).cloned().unwrap_or_default();
                let fabricated = if strict {
                    grounding::filter_fabricated(
                        fabricated,
                        &call.arguments,
                        &[query, self.system.as_deref().unwrap_or_default()],
                    )
                } else {
                    fabricated.into_iter().collect()
                };
                if strict && !fabricated.is_empty() {
                    results.push(json!({"error": format!("ungrounded {}", fabricated.join(", "))}));
                    continue;
                }
                match exec.call(&name, &call.arguments) {
                    Ok(v) => results.push(v),
                    Err(e) => results.push(json!({"error": e})),
                }
            }
            executed.extend(results.clone());
            let feedback = serde_json::to_string(&results).unwrap();
            let mut next = self.feed_result(&feedback, max_new_tokens)?;
            // carry validation of the follow-up turn lightly
            next.error = next.error.take();
            response = next;
        }
        // attach results into the envelope JSON via a wrapper field
        let mut env = response.to_json();
        env["results"] = Value::Array(executed);
        response.results = Some(env["results"].clone());
        Ok(response)
    }

    /// Feed a tool-result back as the next turn (the multi-turn wire format:
    /// the result JSON inside `<tool_result>` markers in a user turn).
    pub fn feed_result(&mut self, result_json: &str, max_new_tokens: usize) -> Result<Response, String> {
        let turn = format!(
            "{}{result_json}{}",
            needle_tokenizer::TOOL_RESULT_START,
            needle_tokenizer::TOOL_RESULT_END
        );
        // Result feedback rides a user turn via the same render path, with the
        // system block re-attached so follow-up turns keep the environment facts.
        let tools_json = serde_json::to_string(&self.tools).unwrap_or_else(|_| "[]".into());
        let mut response = needle_engine::complete_turn(
            &self.model,
            &self.tokenizer,
            &tools_json,
            &self.tools,
            &turn,
            &mut self.cache,
            &mut self.history,
            self.system.as_deref(),
            max_new_tokens,
        )?;
        // The engine prepends the tools block again; that matches the training
        // per-turn template closely enough for the follow-up to parse.
        grounding::annotate_ungrounded(&mut response, &self.tool_schemas, &self.seen_years, self.system.as_deref(), Some(result_json));
        Ok(response)
    }

    /// One-shot structured extraction: the record schema is the only tool.
    pub fn extract(&mut self, text: &str, record_schema: &Value, strict: bool) -> Result<Option<Value>, ExtractionError> {
        self.extract_limited(text, record_schema, 512, strict)
    }

    pub fn extract_limited(
        &mut self,
        text: &str,
        record_schema: &Value,
        max_new_tokens: usize,
        strict: bool,
    ) -> Result<Option<Value>, ExtractionError> {
        let name = record_schema
            .get("name")
            .and_then(|n| n.as_str())
            .unwrap_or("record")
            .to_string();
        let parameters = record_schema
            .get("parameters")
            .cloned()
            .unwrap_or_else(|| record_schema.clone());
        let tool = json!({"name": name, "parameters": parameters});
        let engine = Engine::new(&self.model, self.tokenizer.clone(), vec![tool.clone()]);
        let mut cache = self.model.new_cache();
        let mut history: Vec<u32> = Vec::new();
        let mut response = engine
            .complete_turn(text, &mut cache, &mut history, self.system.as_deref(), max_new_tokens)
            .map_err(ExtractionError::Engine)?;
        // Python's extract annotates the envelope before validating (its
        // `_complete` runs with `ground=True`), so date fabrications raised by
        // the model itself also count as engine-reported flags here.
        let seen = grounding::source_years(text);
        grounding::annotate_ungrounded(&mut response, std::slice::from_ref(&tool), &seen, self.system.as_deref(), Some(text));
        let calls = if response.function_calls.is_empty() {
            &response.suppressed_calls
        } else {
            &response.function_calls
        };
        let Some(call) = calls.first() else {
            return Ok(None);
        };
        if strict {
            validate_extraction(text, &tool, &call.arguments, &response, self.system.as_deref())?;
        }
        Ok(Some(call.arguments.clone()))
    }

    /// Sentence embedding (128-d, normalized) — requires an embedding head in
    /// the archive; the published base carries none, so this returns `None`
    /// unless a tuned archive ships the head.
    pub fn embed(&self, text: &str) -> Option<Vec<f32>> {
        let mut ids = vec![needle_tokenizer::BOS_ID];
        ids.extend(self.tokenizer.encode(text));
        self.engine().embed(&ids)
    }

    /// Rewind the conversation, keep the tools.
    pub fn reset(&mut self) {
        self.cache = self.model.new_cache();
        self.history.clear();
        self.seen_years.clear();
    }
}

/// Prefix the local `date:` fact unless the text already carries one
/// (ports `_with_date_fact`).
pub fn with_date_fact(system: &str) -> String {
    if system.contains("date:") || contains_iso_stamp(system) {
        return system.to_string();
    }
    let now = chrono::Local::now();
    let fact = format!(
        "date: {:04}-{:02}-{:02} {} {:02}:{:02}",
        now.year(),
        now.month(),
        now.day(),
        now.format("%a"),
        now.hour(),
        now.minute()
    );
    if system.trim().is_empty() {
        fact
    } else if system.trim_start().starts_with('{') {
        system.to_string()
    } else {
        format!("{fact}; {system}")
    }
}

/// `auto_date=True` semantics: the agent always carries a date fact, even
/// when the caller passes no system string.
fn dated_system(system: Option<String>) -> Option<String> {
    Some(with_date_fact(&system.unwrap_or_default()))
}

fn contains_iso_stamp(s: &str) -> bool {
    // `\d{4}-\d{2}-\d{2}` (date) or `\d{4}-\d{2}-\d{2}T\d{2}:\d{2}` (datetime)
    let b = s.as_bytes();
    for i in 0..b.len().saturating_sub(9) {
        if b[i].is_ascii_digit()
            && i + 10 <= b.len()
            && b[i + 4] == b'-'
            && b[i + 7] == b'-'
            && b[i + 1..i + 4].iter().all(|c| c.is_ascii_digit())
            && b[i + 5..i + 7].iter().all(|c| c.is_ascii_digit())
            && b[i + 8..i + 10].iter().all(|c| c.is_ascii_digit())
        {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn models_dir() -> PathBuf {
        std::env::var("NEEDLE_MODELS_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models"))
    }

    #[test]
    fn auto_date_fact_with_no_system() {
        // `auto_date=True` parity: a missing system string still yields the date fact
        assert!(dated_system(None).unwrap().starts_with("date: 20"));
        let s = dated_system(Some("locale: en-US".into())).unwrap();
        assert!(s.starts_with("date: 20"));
        assert!(s.ends_with("locale: en-US"));
        let path = models_dir().join("needle3.cact");
        if !path.exists() {
            eprintln!("skip: {} not found", path.display());
            return;
        }
        let agent = Needle::new(&path, Vec::new(), None).unwrap();
        assert!(
            agent.system.as_deref().unwrap().starts_with("date: "),
            "date fact must be injected when system is None"
        );
    }

    #[test]
    fn followup_turn_keeps_system_block() {
        // bug regression: feed_result must re-attach the system block so the
        // follow-up render carries the environment facts (and the date fact)
        let path = models_dir().join("needle3.cact");
        if !path.exists() {
            eprintln!("skip: {} not found", path.display());
            return;
        }
        let mut agent = Needle::new(&path, Vec::new(), Some("locale: en-US".into())).unwrap();
        agent.complete_limited("hello", 64).unwrap();
        let before = agent.history.len();
        agent.feed_result("[{\"ok\": true}]", 64).unwrap();
        let turn = agent.tokenizer.decode(&agent.history[before..]);
        assert!(
            turn.contains("locale: en-US"),
            "follow-up turn lost the system block: {turn}"
        );
    }
}
