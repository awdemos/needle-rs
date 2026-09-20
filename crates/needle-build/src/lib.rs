//! Export a training checkpoint (+ optional LoRA adapter, optional ladder
//! rung) to a `.cact` archive — a port of `needle/model/export.py::_pack_cact`
//! and `needle/model/finetune.py` (rung/merge).

mod perms;
pub mod safetensors;

pub use perms::hada_perms;
pub use safetensors::Safetensors;

use half::f16;
use needle_format::{walsh_hadamard, Config, DTYPE_CQ, DTYPE_FP16, DTYPE_FP32, DTYPE_RAW, TAG_V3, TERNARY_RECORD_BITS};

const CB_BITS: [u32; 3] = [2, 3, 4];

/// A training checkpoint: named f32 tensors + model config.
pub struct Checkpoint {
    pub config: Config,
    pub tensors: Safetensors,
}

/// Parse a `TransformerConfig`-shaped JSON into the header `Config`.
pub fn config_from_json(v: &serde_json::Value) -> Config {
    let u = |k: &str, d: usize| v.get(k).and_then(|x| x.as_u64()).map(|n| n as usize).unwrap_or(d);
    let f = |k: &str, d: f32| v.get(k).and_then(|x| x.as_f64()).map(|n| n as f32).unwrap_or(d);
    let us = |k: &str| {
        v.get(k)
            .and_then(|x| x.as_array())
            .map(|a| a.iter().filter_map(|n| n.as_u64().map(|m| m as usize)).collect())
            .unwrap_or_default()
    };
    let mut orders: Vec<usize> = us("engram_orders");
    if orders.is_empty() {
        orders = vec![2, 3]; // TransformerConfig default
    }
    let global = us("global_layers");
    let num_layers = u("num_layers", 20);
    let _gmask: u64 = global.iter().filter(|&&g| g < 64).map(|&g| 1u64 << g).sum();
    Config {
        vocab_size: u("vocab_size", 8192),
        out_vocab: u("out_vocab", 0),
        d_model: u("d_model", 768),
        num_heads: u("num_heads", 12),
        num_kv_heads: u("num_kv_heads", 2),
        num_layers,
        qk_head_dim: u("qk_head_dim", 48),
        v_head_dim: u("v_head_dim", 64),
        max_seq_len: u("max_seq_len", 8192),
        hada_n: 0, // filled below
        mhc_lanes: u("mhc_lanes", 4),
        sliding_window: u("sliding_window", 1024),
        global_layers: global,
        qkv_conv_taps: u("qkv_conv_taps", 3),
        engram_slots: u("engram_slots", 18432),
        engram_sub_dim: u("engram_sub_dim", 128),
        num_engram_tables: 0, // filled below
        engram_conv_taps: 4,
        engram_conv_dilation: orders.iter().copied().max().unwrap_or(3),
        engram_seed_heads: u("engram_seed_heads", 0),
        engram_orders: orders.clone(),
        engram_layers: us("engram_layers"),
        rope_theta: f("rope_theta", 100000.0),
        kv_window: u("kv_window", 0),
        kv_bits: u("kv_bits", 8) as u32,
    }
    
}

/// Fill derived geometry (hada_n, engram tables) like `export._geometry`.
pub fn with_geometry(mut this: Config) -> Config {
    if this.hada_n == 0 {
        this.hada_n = 1 << ((this.d_model - 1).ilog2() as usize + 1);
    }
    if this.num_engram_tables == 0 && !this.engram_layers.is_empty() {
        let orders_len = this.engram_orders.len().max(1);
        let heads = (this.d_model / (orders_len * 128)).max(1);
        this.num_engram_tables = orders_len * heads;
        if this.engram_sub_dim == 0 {
            this.engram_sub_dim = this.d_model / (orders_len * heads);
        }
    }
    gmask_check(&this);
    this
}

fn gmask_check(_c: &Config) {}

impl Checkpoint {
    pub fn load(path: &std::path::Path) -> Result<Checkpoint, String> {
        let tensors = Safetensors::read(path)?;
        let config_json: serde_json::Value = tensors
            .metadata
            .get("config")
            .map(|s| serde_json::from_str(s).unwrap_or(serde_json::Value::Null))
            .unwrap_or(serde_json::Value::Null);
        let config = with_geometry(config_from_json(&config_json));
        Ok(Checkpoint { config, tensors })
    }

    /// `merge_lora`: W += scale * A @ B for each adapter pair.
    pub fn merge_lora(&mut self, adapter: &Safetensors, scale: f32) -> Result<usize, String> {
        // collect the targeted weight paths first so a B side without its A
        // (or an adapter with no lora tensors at all) is a hard error
        let mut targets: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
        for name in adapter.tensors.keys() {
            let Some((path, which)) = name.rsplit_once('/') else { continue };
            if !path.starts_with("lora/") || (which != "A" && which != "B") {
                continue;
            }
            targets.insert(path.strip_prefix("lora/").unwrap());
        }
        if targets.is_empty() {
            return Err("adapter carries no lora/<target>/{A,B} tensors".into());
        }
        let mut merged = 0;
        for target in targets {
            let a_name = format!("lora/{target}/A");
            let b_name = format!("lora/{target}/B");
            let a = adapter
                .get(&a_name)
                .ok_or_else(|| format!("adapter is missing the A side ({a_name})"))?
                .to_vec();
            let b = adapter
                .get(&b_name)
                .ok_or_else(|| format!("adapter is missing the B side ({b_name})"))?
                .to_vec();
            let shape = self
                .tensors
                .shape(target)
                .ok_or_else(|| format!("adapter target {target} not in checkpoint"))?
                .to_vec();
            if shape.len() < 2 {
                return Err(format!("adapter target {target} must be at least 2-D"));
            }
            let w = self
                .tensors
                .tensors
                .get_mut(target)
                .ok_or_else(|| format!("adapter target {target} not in checkpoint"))?;
            // W (..., in, out) += scale * A (..., in, r) @ B (..., r, out)
            let lead: usize = shape[..shape.len() - 2].iter().product();
            let in_dim = shape[shape.len() - 2];
            let out_dim = shape[shape.len() - 1];
            if w.len() != lead * in_dim * out_dim {
                return Err(format!("checkpoint tensor {target} is corrupt ({} elems, expect {lead}x{in_dim}x{out_dim})", w.len()));
            }
            let rank = adapter
                .shape(&a_name)
                .and_then(|s| s.last().copied())
                .ok_or_else(|| format!("adapter tensor {a_name} has no shape"))?;
            if a.len() != lead * in_dim * rank {
                return Err(format!("{a_name}: shape mismatch ({} elems, expect {lead}x{in_dim}x{rank})", a.len()));
            }
            if b.len() != lead * rank * out_dim {
                return Err(format!("{b_name}: shape mismatch ({} elems, expect {lead}x{rank}x{out_dim})", b.len()));
            }
            for l in 0..lead {
                let wb = &mut w[l * in_dim * out_dim..(l + 1) * in_dim * out_dim];
                for i in 0..in_dim {
                    for r in 0..rank {
                        let av = a[l * in_dim * rank + i * rank + r];
                        if av == 0.0 {
                            continue;
                        }
                        let bv = &b[l * rank * out_dim + r * out_dim..(r + 1) * out_dim];
                        let wrow = &mut wb[i * out_dim..(i + 1) * out_dim];
                        for (o, wv) in wrow.iter_mut().enumerate() {
                            *wv += scale * av * bv[o];
                        }
                    }
                }
            }
            merged += 1;
        }
        Ok(merged)
    }
}

/// `_ladder_layer_order`: bisection order of block indices.
pub fn ladder_order(num_layers: usize) -> Vec<usize> {
    assert!(num_layers >= 1);
    if num_layers == 1 {
        return vec![0];
    }
    let mut selected = vec![0usize, num_layers - 1];
    let mut order = selected.clone();
    while order.len() < num_layers {
        selected.sort_unstable();
        let mut best: Option<(usize, usize, usize)> = None;
        for w in selected.windows(2) {
            let (l, r) = (w[0], w[1]);
            if r - l <= 1 {
                continue;
            }
            let gap = r - l;
            match best {
                Some((g, bl, _)) if (g, usize::MAX - bl) >= (gap, usize::MAX - l) => {}
                _ => best = Some((gap, l, r)),
            }
        }
        let (_, l, r) = best.expect("ladder bisection");
        let candidate = (l + r) / 2;
        selected.push(candidate);
        order.push(candidate);
    }
    order
}

/// `ladder_layer_indices`: stable nested indices for a depth rung.
pub fn ladder_layer_indices(num_layers: usize, depth: usize) -> Result<Vec<usize>, String> {
    if depth < 2 || depth > num_layers {
        return Err(format!("ladder depth must be in [2, {num_layers}], got {depth}"));
    }
    let order = ladder_order(num_layers);
    let mut sel: Vec<usize> = order[..depth].to_vec();
    sel.sort_unstable();
    Ok(sel)
}

impl Checkpoint {
    /// Slice to the `depth`-layer rung (`ladder_slice` + `ladder_config`).
    pub fn ladder_slice(&mut self, depth: usize) -> Result<(), String> {
        let num_layers = self.config.num_layers;
        if depth == num_layers {
            return Ok(());
        }
        let selected = ladder_layer_indices(num_layers, depth)?;
        let sel_set: std::collections::BTreeSet<usize> = selected.iter().copied().collect();

        // stack/layers/block/* and stack/mhc_*: axis 0
        for name in self.tensors.tensors.keys().cloned().collect::<Vec<_>>() {
            if name.starts_with("stack/layers/block/") || name.starts_with("stack/mhc_") {
                let shape = self.tensors.shape(&name).unwrap().to_vec();
                if !shape.is_empty() && shape[0] == num_layers {
                    let data = self.tensors.tensors.get(&name).unwrap();
                    let per: usize = shape[1..].iter().product();
                    let mut out = Vec::with_capacity(selected.len() * per);
                    for &l in &selected {
                        out.extend_from_slice(&data[l * per..(l + 1) * per]);
                    }
                    let mut new_shape = shape.clone();
                    new_shape[0] = selected.len();
                    self.tensors.tensors.insert(name.clone(), out);
                    self.tensors.shapes.insert(name, new_shape);
                }
            }
        }
        // heads: probes/gain rows {0} ∪ {l+1}; row_bias cols
        for head in ["embedding_head", "confidence_head", "router_head"] {
            let probes_key = format!("{head}/probes");
            if self.tensors.get(&probes_key).is_none() {
                continue;
            }
            let rows: Vec<usize> = std::iter::once(0)
                .chain(selected.iter().map(|&l| l + 1))
                .collect();
            for sub in ["probes", "gain"] {
                let key = format!("{head}/{sub}");
                let shape = self
                    .tensors
                    .shape(&key)
                    .ok_or_else(|| format!("checkpoint tensor {key} missing"))?
                    .to_vec();
                let data = self
                    .tensors
                    .tensors
                    .get(&key)
                    .ok_or_else(|| format!("checkpoint tensor {key} missing"))?;
                let cols: usize = shape[1..].iter().product();
                let mut out = Vec::with_capacity(rows.len() * cols);
                for &r in &rows {
                    out.extend_from_slice(&data[r * cols..(r + 1) * cols]);
                }
                let mut ns = shape.clone();
                ns[0] = rows.len();
                self.tensors.tensors.insert(key.clone(), out);
                self.tensors.shapes.insert(key, ns);
            }
            // row_bias (q, L+1, k): keep cols
            let rb_key = format!("{head}/row_bias");
            let shape = self
                .tensors
                .shape(&rb_key)
                .ok_or_else(|| format!("checkpoint tensor {rb_key} missing"))?
                .to_vec();
            if shape.len() != 3 {
                return Err(format!("checkpoint tensor {rb_key} has shape {shape:?}, expected [q, L+1, k]"));
            }
            let data = self
                .tensors
                .tensors
                .get(&rb_key)
                .ok_or_else(|| format!("checkpoint tensor {rb_key} missing"))?;
            let q = shape[0];
            let l1 = shape[1];
            let k = shape[2];
            let mut out = Vec::with_capacity(q * rows.len() * k);
            for qi in 0..q {
                for &r in &rows {
                    out.extend_from_slice(&data[(qi * l1 + r) * k..(qi * l1 + r + 1) * k]);
                }
            }
            self.tensors
                .tensors
                .insert(rb_key.clone(), out);
            self.tensors
                .shapes
                .insert(rb_key, vec![q, rows.len(), k]);
        }
        // engram sites: keep sites on selected layers, rename compactly
        let sites = self.config.engram_layers.clone();
        let mut new_sites = Vec::new();
        let mut mapping: Vec<(usize, usize)> = Vec::new(); // (old site, new site)
        for (s, &layer) in sites.iter().enumerate() {
            if sel_set.contains(&layer) {
                mapping.push((s, new_sites.len()));
                new_sites.push(mapping.len() - 1);
            }
        }
        let mut renamed: Vec<(String, Vec<f32>, Vec<usize>)> = Vec::new();
        for (old, new) in mapping {
            for sub in ["embedding", "key_proj/kernel", "value_proj/kernel", "taps"] {
                let old_key = format!("engrams_{old}/{sub}");
                if let Some(data) = self.tensors.tensors.remove(&old_key) {
                    let shape = self.tensors.shapes.remove(&old_key).unwrap();
                    renamed.push((format!("engrams_{new}/{sub}"), data, shape));
                }
            }
        }
        // drop leftover old sites
        let old_count = sites.len();
        for s in 0..old_count {
            for sub in ["embedding", "key_proj/kernel", "value_proj/kernel", "taps"] {
                self.tensors.tensors.remove(&format!("engrams_{s}/{sub}"));
                self.tensors.shapes.remove(&format!("engrams_{s}/{sub}"));
            }
        }
        for (k, v, s) in renamed {
            self.tensors.shapes.insert(k.clone(), s);
            self.tensors.tensors.insert(k, v);
        }
        self.config.engram_layers = new_sites;
        self.config.num_layers = depth;
        // global layers remap
        let remap: std::collections::HashMap<usize, usize> = selected
            .iter()
            .enumerate()
            .map(|(i, &l)| (l, i))
            .collect();
        self.config.global_layers = self
            .config
            .global_layers
            .iter()
            .filter_map(|l| remap.get(l).copied())
            .collect();
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// export
// ---------------------------------------------------------------------------

fn erf(x: f64) -> f64 {
    // Abramowitz & Stegun 7.1.26 (|err| < 1.5e-7)
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    sign * y
}

fn norm_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

fn norm_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (std::f64::consts::TAU).sqrt()
}

/// `_lloyd_max_gaussian` by exact bin-wise conditional means.
fn lloyd_max_gaussian(bits: u32) -> Vec<f32> {
    if bits == 1 {
        let c = (2.0f64 / std::f64::consts::PI).sqrt();
        return vec![-c as f32, c as f32];
    }
    let levels = 1usize << bits;
    // find the standard-normal quantile at p by bisection
    let quantile = |p: f64| {
        let (mut lo, mut hi) = (-10.0, 10.0);
        for _ in 0..200 {
            let mid = (lo + hi) / 2.0;
            if norm_cdf(mid) < p {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        (lo + hi) / 2.0
    };
    let mut out = Vec::with_capacity(levels);
    for k in 0..levels {
        let a = quantile(k as f64 / levels as f64);
        let b = quantile((k + 1) as f64 / levels as f64);
        let centroid = if k == 0 || k == levels - 1 {
            // truncated tail: mean = φ(a)/Φ(a) style
            (norm_pdf(a) - norm_pdf(b)) / (norm_cdf(b) - norm_cdf(a))
        } else {
            (norm_pdf(a) - norm_pdf(b)) / (norm_cdf(b) - norm_cdf(a))
        };
        out.push(centroid as f32);
    }
    out
}

fn codebook(bits: u32, group: usize) -> Vec<f32> {
    let inv = 1.0 / (group as f32).sqrt();
    lloyd_max_gaussian(bits).iter().map(|v| v * inv).collect()
}

/// `_cq_pack` for a `[out, in]` f32 matrix.
fn cq_pack(w: &[f32], out: usize, inp: usize, bits: u32, group: usize, cb: &[f32]) -> CqMat {
    let h = walsh_hadamard(group);
    let in_pad = inp.div_ceil(group) * group;
    let groups = in_pad / group;
    let mut packed = vec![0u8; out * in_pad * bits as usize / 8];
    let mut norms = vec![f16::ZERO; out * groups];
    let mut tmp = vec![0.0f32; group];
    for row in 0..out {
        for g in 0..groups {
            for (k, slot) in tmp.iter_mut().enumerate().take(group) {
                let c = g * group + k;
                *slot = if c < inp { w[row * inp + c] } else { 0.0 };
            }
            // rot = tmp @ H (symmetric)
            let mut rot = tmp.clone();
            fwht_apply(&mut rot, &h, group);
            let norm: f32 = rot.iter().map(|v| v * v).sum::<f32>().sqrt();
            norms[row * groups + g] = f16::from_f32(norm);
            let inv = if norm > 1e-12 { 1.0 / norm } else { 0.0 };
            for (k, &rv) in rot.iter().enumerate().take(group) {
                let unit = rv * inv;
                // nearest codebook entry, ties to the smaller index
                let mut best = 0usize;
                let mut bd = f32::MAX;
                for (ci, &c) in cb.iter().enumerate() {
                    let d = (unit - c).abs();
                    if d < bd {
                        bd = d;
                        best = ci;
                    }
                }
                let bitpos = (g * group + k) * bits as usize;
                let base = row * (in_pad * bits as usize / 8);
                let v = best as u64;
                for b in 0..bits as usize {
                    if (v >> b) & 1 == 1 {
                        packed[base + (bitpos + b) / 8] |= 1 << ((bitpos + b) % 8);
                    }
                }
            }
        }
    }
    CqMat {
        packed,
        norms,
        bits: if bits == 3 { 3 } else { bits },
    }
}

fn fwht_apply(x: &mut [f32], h: &[f32], n: usize) {
    let src = x.to_vec();
    for i in 0..n {
        let mut acc = 0.0f32;
        for j in 0..n {
            acc += h[i * n + j] * src[j];
        }
        x[i] = acc;
    }
}

struct CqMat {
    packed: Vec<u8>,
    norms: Vec<f16>,
    bits: u32,
}

struct TensorOut {
    dtype: u8,
    shape: Vec<usize>,
    blob: Vec<u8>,
    group: u32,
    bits: u32,
}

fn fp16_tensor(shape: &[usize], data: &[f32]) -> TensorOut {
    let mut blob = Vec::with_capacity(data.len() * 2);
    for v in data {
        blob.extend_from_slice(&f16::from_f32(*v).to_le_bytes());
    }
    TensorOut { dtype: DTYPE_FP16, shape: shape.to_vec(), blob, group: 0, bits: 0 }
}

fn fp32_tensor(shape: &[usize], data: &[f32]) -> TensorOut {
    let mut blob = Vec::with_capacity(data.len() * 4);
    for v in data {
        blob.extend_from_slice(&v.to_le_bytes());
    }
    TensorOut { dtype: DTYPE_FP32, shape: shape.to_vec(), blob, group: 0, bits: 0 }
}

fn cq_tensor(w: &[f32], out: usize, inp: usize, bits: u32, group: usize) -> TensorOut {
    let cb = codebook(bits, group);
    let m = cq_pack(w, out, inp, bits, group, &cb);
    let mut blob = m.packed;
    for n in &m.norms {
        blob.extend_from_slice(&n.to_le_bytes());
    }
    TensorOut { dtype: DTYPE_CQ, shape: vec![out, inp], blob, group: group as u32, bits: m.bits }
}

fn transpose(w: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; w.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = w[r * cols + c];
        }
    }
    out
}

impl Checkpoint {
    /// Write the `.cact` archive. `tokenizer_blob` is the RAW tokenizer
    /// attachment (copy it from the base archive so the export is self-contained).
    pub fn write_cact(
        &self,
        out_path: &std::path::Path,
        tokenizer_blob: Option<&[u8]>,
        bits: u32,
        group: usize,
    ) -> Result<(), String> {
        let c = &self.config;
        let t = &self.tensors;
        let d = c.d_model;
        let taps_n = c.qkv_conv_taps;
        // 5 is the ternary-record tag on read (see needle_format::TERNARY_RECORD_BITS);
        // wider widths have no codebook — anything else corrupts silently
        if !(1..=4).contains(&bits) {
            return Err(format!("CQ width bits must be in 1..=4, got {bits}"));
        }
        let mut ts: Vec<TensorOut> = Vec::new();

        // embedding [vocab, d]
        let emb = t.get("embedding/embedding").ok_or("embedding missing")?;
        ts.push(cq_tensor(emb, c.vocab_size, d, bits, group));

        // per layer
        let base = "stack/layers/block";
        for l in 0..c.num_layers {
            let at = |suffix: &str| format!("{base}/{suffix}");
            let li = |suffix: &str| -> Result<Vec<f32>, String> {
                let full = at(suffix);
                let shape = t
                    .shape(&full)
                    .ok_or_else(|| format!("checkpoint tensor {full} missing"))?;
                if shape.is_empty() || shape[0] != c.num_layers {
                    return Err(format!("checkpoint tensor {full} has shape {shape:?}, expected {} leading layers", c.num_layers));
                }
                let per: usize = shape[1..].iter().product::<usize>();
                let data = t
                    .get(&full)
                    .ok_or_else(|| format!("checkpoint tensor {full} missing"))?;
                if data.len() != shape.iter().product::<usize>() {
                    return Err(format!("checkpoint tensor {full} is corrupt ({} elems, shape {shape:?})", data.len()));
                }
                Ok(data[l * per..(l + 1) * per].to_vec())
            };
            ts.push(fp16_tensor(&[d], &li("ZCRMSNorm_0/scale")?));
            for proj in ["q_proj", "k_proj", "v_proj"] {
                let w = li(&format!("self_attn/{proj}/kernel"))?; // (in, out)
                let out_dim = match proj {
                    "q_proj" => c.num_heads * c.qk_head_dim,
                    "k_proj" => c.num_kv_heads * c.qk_head_dim,
                    _ => c.num_kv_heads * c.v_head_dim,
                };
                let wt = transpose(&w, d, out_dim);
                ts.push(cq_tensor(&wt, out_dim, d, bits, group));
            }
            if taps_n > 0 {
                let qh = c.num_heads * c.qk_head_dim;
                let kh = c.num_kv_heads * c.qk_head_dim;
                let vh = c.num_kv_heads * c.v_head_dim;
                ts.push(fp16_tensor(&[taps_n, qh], &li("self_attn/q_taps")?));
                ts.push(fp16_tensor(&[taps_n, kh], &li("self_attn/k_taps")?));
                ts.push(fp16_tensor(&[taps_n, vh], &li("self_attn/v_taps")?));
            }
            ts.push(fp16_tensor(&[c.qk_head_dim], &li("self_attn/q_norm/scale")?));
            ts.push(fp16_tensor(&[c.qk_head_dim], &li("self_attn/k_norm/scale")?));
            let gout = c.num_heads * c.v_head_dim;
            let gate = li("self_attn/gate_proj/kernel")?;
            ts.push(cq_tensor(&transpose(&gate, d, gout), gout, d, bits, group));
            let op = li("self_attn/out_proj/kernel")?;
            ts.push(cq_tensor(&transpose(&op, gout, d), d, gout, bits, group));
            ts.push(fp16_tensor(&[d], &li("post_attn_norm/scale")?));
            let ag = li("attn_gate")?;
            ts.push(fp16_tensor(&[1], &ag[..1.min(ag.len())]));
            ts.push(fp16_tensor(&[d], &li("pre_hada_norm/scale")?));
            for diag in ["d1", "d2", "b2", "d3", "d4"] {
                ts.push(fp16_tensor(&[c.hada_n], &li(&format!("hadamard_mlp/{diag}"))?));
            }
            for w in ["w1a", "w1b", "w2a", "w2b", "w3a", "w3b"] {
                let data = li(&format!("hadamard_mlp/{w}"))?;
                let dim = (data.len() as f64).sqrt() as usize;
                ts.push(fp16_tensor(&[dim, dim], &data));
            }
            let cv = li("hadamard_mlp/cond_v")?;
            ts.push(fp16_tensor(&[d, 8], &cv));
            let cu = li("hadamard_mlp/cond_u")?;
            ts.push(fp16_tensor(&[8, c.hada_n], &cu));
        }

        // mhc
        for name in ["mhc_a_pre", "mhc_a_post", "mhc_a_res", "mhc_b_pre", "mhc_b_post", "mhc_b_res"] {
            let data = t
                .get(&format!("stack/{name}"))
                .ok_or_else(|| format!("checkpoint tensor stack/{name} missing"))?;
            let shape = t
                .shape(&format!("stack/{name}"))
                .ok_or_else(|| format!("checkpoint tensor stack/{name} missing"))?;
            ts.push(fp16_tensor(shape, data));
        }
        let n = c.mhc_lanes;
        let nc = n * d;
        for name in ["mhc_phi_pre", "mhc_phi_post", "mhc_phi_res"] {
            let key = format!("stack/{name}");
            let data = t
                .get(&key)
                .ok_or_else(|| format!("checkpoint tensor {key} missing"))?; // (L, nC, X)
            let shape = t
                .shape(&key)
                .ok_or_else(|| format!("checkpoint tensor {key} missing"))?;
            if shape.len() != 3 || shape[0] != c.num_layers || shape[1] != nc {
                return Err(format!("checkpoint tensor {key} has shape {shape:?}, expected [{}, {nc}, X]", c.num_layers));
            }
            let x = shape[2];
            // export: transpose(0,2,1).reshape(L*X, nC)
            let mut out = vec![0.0f32; data.len()];
            for l in 0..c.num_layers {
                for i in 0..x {
                    for cc in 0..nc {
                        out[(l * x + i) * nc + cc] = data[(l * nc + cc) * x + i];
                    }
                }
            }
            ts.push(cq_tensor(&out, c.num_layers * x, nc, bits, group));
        }

        // hada perms
        let split = false; // ladder widths are not exported in this port
        let (p1, p2) = hada_perms(c.hada_n, split);
        ts.push(fp32_tensor(&[c.hada_n], &p1.iter().map(|&v| v as f32).collect::<Vec<_>>()));
        ts.push(fp32_tensor(&[c.hada_n], &p2.iter().map(|&v| v as f32).collect::<Vec<_>>()));

        // engram sites
        let num_tables = c.num_engram_tables;
        let sub = c.engram_sub_dim;
        for s in 0..c.engram_layers.len() {
            let tables = t
                .get(&format!("engrams_{s}/embedding"))
                .ok_or_else(|| format!("checkpoint tensor engrams_{s}/embedding missing"))?;
            ts.push(cq_tensor(tables, num_tables * c.engram_slots, sub, bits, group));
            for proj in ["key_proj", "value_proj"] {
                let key = format!("engrams_{s}/{proj}/kernel");
                let w = t
                    .get(&key)
                    .ok_or_else(|| format!("checkpoint tensor {key} missing"))?;
                let wt = transpose(w, num_tables * sub, d);
                ts.push(cq_tensor(&wt, d, num_tables * sub, bits, group));
            }
            let taps = t
                .get(&format!("engrams_{s}/taps"))
                .ok_or_else(|| format!("checkpoint tensor engrams_{s}/taps missing"))?;
            ts.push(fp16_tensor(&[4, d], taps));
        }

        ts.push(fp16_tensor(
            &[d],
            t.get("stack/final_norm/scale")
                .ok_or("checkpoint tensor stack/final_norm/scale missing")?,
        ));

        // heads (only those present in the checkpoint)
        let head_codes: Vec<f32> = [("embedding_head", 1.0), ("confidence_head", 2.0), ("router_head", 3.0)]
            .iter()
            .filter(|(name, _)| t.get(&format!("{name}/probes")).is_some())
            .map(|(_, code)| *code)
            .collect();
        if !head_codes.is_empty() {
            ts.push(fp16_tensor(&[head_codes.len()], &head_codes));
            let rows = c.num_layers + 1;
            for (name, _) in [("embedding_head", 1.0), ("confidence_head", 2.0), ("router_head", 3.0)] {
                if t.get(&format!("{name}/probes")).is_none() {
                    continue;
                }
                let probes = t
                    .get(&format!("{name}/probes"))
                    .ok_or_else(|| format!("checkpoint tensor {name}/probes missing"))?; // (rows, k, d)
                let k = probes.len() / (rows * d);
                ts.push(cq_tensor(probes, rows * k, d, bits, group));
                let gain = t
                    .get(&format!("{name}/gain"))
                    .ok_or_else(|| format!("checkpoint tensor {name}/gain missing"))?;
                ts.push(fp16_tensor(&[rows, k], gain));
                let query = t
                    .get(&format!("{name}/query"))
                    .ok_or_else(|| format!("checkpoint tensor {name}/query missing"))?;
                let q = query.len() / d;
                ts.push(cq_tensor(query, q, d, bits, group));
                let rb = t
                    .get(&format!("{name}/row_bias"))
                    .ok_or_else(|| format!("checkpoint tensor {name}/row_bias missing"))?;
                ts.push(fp16_tensor(&[q, rows, k], rb));
                // ProbeHead.export: the flax kernel is [q*d, out]; the archive
                // stores row-major [out, q*d] like every other kernel
                let proj = t
                    .get(&format!("{name}/proj/kernel"))
                    .ok_or_else(|| format!("checkpoint tensor {name}/proj/kernel missing"))?;
                let out_dim = proj.len() / (q * d);
                ts.push(cq_tensor(&transpose(proj, q * d, out_dim), out_dim, q * d, bits, group));
                let bias = t
                    .get(&format!("{name}/proj/bias"))
                    .map(|b| b.to_vec())
                    .unwrap_or(vec![0.0; out_dim]);
                ts.push(fp16_tensor(&[out_dim], &bias));
                if name == "router_head" {
                    let cal = t
                        .get("router_head/calibration")
                        .map(|v| v.to_vec())
                        .unwrap_or_else(|| vec![0.9, 0.0, 0.6]);
                    ts.push(fp16_tensor(&[3], &cal));
                }
            }
        }

        if let Some(blob) = tokenizer_blob {
            ts.push(TensorOut { dtype: DTYPE_RAW, shape: vec![], blob: blob.to_vec(), group: 0, bits: 0 });
        }

        // ---- header ----
        let cb_all: Vec<f32> = CB_BITS.iter().flat_map(|&b| codebook(b, group)).collect();
        let gmask: u64 = c.global_layers.iter().filter(|&&g| g < 64).map(|&g| 1u64 << g).sum();
        let mut orders4 = c.engram_orders.clone();
        orders4.resize(4, 0);
        let mut sites16 = c.engram_layers.clone();
        sites16.resize(16, 0);
        let mut hdr: Vec<u8> = Vec::new();
        let u32le = |v: u32, h: &mut Vec<u8>| h.extend_from_slice(&v.to_le_bytes());
        u32le(TAG_V3, &mut hdr);
        u32le(ts.len() as u32, &mut hdr);
        u32le(cb_all.len() as u32, &mut hdr);
        u32le(c.kv_window as u32, &mut hdr);
        u32le(c.kv_bits, &mut hdr);
        u32le(c.vocab_size as u32, &mut hdr);
        u32le(c.out_vocab as u32, &mut hdr);
        u32le(d as u32, &mut hdr);
        u32le(c.num_heads as u32, &mut hdr);
        u32le(c.num_kv_heads as u32, &mut hdr);
        u32le(c.num_layers as u32, &mut hdr);
        u32le(c.qk_head_dim as u32, &mut hdr);
        u32le(c.v_head_dim as u32, &mut hdr);
        u32le(c.max_seq_len as u32, &mut hdr);
        u32le(c.hada_n as u32, &mut hdr);
        u32le(n as u32, &mut hdr);
        u32le(c.sliding_window as u32, &mut hdr);
        u32le((gmask & 0xFFFF_FFFF) as u32, &mut hdr);
        u32le((gmask >> 32) as u32, &mut hdr);
        u32le(taps_n as u32, &mut hdr);
        u32le(c.engram_slots as u32, &mut hdr);
        u32le(sub as u32, &mut hdr);
        u32le(num_tables as u32, &mut hdr);
        u32le(4, &mut hdr); // engram conv taps
        u32le(c.engram_conv_dilation.max(1) as u32, &mut hdr);
        u32le(c.engram_seed_heads as u32, &mut hdr);
        u32le(c.engram_orders.len() as u32, &mut hdr);
        for o in &orders4 {
            u32le(*o as u32, &mut hdr);
        }
        u32le(c.engram_layers.len() as u32, &mut hdr);
        for s in &sites16 {
            u32le(*s as u32, &mut hdr);
        }
        hdr.extend_from_slice(&c.rope_theta.to_le_bytes());
        for v in &cb_all {
            hdr.extend_from_slice(&v.to_le_bytes());
        }

        // ---- directory + blobs ----
        let mut dir = Vec::new();
        let mut blobs = Vec::new();
        let blob_base = (hdr.len() + ts.len() * 44) as u64;
        let mut pos = blob_base;
        for t in &ts {
            let aligned = (pos + 63) & !63;
            blobs.resize((aligned - blob_base) as usize, 0);
            blobs.extend_from_slice(&t.blob);
            let rec_off = aligned;
            pos = aligned + t.blob.len() as u64;
            let mut shape4 = [0u32; 4];
            for (i, s) in t.shape.iter().take(4).enumerate() {
                shape4[i] = *s as u32;
            }
            dir.push((t.dtype, t.shape.len() as u8, shape4, rec_off, t.blob.len() as u64, t.group, t.bits));
        }
        let mut out = hdr;
        for (dtype, ndim, shape4, off, nbytes, group, bits) in dir {
            out.push(dtype);
            out.push(ndim);
            out.extend_from_slice(&[0u8; 2]);
            for s in shape4 {
                out.extend_from_slice(&s.to_le_bytes());
            }
            out.extend_from_slice(&off.to_le_bytes());
            out.extend_from_slice(&nbytes.to_le_bytes());
            out.extend_from_slice(&group.to_le_bytes());
            out.extend_from_slice(&bits.to_le_bytes());
        }
        out.extend_from_slice(&blobs);
        std::fs::write(out_path, &out).map_err(|e| format!("write {}: {e}", out_path.display()))?;
        Ok(())
    }
}

const _: () = {
    let _ = TERNARY_RECORD_BITS;
};

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn st(tensors: &[(&str, Vec<usize>, Vec<f32>)]) -> Safetensors {
        Safetensors {
            tensors: tensors.iter().map(|(k, _, v)| (k.to_string(), v.clone())).collect(),
            shapes: tensors.iter().map(|(k, s, _)| (k.to_string(), s.clone())).collect(),
            metadata: HashMap::new(),
        }
    }

    // a minimal but complete 1-layer checkpoint (d_model=8) with a router head
    fn fake_checkpoint() -> Checkpoint {
        let d = 8usize;
        let ones = |n: usize| (0..n).map(|i| (i as f32 * 0.1).sin()).collect::<Vec<f32>>();
        let mut tensors: Vec<(&str, Vec<usize>, Vec<f32>)> = vec![
            ("embedding/embedding", vec![16, d], ones(16 * d)),
            ("stack/layers/block/ZCRMSNorm_0/scale", vec![1, d], ones(d)),
            ("stack/layers/block/self_attn/q_proj/kernel", vec![1, d, 2], ones(d * 2)),
            ("stack/layers/block/self_attn/k_proj/kernel", vec![1, d, 2], ones(d * 2)),
            ("stack/layers/block/self_attn/v_proj/kernel", vec![1, d, 2], ones(d * 2)),
            ("stack/layers/block/self_attn/q_norm/scale", vec![1, 2], ones(2)),
            ("stack/layers/block/self_attn/k_norm/scale", vec![1, 2], ones(2)),
            ("stack/layers/block/self_attn/gate_proj/kernel", vec![1, d, 2], ones(d * 2)),
            ("stack/layers/block/self_attn/out_proj/kernel", vec![1, 2, d], ones(d * 2)),
            ("stack/layers/block/post_attn_norm/scale", vec![1, d], ones(d)),
            ("stack/layers/block/attn_gate", vec![1], vec![0.5]),
            ("stack/layers/block/pre_hada_norm/scale", vec![1, d], ones(d)),
        ];
        for diag in ["d1", "d2", "b2", "d3", "d4"] {
            tensors.push((Box::leak(format!("stack/layers/block/hadamard_mlp/{diag}").into_boxed_str()), vec![1, 8], ones(8)));
        }
        for w in ["w1a", "w1b", "w2a", "w2b", "w3a", "w3b"] {
            tensors.push((Box::leak(format!("stack/layers/block/hadamard_mlp/{w}").into_boxed_str()), vec![1, 64], ones(64)));
        }
        tensors.push(("stack/layers/block/hadamard_mlp/cond_v", vec![1, 64], ones(64)));
        tensors.push(("stack/layers/block/hadamard_mlp/cond_u", vec![1, 64], ones(64)));
        for name in ["mhc_a_pre", "mhc_a_post", "mhc_a_res", "mhc_b_pre", "mhc_b_post", "mhc_b_res"] {
            tensors.push((Box::leak(format!("stack/{name}").into_boxed_str()), vec![d], ones(d)));
        }
        for name in ["mhc_phi_pre", "mhc_phi_post", "mhc_phi_res"] {
            tensors.push((Box::leak(format!("stack/{name}").into_boxed_str()), vec![1, d, 2], ones(2 * d)));
        }
        tensors.push(("stack/final_norm/scale", vec![d], ones(d)));
        // router head: rows = L+1 = 2, k = 2 probes/row, q = 2 queries, out_dim = 3
        tensors.push(("router_head/probes", vec![2, 2, d], ones(4 * d)));
        tensors.push(("router_head/gain", vec![2, 2], ones(4)));
        tensors.push(("router_head/query", vec![2, d], ones(2 * d)));
        tensors.push(("router_head/row_bias", vec![2, 2, 2], ones(8)));
        let mut kernel = vec![0.0f32; 16 * 3]; // flax layout [q*d, out]
        for i in 0..16 {
            for o in 0..3 {
                kernel[i * 3 + o] = (i * 7 + o * 3) as f32 * 0.11 - 0.9;
            }
        }
        tensors.push(("router_head/proj/kernel", vec![16, 3], kernel));
        tensors.push(("router_head/proj/bias", vec![3], vec![0.1, 0.2, 0.3]));
        tensors.push(("router_head/calibration", vec![3], vec![0.9, 0.0, 0.6]));
        let config = Config {
            vocab_size: 16,
            out_vocab: 0,
            d_model: d,
            num_heads: 1,
            num_kv_heads: 1,
            num_layers: 1,
            qk_head_dim: 2,
            v_head_dim: 2,
            max_seq_len: 64,
            hada_n: 8,
            mhc_lanes: 1,
            sliding_window: 8,
            global_layers: vec![],
            qkv_conv_taps: 0,
            engram_slots: 16,
            engram_sub_dim: 8,
            num_engram_tables: 0,
            engram_conv_taps: 4,
            engram_conv_dilation: 3,
            engram_seed_heads: 0,
            engram_orders: vec![2, 3],
            engram_layers: vec![],
            rope_theta: 10000.0,
            kv_window: 0,
            kv_bits: 8,
        };
        Checkpoint { config, tensors: st(&tensors) }
    }

    fn temp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("needle-build-test-{}-{}", std::process::id(), name))
    }

    #[test]
    fn head_export_roundtrips_out_dim_gt1() {
        // regression: the head proj kernel is flax [q*d, out] and must be
        // transposed to row-major [out, q*d] like every other kernel
        let ckpt = fake_checkpoint();
        let kernel = ckpt.tensors.get("router_head/proj/kernel").unwrap().to_vec();
        let qd = 16usize;
        let out_dim = 3usize;
        let path = temp_path("router.cact");
        ckpt.write_cact(&path, None, 4, 4).unwrap();
        let ar = needle_format::read_archive(&path).unwrap();
        std::fs::remove_file(&path).ok();
        let mat = ar
            .tensors
            .iter()
            .find_map(|t| match t {
                needle_format::Tensor::Cq(m) if m.out == out_dim && m.inp == qd => Some(m.clone()),
                _ => None,
            })
            .expect("router proj [3, 16] CQ tensor");
        let want = cq_pack(&transpose(&kernel, qd, out_dim), out_dim, qd, 4, 4, &codebook(4, 4));
        assert_eq!(mat.bits, 4);
        assert_eq!(mat.packed, want.packed, "packed rows must be the transposed kernel");
        assert_eq!(mat.norms, want.norms);
        // and the pre-fix layout (kernel read as [out, q*d]) must differ, or the
        // test would not catch the regression
        let buggy = cq_pack(&kernel, out_dim, qd, 4, 4, &codebook(4, 4));
        assert_ne!(mat.packed, buggy.packed);
    }

    #[test]
    fn head_export_rejects_bad_bits() {
        let ckpt = fake_checkpoint();
        let path = temp_path("bits5.cact");
        let err = ckpt.write_cact(&path, None, 5, 4).unwrap_err();
        assert!(err.contains("1..=4"), "{err}");
        assert!(!path.exists());
    }

    #[test]
    fn head_export_labels_missing_tensors() {
        let mut ckpt = fake_checkpoint();
        ckpt.tensors.tensors.remove("stack/final_norm/scale");
        ckpt.tensors.shapes.remove("stack/final_norm/scale");
        let err = ckpt.write_cact(&temp_path("missing.cact"), None, 4, 4).unwrap_err();
        assert!(err.contains("stack/final_norm/scale"), "{err}");
    }

    #[test]
    fn merge_lora_merges_and_counts() {
        let w0 = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // [in=2, out=3]
        let mut ckpt = Checkpoint {
            config: fake_checkpoint().config,
            tensors: st(&[("w", vec![2, 3], w0.clone())]),
        };
        let a = vec![1.0f32, 0.0, 0.0, 1.0]; // [in=2, r=2]
        let b = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // [r=2, out=3]
        let adapter = st(&[("lora/w/A", vec![2, 2], a), ("lora/w/B", vec![2, 3], b.clone())]);
        let n = ckpt.merge_lora(&adapter, 2.0).unwrap();
        assert_eq!(n, 1);
        // W += 2 * A @ B = 2 * B (A is the identity)
        let merged = ckpt.tensors.get("w").unwrap();
        for (got, (w, b)) in merged.iter().zip(w0.iter().zip(b.iter())) {
            assert!((got - (w + 2.0 * b)).abs() < 1e-6, "got {got}");
        }
    }

    #[test]
    fn merge_lora_errors_instead_of_panicking() {
        let base = fake_checkpoint();
        let mut ckpt = Checkpoint {
            config: base.config.clone(),
            tensors: st(&[("w", vec![2, 3], vec![0.0; 6])]),
        };
        // adapter targets a tensor the checkpoint does not have (was a panic)
        let adapter = st(&[
            ("lora/absent/A", vec![2, 2], vec![0.0; 4]),
            ("lora/absent/B", vec![2, 3], vec![0.0; 6]),
        ]);
        let err = ckpt.merge_lora(&adapter, 1.0).unwrap_err();
        assert!(err.contains("absent"), "{err}");
        // A side missing (was silently skipped)
        let adapter = st(&[("lora/w/B", vec![2, 3], vec![0.0; 6])]);
        let err = ckpt.merge_lora(&adapter, 1.0).unwrap_err();
        assert!(err.contains("A side"), "{err}");
        // B side missing (was an unlabeled "missing B")
        let adapter = st(&[("lora/w/A", vec![2, 2], vec![0.0; 4])]);
        let err = ckpt.merge_lora(&adapter, 1.0).unwrap_err();
        assert!(err.contains("B side"), "{err}");
        // a safetensors with no lora tensors at all (was Ok(0))
        let adapter = st(&[("unrelated", vec![2], vec![0.0; 2])]);
        let err = ckpt.merge_lora(&adapter, 1.0).unwrap_err();
        assert!(err.contains("no lora/"), "{err}");
        // rank/shape mismatch against the target
        let adapter = st(&[("lora/w/A", vec![3, 2], vec![0.0; 6]), ("lora/w/B", vec![2, 3], vec![0.0; 6])]);
        let err = ckpt.merge_lora(&adapter, 1.0).unwrap_err();
        assert!(err.contains("shape mismatch"), "{err}");
    }
}
