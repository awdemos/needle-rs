//! The Needle 3 "Simple Attention Network" model: weight structures and
//! `.cact` loading. Ported from `needle/model/architecture.py` +
//! `needle/model/export.py` (`_tensors` canonical tensor order).
//!
//! All CQ matrices are dequantized to f32 at load; compute is f32 throughout
//! (the deployed engine runs W4A8; the difference does not change greedy
//! outputs in practice — see README "Porting notes").

mod forward;
mod ops;

pub use forward::{Cache, Cells};
pub use needle_format::Config;

use needle_format::{Archive, Tensor};
use std::fmt;

#[derive(Debug)]
pub enum Error {
    Format(needle_format::Error),
    Missing(&'static str),
    Shape(&'static str),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Format(e) => write!(f, "archive: {e}"),
            Error::Missing(n) => write!(f, "missing tensor {n}"),
            Error::Shape(n) => write!(f, "bad shape for {n}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<needle_format::Error> for Error {
    fn from(e: needle_format::Error) -> Error {
        Error::Format(e)
    }
}

pub type Result<T> = std::result::Result<T, Error>;

const ENGRAM_SEED: u32 = 0x9E37_79B9;
const ENGRAM_PRIME: u32 = 0x0100_0193;

pub struct LayerWeights {
    pub norm_in: Vec<f32>,
    pub q_proj: Vec<f32>,   // [H*qh, d]
    pub k_proj: Vec<f32>,   // [K*qh, d]
    pub v_proj: Vec<f32>,   // [K*vh, d]
    pub q_taps: Vec<f32>,   // [3, H*qh]
    pub k_taps: Vec<f32>,   // [3, K*qh]
    pub v_taps: Vec<f32>,   // [3, K*vh]
    pub q_norm: Vec<f32>,   // [qh]
    pub k_norm: Vec<f32>,   // [qh]
    pub gate_proj: Vec<f32>,// [H*vh, d]
    pub out_proj: Vec<f32>, // [d, H*vh]
    pub post_norm: Vec<f32>,
    pub attn_gate: f32,
    pub pre_hada: Vec<f32>,
    pub d1: Vec<f32>,
    pub d2: Vec<f32>,
    pub b2: Vec<f32>,
    pub d3: Vec<f32>,
    pub d4: Vec<f32>,
    pub w1a: Vec<f32>, // [ba, ba]
    pub w1b: Vec<f32>, // [bb, bb]
    pub w2a: Vec<f32>,
    pub w2b: Vec<f32>,
    pub w3a: Vec<f32>,
    pub w3b: Vec<f32>,
    pub cond_v: Vec<f32>, // [d, 8]
    pub cond_u: Vec<f32>, // [8, hada_n]
}

pub struct MhcWeights {
    pub a_pre: Vec<f32>,  // [L]
    pub a_post: Vec<f32>, // [L]
    pub a_res: Vec<f32>,  // [L]
    pub b_pre: Vec<f32>,  // [L, n]
    pub b_post: Vec<f32>, // [L, n]
    pub b_res: Vec<f32>,  // [L, n, n]
    pub phi_pre: Vec<f32>,// [L, nC, n] (restored from exported [L*n, nC])
    pub phi_post: Vec<f32>,
    pub phi_res: Vec<f32>,// [L, nC, n*n]
}

pub struct EngramWeights {
    pub tables: Vec<f32>,    // [num_tables, slots, sub_dim]
    pub key_proj: Vec<f32>,  // [d, num_tables*sub_dim]
    pub value_proj: Vec<f32>,// [d, num_tables*sub_dim]
    pub taps: Vec<f32>,      // [4, d]
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeadKind {
    Embedding,
    Confidence,
    Router,
}

impl HeadKind {
    pub fn from_code(code: u16) -> Option<HeadKind> {
        match code {
            1 => Some(HeadKind::Embedding),
            2 => Some(HeadKind::Confidence),
            3 => Some(HeadKind::Router),
            _ => None,
        }
    }
    pub fn code(self) -> u16 {
        match self {
            HeadKind::Embedding => 1,
            HeadKind::Confidence => 2,
            HeadKind::Router => 3,
        }
    }
}

pub struct HeadWeights {
    pub kind: HeadKind,
    pub k: usize,             // probes per row
    pub q: usize,             // queries
    pub probes: Vec<f32>,     // [(L+1)*k, d]
    pub gain: Vec<f32>,       // [L+1, k]
    pub query: Vec<f32>,      // [q, d]
    pub row_bias: Vec<f32>,   // [q, L+1, k]
    pub proj: Vec<f32>,       // [out, q*d]
    pub bias: Vec<f32>,       // [out]
    pub calibration: Option<Vec<f32>>, // router: [3]
}

pub struct Model {
    pub config: Config,
    pub embedding: Vec<f32>, // [vocab, d]
    pub layers: Vec<LayerWeights>,
    pub mhc: MhcWeights,
    pub hada_p1: Vec<f32>,
    pub hada_p2: Vec<f32>,
    pub engrams: Vec<EngramWeights>,
    pub final_norm: Vec<f32>,
    pub heads: Vec<HeadWeights>,
    /// rows past `out_vocab` are input-only code embeddings (0 = full vocab).
    pub out_vocab: usize,
    pub embed_scale: f32,
    pub rope: ops::Rope,
}

struct Cursor<'a> {
    tensors: &'a [Tensor],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn next(&mut self, what: &'static str) -> Result<&'a Tensor> {
        let t = self.tensors.get(self.pos).ok_or(Error::Missing(what))?;
        self.pos += 1;
        Ok(t)
    }

    fn f32s(&mut self, what: &'static str, ar: &Archive) -> Result<(Vec<usize>, Vec<f32>)> {
        match self.next(what)? {
            Tensor::Fp16 { shape, data } | Tensor::Fp32 { shape, data } => {
                Ok((shape.clone(), data.clone()))
            }
            Tensor::Cq(m) => {
                let cb = ar.codebook_for(m.bits, m.group_size)?;
                Ok((vec![m.out, m.inp], needle_format::dequant_cq(m, &cb)?))
            }
            _ => Err(Error::Shape(what)),
        }
    }

    fn mat(&mut self, what: &'static str, ar: &Archive) -> Result<Vec<f32>> {
        Ok(self.f32s(what, ar)?.1)
    }

    fn vec1(&mut self, what: &'static str, ar: &Archive) -> Result<Vec<f32>> {
        let (shape, data) = self.f32s(what, ar)?;
        // the directory shape must describe exactly the stored elements
        // (catches truncated blobs and unexpected ranks at the source)
        if !shape.is_empty() && shape.iter().product::<usize>() != data.len() {
            return Err(Error::Shape(what));
        }
        Ok(data)
    }
}

/// Reject header geometries the forward pass cannot execute and the Python
/// reference never produces: `architecture.py` asserts the engram-layer range,
/// `export.py` hardwires `engram_conv_dilation = max(engram_orders)`, and the
/// GQA/engram/MHC divisors below are assumed throughout `forward.rs`.
fn validate_geometry(cfg: &Config) -> Result<()> {
    if cfg.num_kv_heads == 0 || cfg.num_kv_heads > cfg.num_heads {
        return Err(Error::Shape("num_kv_heads must be in 1..=num_heads"));
    }
    if !cfg.num_heads.is_multiple_of(cfg.num_kv_heads) {
        return Err(Error::Shape("num_heads must be divisible by num_kv_heads"));
    }
    if cfg.hada_n == 0 {
        return Err(Error::Shape("hada_n must be > 0"));
    }
    if cfg.mhc_lanes == 0 {
        return Err(Error::Shape("mhc_lanes must be > 0"));
    }
    if cfg.engram_orders.is_empty() {
        return Err(Error::Shape("engram_orders must be non-empty"));
    }
    let orders_n = cfg.engram_orders.len();
    if cfg.num_engram_tables == 0 || !cfg.num_engram_tables.is_multiple_of(orders_n) {
        return Err(Error::Shape(
            "num_engram_tables must be a positive multiple of len(engram_orders)",
        ));
    }
    if cfg.engram_slots == 0 {
        return Err(Error::Shape("engram_slots must be > 0"));
    }
    if cfg.engram_conv_taps == 0 {
        return Err(Error::Shape("engram_conv_taps must be > 0"));
    }
    if cfg.engram_layers.iter().any(|&l| l >= cfg.num_layers) {
        return Err(Error::Shape("engram_layers contains an index >= num_layers"));
    }
    let max_order = cfg.engram_orders.iter().copied().max().unwrap_or(1);
    if cfg.engram_conv_dilation != max_order {
        return Err(Error::Shape("engram_conv_dilation must equal max(engram_orders)"));
    }
    Ok(())
}

/// `len == want` or a descriptive `Error::Shape(what)`.
fn expect_len(what: &'static str, len: usize, want: usize) -> Result<()> {
    if len != want {
        return Err(Error::Shape(what));
    }
    Ok(())
}

/// Every entry an integer in `0..n` and each index hit exactly once.
fn expect_perm(what: &'static str, perm: &[f32], n: usize) -> Result<()> {
    if perm.len() != n {
        return Err(Error::Shape(what));
    }
    let mut seen = vec![false; n];
    for &v in perm {
        if v.fract() != 0.0 || v < 0.0 {
            return Err(Error::Shape(what));
        }
        let i = v as usize;
        if i >= n || seen[i] {
            return Err(Error::Shape(what));
        }
        seen[i] = true;
    }
    Ok(())
}

impl Model {
    pub fn from_archive(ar: &Archive) -> Result<Model> {
        validate_geometry(&ar.config)?;
        let cfg = &ar.config;
        let d = cfg.d_model;
        let mut cur = Cursor {
            tensors: &ar.tensors,
            pos: 0,
        };

        let embedding = cur.mat("embedding", ar)?;
        if embedding.len() != cfg.vocab_size * d {
            return Err(Error::Shape("embedding"));
        }

        let taps_n = cfg.qkv_conv_taps;
        let h = cfg.num_heads;
        let kvh = cfg.num_kv_heads;
        let qh = cfg.qk_head_dim;
        let vh = cfg.v_head_dim;
        let hada_n = cfg.hada_n;
        let (ba, bb) = Config::hada_blocks(hada_n);
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for _ in 0..cfg.num_layers {
            let norm_in = cur.vec1("norm_in", ar)?;
            let q_proj = cur.mat("q_proj", ar)?;
            let k_proj = cur.mat("k_proj", ar)?;
            let v_proj = cur.mat("v_proj", ar)?;
            let (q_taps, k_taps, v_taps) = if taps_n > 0 {
                (
                    cur.vec1("q_taps", ar)?,
                    cur.vec1("k_taps", ar)?,
                    cur.vec1("v_taps", ar)?,
                )
            } else {
                (Vec::new(), Vec::new(), Vec::new())
            };
            let q_norm = cur.vec1("q_norm", ar)?;
            let k_norm = cur.vec1("k_norm", ar)?;
            let gate_proj = cur.mat("gate_proj", ar)?;
            let out_proj = cur.mat("out_proj", ar)?;
            let post_norm = cur.vec1("post_norm", ar)?;
            let attn_gate_vec = cur.vec1("attn_gate", ar)?;
            let pre_hada = cur.vec1("pre_hada", ar)?;
            let d1 = cur.vec1("d1", ar)?;
            let d2 = cur.vec1("d2", ar)?;
            let b2 = cur.vec1("b2", ar)?;
            let d3 = cur.vec1("d3", ar)?;
            let d4 = cur.vec1("d4", ar)?;
            let w1a = cur.vec1("w1a", ar)?;
            let w1b = cur.vec1("w1b", ar)?;
            let w2a = cur.vec1("w2a", ar)?;
            let w2b = cur.vec1("w2b", ar)?;
            let w3a = cur.vec1("w3a", ar)?;
            let w3b = cur.vec1("w3b", ar)?;
            let cond_v = cur.vec1("cond_v", ar)?;
            let cond_u = cur.vec1("cond_u", ar)?;
            // exact geometry: a short tensor would panic inside a rayon matvec
            expect_len("norm_in", norm_in.len(), d)?;
            expect_len("q_proj", q_proj.len(), h * qh * d)?;
            expect_len("k_proj", k_proj.len(), kvh * qh * d)?;
            expect_len("v_proj", v_proj.len(), kvh * vh * d)?;
            if taps_n > 0 {
                expect_len("q_taps", q_taps.len(), taps_n * h * qh)?;
                expect_len("k_taps", k_taps.len(), taps_n * kvh * qh)?;
                expect_len("v_taps", v_taps.len(), taps_n * kvh * vh)?;
            }
            expect_len("q_norm", q_norm.len(), qh)?;
            expect_len("k_norm", k_norm.len(), qh)?;
            expect_len("gate_proj", gate_proj.len(), h * vh * d)?;
            expect_len("out_proj", out_proj.len(), d * h * vh)?;
            expect_len("post_norm", post_norm.len(), d)?;
            let attn_gate = attn_gate_vec.first().copied().ok_or(Error::Shape("attn_gate"))?;
            expect_len("pre_hada", pre_hada.len(), d)?;
            expect_len("d1", d1.len(), hada_n)?;
            expect_len("d2", d2.len(), hada_n)?;
            expect_len("b2", b2.len(), hada_n)?;
            expect_len("d3", d3.len(), hada_n)?;
            expect_len("d4", d4.len(), hada_n)?;
            expect_len("w1a", w1a.len(), ba * ba)?;
            expect_len("w1b", w1b.len(), bb * bb)?;
            expect_len("w2a", w2a.len(), ba * ba)?;
            expect_len("w2b", w2b.len(), bb * bb)?;
            expect_len("w3a", w3a.len(), ba * ba)?;
            expect_len("w3b", w3b.len(), bb * bb)?;
            expect_len("cond_v", cond_v.len(), d * 8)?;
            expect_len("cond_u", cond_u.len(), 8 * hada_n)?;
            layers.push(LayerWeights {
                norm_in,
                q_proj,
                k_proj,
                v_proj,
                q_taps,
                k_taps,
                v_taps,
                q_norm,
                k_norm,
                gate_proj,
                out_proj,
                post_norm,
                attn_gate,
                pre_hada,
                d1,
                d2,
                b2,
                d3,
                d4,
                w1a,
                w1b,
                w2a,
                w2b,
                w3a,
                w3b,
                cond_v,
                cond_u,
            });
        }

        let l = cfg.num_layers;
        let n = cfg.mhc_lanes;
        let a_pre = cur.vec1("mhc_a_pre", ar)?;
        let a_post = cur.vec1("mhc_a_post", ar)?;
        let a_res = cur.vec1("mhc_a_res", ar)?;
        let b_pre = cur.vec1("mhc_b_pre", ar)?;
        let b_post = cur.vec1("mhc_b_post", ar)?;
        let b_res = cur.vec1("mhc_b_res", ar)?;
        expect_len("mhc_a_pre", a_pre.len(), l)?;
        expect_len("mhc_a_post", a_post.len(), l)?;
        expect_len("mhc_a_res", a_res.len(), l)?;
        expect_len("mhc_b_pre", b_pre.len(), l * n)?;
        expect_len("mhc_b_post", b_post.len(), l * n)?;
        expect_len("mhc_b_res", b_res.len(), l * n * n)?;
        let n_c = n * d;
        let phi_pre = restore_phi(&cur.mat("mhc_phi_pre", ar)?, l, n_c, n)?;
        let phi_post = restore_phi(&cur.mat("mhc_phi_post", ar)?, l, n_c, n)?;
        let phi_res = restore_phi(&cur.mat("mhc_phi_res", ar)?, l, n_c, n * n)?;
        let mhc = MhcWeights {
            a_pre,
            a_post,
            a_res,
            b_pre,
            b_post,
            b_res,
            phi_pre,
            phi_post,
            phi_res,
        };

        let hada_p1 = cur.vec1("hada_p1", ar)?;
        let hada_p2 = cur.vec1("hada_p2", ar)?;
        // permute() indexes by these values; a bad one panics inside the layer
        expect_perm("hada_p1", &hada_p1, hada_n)?;
        expect_perm("hada_p2", &hada_p2, hada_n)?;

        let mut engrams = Vec::with_capacity(cfg.engram_layers.len());
        let table_dim = cfg.num_engram_tables * cfg.engram_sub_dim;
        for _ in 0..cfg.engram_layers.len() {
            let tables = cur.mat("engram.tables", ar)?;
            let key_proj = cur.mat("engram.key_proj", ar)?;
            let value_proj = cur.mat("engram.value_proj", ar)?;
            let taps = cur.vec1("engram.taps", ar)?;
            expect_len(
                "engram.tables",
                tables.len(),
                cfg.num_engram_tables * cfg.engram_slots * cfg.engram_sub_dim,
            )?;
            expect_len("engram.key_proj", key_proj.len(), d * table_dim)?;
            expect_len("engram.value_proj", value_proj.len(), d * table_dim)?;
            expect_len("engram.taps", taps.len(), cfg.engram_conv_taps * d)?;
            engrams.push(EngramWeights {
                tables,
                key_proj,
                value_proj,
                taps,
            });
        }

        let final_norm = cur.vec1("final_norm", ar)?;
        expect_len("final_norm", final_norm.len(), d)?;

        // optional probe heads: a small 1-D FP16 `heads.manifest` right after
        // final_norm, then per-head tensors; the tokenizer RAW (if any) follows.
        let mut heads = Vec::new();
        let is_manifest = matches!(
            ar.tensors.get(cur.pos),
            Some(Tensor::Fp16 { shape, data })
                if shape.len() == 1 && !data.is_empty() && data.len() <= 3
                    && data.iter().all(|&c| HeadKind::from_code(c as u16).is_some())
        );
        if is_manifest {
            let Tensor::Fp16 { data, .. } = cur.next("heads.manifest")? else {
                unreachable!()
            };
            for &c in data {
                let kind = HeadKind::from_code(c as u16).ok_or(Error::Shape("heads.manifest code"))?;
                let rows = l + 1;
                let probes = cur.mat("head.probes", ar)?;
                if probes.is_empty() || probes.len() % (rows * d) != 0 {
                    return Err(Error::Shape("head.probes"));
                }
                let k = probes.len() / (rows * d);
                let gain = cur.vec1("head.gain", ar)?;
                let query = cur.mat("head.query", ar)?;
                if query.is_empty() || query.len() % d != 0 {
                    return Err(Error::Shape("head.query"));
                }
                let q = query.len() / d;
                let row_bias = cur.vec1("head.row_bias", ar)?;
                let proj = cur.mat("head.proj", ar)?;
                if proj.is_empty() || proj.len() % (q * d) != 0 {
                    return Err(Error::Shape("head.proj"));
                }
                let out_dim = proj.len() / (q * d);
                let bias = cur.vec1("head.bias", ar)?;
                let calibration = if kind == HeadKind::Router {
                    let c = cur.vec1("head.calibration", ar)?;
                    expect_len("head.calibration", c.len(), 3)?;
                    Some(c)
                } else {
                    None
                };
                expect_len("head.gain", gain.len(), rows * k)?;
                expect_len("head.row_bias", row_bias.len(), q * rows * k)?;
                expect_len("head.bias", bias.len(), out_dim)?;
                heads.push(HeadWeights {
                    kind,
                    k,
                    q,
                    probes,
                    gain,
                    query,
                    row_bias,
                    proj,
                    bias,
                    calibration,
                });
            }
        }

        Ok(Model {
            embed_scale: (d as f32).sqrt(),
            out_vocab: if cfg.out_vocab > 0 {
                cfg.out_vocab
            } else {
                cfg.vocab_size
            },
            rope: ops::Rope::new(cfg.qk_head_dim, cfg.max_seq_len, cfg.rope_theta),
            config: cfg.clone(),
            embedding,
            layers,
            mhc,
            hada_p1,
            hada_p2,
            engrams,
            final_norm,
            heads,
        })
    }

    pub fn head(&self, kind: HeadKind) -> Option<&HeadWeights> {
        self.heads.iter().find(|h| h.kind == kind)
    }
}

/// Exported phi is `[L*X, nC]` (per layer `[X, nC]`, X = n for pre/post, n*n for
/// res); restore to `[L, nC, X]`.
fn restore_phi(raw: &[f32], l: usize, nc: usize, x: usize) -> Result<Vec<f32>> {
    if raw.len() != l * x * nc {
        return Err(Error::Shape("mhc_phi"));
    }
    let mut out = vec![0.0f32; raw.len()];
    for li in 0..l {
        for i in 0..x {
            for c in 0..nc {
                out[(li * nc + c) * x + i] = raw[(li * x + i) * nc + c];
            }
        }
    }
    Ok(out)
}

/// Engram table indices — the FNV-style n-gram hash from `architecture.py`.
/// Returns `[T, num_tables]` row-major (`num_tables = len(orders) * heads`).
pub fn engram_indices(
    tokens: &[u32],
    orders: &[usize],
    heads: usize,
    slots: usize,
    seed_heads: usize,
) -> Vec<u32> {
    let t_len = tokens.len();
    let num_tables = orders.len() * heads;
    let stride = if seed_heads > 0 { seed_heads } else { heads };
    let mut out = vec![0u32; t_len * num_tables];
    for (oi, &order) in orders.iter().enumerate() {
        for h in 0..heads {
            let col = oi * heads + h;
            let seed = ENGRAM_SEED.wrapping_mul((oi * stride + h + 1) as u32);
            for t in 0..t_len {
                let mut acc = seed;
                for j in 0..order {
                    // shift right by j positions: token at t-j, zero-padded
                    let shifted = if t >= j { tokens[t - j] } else { 0 };
                    acc = (acc ^ shifted).wrapping_mul(ENGRAM_PRIME);
                }
                acc ^= acc >> 15;
                out[t * num_tables + col] = acc % slots as u32;
            }
        }
    }
    out
}
