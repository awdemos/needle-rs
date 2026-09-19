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
        if shape.len() > 1 && shape.iter().product::<usize>() != data.len() {
            return Err(Error::Shape(what));
        }
        Ok(data)
    }
}

impl Model {
    pub fn from_archive(ar: &Archive) -> Result<Model> {
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
            let attn_gate = cur.vec1("attn_gate", ar)?[0];
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

        let mut engrams = Vec::with_capacity(cfg.engram_layers.len());
        for _ in 0..cfg.engram_layers.len() {
            let tables = cur.mat("engram.tables", ar)?;
            let key_proj = cur.mat("engram.key_proj", ar)?;
            let value_proj = cur.mat("engram.value_proj", ar)?;
            let taps = cur.vec1("engram.taps", ar)?;
            engrams.push(EngramWeights {
                tables,
                key_proj,
                value_proj,
                taps,
            });
        }

        let final_norm = cur.vec1("final_norm", ar)?;

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
                let k = probes.len() / (rows * d);
                let gain = cur.vec1("head.gain", ar)?;
                let query = cur.mat("head.query", ar)?;
                let q = query.len() / d;
                let row_bias = cur.vec1("head.row_bias", ar)?;
                let proj = cur.mat("head.proj", ar)?;
                let out_dim = proj.len() / (q * d);
                let bias = cur.vec1("head.bias", ar)?;
                let calibration = if kind == HeadKind::Router {
                    Some(cur.vec1("head.calibration", ar)?)
                } else {
                    None
                };
                debug_assert_eq!(gain.len(), rows * k);
                debug_assert_eq!(row_bias.len(), q * rows * k);
                debug_assert_eq!(bias.len(), out_dim);
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
