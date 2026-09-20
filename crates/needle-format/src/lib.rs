//! Reader for the Needle 3 `.cact` archive — the deployment container the runtime
//! maps and reads. Ported from `needle/model/export.py` (see its module docstring,
//! which fully specifies the byte layout).
//!
//! Layout: a fixed 196-byte header (48 u32 geometry fields + 1 f32 rope_theta),
//! `codebook_len` f32 codebook values (cb2[4] | cb3[8] | cb4[16]), a nameless tensor
//! directory of 44-byte records (`<BBHIIIIQQII`), then 64-byte-aligned tensor blobs.

use half::f16;
use std::fmt;

pub const TAG_V3: u32 = 0x05E12A84;
/// The Needle 2 generation tag. V2 archives use a different layout handled by a
/// different engine generation (see Python `_CACT_GENERATIONS`); this reader
/// rejects them with [`Error::UnsupportedVersion`].
pub const TAG_V2: u32 = 0x05E12A83;
pub const ALIGN: usize = 64;

pub const DTYPE_FP16: u8 = 1;
pub const DTYPE_FP32: u8 = 2;
pub const DTYPE_CQ: u8 = 3;
pub const DTYPE_RAW: u8 = 4;

/// Record bits value that selects the analytic ternary crumb codec.
pub const TERNARY_RECORD_BITS: u32 = 5;

const HEADER_U32S: usize = 48;
const HEADER_LEN: usize = HEADER_U32S * 4 + 4;
/// struct.calcsize("<BBHIIIIQQII")
const REC_LEN: usize = 44;

pub(crate) const CB2: usize = 4;
pub(crate) const CB3: usize = 8;
#[allow(dead_code)]
pub(crate) const CB4: usize = 16;

#[derive(Debug)]
pub enum Error {
    BadTag(u32),
    /// A recognized but unsupported archive generation (V2).
    UnsupportedVersion(u32),
    /// The archive file could not be read at all (missing, permissions, ...).
    Io(std::io::Error),
    Truncated(&'static str),
    BadDtype(u8),
    BadBits(u32),
    BadShape,
    /// A header/record field is outside the range the format can encode.
    BadGeometry(&'static str),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::BadTag(t) => write!(f, "bad archive tag 0x{t:08x}"),
            Error::UnsupportedVersion(t) => write!(
                f,
                "unsupported archive version (tag 0x{t:08x}); this reader only supports .cact V3"
            ),
            Error::Io(e) => write!(f, "cannot read archive: {e}"),
            Error::Truncated(what) => write!(f, "truncated archive: {what}"),
            Error::BadDtype(d) => write!(f, "unsupported record dtype {d}"),
            Error::BadBits(b) => write!(f, "unsupported CQ width bits={b}"),
            Error::BadShape => write!(f, "record shape/ndim mismatch"),
            Error::BadGeometry(what) => write!(f, "bad archive geometry: {what}"),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

/// Model geometry recovered from the `.cact` header.
#[derive(Debug, Clone)]
pub struct Config {
    pub vocab_size: usize,
    pub out_vocab: usize,
    pub d_model: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub num_layers: usize,
    pub qk_head_dim: usize,
    pub v_head_dim: usize,
    pub max_seq_len: usize,
    pub hada_n: usize,
    pub mhc_lanes: usize,
    pub sliding_window: usize,
    pub global_layers: Vec<usize>,
    pub qkv_conv_taps: usize,
    pub engram_slots: usize,
    pub engram_sub_dim: usize,
    pub num_engram_tables: usize,
    pub engram_conv_taps: usize,
    pub engram_conv_dilation: usize,
    pub engram_seed_heads: usize,
    pub engram_orders: Vec<usize>,
    pub engram_layers: Vec<usize>,
    pub rope_theta: f32,
    pub kv_window: usize,
    pub kv_bits: u32,
}

impl Config {
    pub fn head_dims(&self) -> (usize, usize) {
        (self.qk_head_dim, self.v_head_dim)
    }

    /// `_hada_blocks`: split `n` into (ba, bb) Kronecker factors.
    pub fn hada_blocks(n: usize) -> (usize, usize) {
        let n = n.max(1); // a corrupt hada_n = 0 must not underflow
        let bits = (usize::BITS - (n - 1).leading_zeros()) as usize;
        let b = 1usize << (bits / 2);
        (b, n / b)
    }
}

#[derive(Debug, Clone)]
pub struct Record {
    pub dtype: u8,
    pub shape: Vec<usize>,
    pub offset: u64,
    pub nbytes: u64,
    pub group_size: u32,
    pub bits: u32,
}

/// A decoded tensor. CQ matrices stay packed; use [`dequant_cq`].
#[derive(Debug, Clone)]
pub enum Tensor {
    Fp16 { shape: Vec<usize>, data: Vec<f32> },
    Fp32 { shape: Vec<usize>, data: Vec<f32> },
    Cq(CqMatrix),
    /// Opaque attachment (the embedded tokenizer).
    Raw(Vec<u8>),
}

/// A CQ-quantized `[out, in]` matrix: LSB-first packed code indices plus the
/// per-group L2 norms (f16).
#[derive(Debug, Clone)]
pub struct CqMatrix {
    pub out: usize,
    pub inp: usize,
    pub packed: Vec<u8>,
    pub norms: Vec<f16>,
    pub group_size: usize,
    pub bits: u32,
}

impl CqMatrix {
    pub fn in_pad(&self) -> usize {
        self.inp.div_ceil(self.group_size) * self.group_size
    }
}

pub struct Archive {
    pub config: Config,
    /// cb2[4] | cb3[8] | cb4[16], each already divided by sqrt(group_size).
    pub codebook: Vec<f32>,
    pub records: Vec<Record>,
    pub tensors: Vec<Tensor>,
    pub raw: Vec<u8>,
}

fn le_u32(raw: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(raw[at..at + 4].try_into().unwrap())
}

fn le_u64(raw: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(raw[at..at + 8].try_into().unwrap())
}

pub fn read_archive(path: &std::path::Path) -> Result<Archive> {
    let raw = std::fs::read(path).map_err(Error::Io)?;
    read_archive_from(raw)
}

pub fn read_archive_from(raw: Vec<u8>) -> Result<Archive> {
    if raw.len() < HEADER_LEN {
        return Err(Error::Truncated("header"));
    }
    let tag = le_u32(&raw, 0);
    if tag == TAG_V2 {
        return Err(Error::UnsupportedVersion(tag));
    }
    if tag != TAG_V3 {
        return Err(Error::BadTag(tag));
    }
    let num_tensors = le_u32(&raw, 4) as usize;
    let cb_len = le_u32(&raw, 8) as usize;
    let kv_window = le_u32(&raw, 12) as usize;
    let kv_bits = le_u32(&raw, 16);
    let mut off = HEADER_LEN;
    if raw.len() < off + cb_len * 4 {
        return Err(Error::Truncated("codebook"));
    }
    let codebook: Vec<f32> = (0..cb_len)
        .map(|i| f32::from_le_bytes(raw[off + i * 4..off + i * 4 + 4].try_into().unwrap()))
        .collect();
    off += cb_len * 4;

    let num_layers = le_u32(&raw, 40) as usize;
    if num_layers > 64 {
        // The header's global_mask holds 64 layers; a larger count would both
        // wrap the `(gmask >> i)` shifts and spin for billions of iterations.
        return Err(Error::BadGeometry("global_mask holds at most 64 layers"));
    }
    let gmask = le_u32(&raw, 68) as u64 | ((le_u32(&raw, 72) as u64) << 32);
    let global_layers = (0..num_layers).filter(|&i| (gmask >> i) & 1 == 1).collect();
    // The header has fixed 4-order / 16-site slots; larger counts are capped
    // like the Python reader slices `orders4[:num_orders]`.
    let num_orders = le_u32(&raw, 104) as usize;
    let engram_orders: Vec<usize> = (0..num_orders.min(4)).map(|i| le_u32(&raw, 108 + i * 4) as usize).collect();
    let num_sites = le_u32(&raw, 124) as usize;
    let engram_layers: Vec<usize> = (0..num_sites.min(16)).map(|i| le_u32(&raw, 128 + i * 4) as usize).collect();

    let config = Config {
        vocab_size: le_u32(&raw, 20) as usize,
        out_vocab: le_u32(&raw, 24) as usize,
        d_model: le_u32(&raw, 28) as usize,
        num_heads: le_u32(&raw, 32) as usize,
        num_kv_heads: le_u32(&raw, 36) as usize,
        num_layers,
        qk_head_dim: le_u32(&raw, 44) as usize,
        v_head_dim: le_u32(&raw, 48) as usize,
        max_seq_len: le_u32(&raw, 52) as usize,
        hada_n: le_u32(&raw, 56) as usize,
        mhc_lanes: le_u32(&raw, 60) as usize,
        sliding_window: le_u32(&raw, 64) as usize,
        global_layers,
        qkv_conv_taps: le_u32(&raw, 76) as usize,
        engram_slots: le_u32(&raw, 80) as usize,
        engram_sub_dim: le_u32(&raw, 84) as usize,
        num_engram_tables: le_u32(&raw, 88) as usize,
        engram_conv_taps: le_u32(&raw, 92) as usize,
        engram_conv_dilation: le_u32(&raw, 96) as usize,
        engram_seed_heads: le_u32(&raw, 100) as usize,
        engram_orders,
        engram_layers,
        rope_theta: f32::from_le_bytes(raw[HEADER_LEN - 4..HEADER_LEN].try_into().unwrap()),
        kv_window,
        kv_bits,
    };

    // Cap the reservation: num_tensors is attacker-controlled up to 2^32 and
    // must not abort the process before the per-record bounds checks run.
    let mut records: Vec<Record> = Vec::with_capacity(num_tensors.min(4096));
    let mut tensors: Vec<Tensor> = Vec::with_capacity(num_tensors.min(4096));
    for _ in 0..num_tensors {
        if raw.len() < off + REC_LEN {
            return Err(Error::Truncated("directory"));
        }
        let r = &raw[off..off + REC_LEN];
        off += REC_LEN;
        let dtype = r[0];
        let ndim = r[1] as usize;
        if ndim > 4 {
            return Err(Error::BadShape);
        }
        let shape: Vec<usize> = (0..ndim).map(|i| le_u32(r, 4 + i * 4) as usize).collect();
        let offset = le_u64(r, 20);
        let nbytes = le_u64(r, 28);
        let group_size = le_u32(r, 36);
        let bits = le_u32(r, 40);
        let end = offset.checked_add(nbytes).ok_or(Error::Truncated("tensor range"))? as usize;
        if end > raw.len() {
            return Err(Error::Truncated("tensor blob"));
        }
        let blob = &raw[offset as usize..end];
        let tensor = match dtype {
            DTYPE_FP16 => {
                let data: Vec<f32> = blob
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| f16::from_le_bytes(*c).to_f32())
                    .collect();
                Tensor::Fp16 {
                    shape: shape.clone(),
                    data,
                }
            }
            DTYPE_FP32 => {
                let data: Vec<f32> = blob
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes(*c))
                    .collect();
                Tensor::Fp32 {
                    shape: shape.clone(),
                    data,
                }
            }
            DTYPE_CQ => {
                if shape.len() != 2 {
                    return Err(Error::BadShape);
                }
                // Valid widths are {1, 2, 3, 4, 5}; anything else would overflow
                // `in_pad * bits` / `1u64 << bits` downstream.
                if !matches!(bits, 1..=5) {
                    return Err(Error::BadBits(bits));
                }
                let out = shape[0];
                let inp = shape[1];
                let g = group_size as usize;
                // group_size must divide evenly for the norms check and be a
                // power of two for the Hadamard transform in `dequant_cq`.
                if g == 0 {
                    return Err(Error::BadGeometry("CQ group_size must be >= 1"));
                }
                if !g.is_power_of_two() {
                    return Err(Error::BadGeometry("CQ group_size must be a power of two"));
                }
                let in_pad = inp.div_ceil(g) * g;
                let n_packed = out
                    .checked_mul(packed_row_bytes(in_pad, bits))
                    .ok_or(Error::Truncated("cq packed"))?;
                if blob.len() < n_packed {
                    return Err(Error::Truncated("cq packed"));
                }
                let norms: Vec<f16> = blob[n_packed..]
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| f16::from_le_bytes(*c))
                    .collect();
                if norms.len() != out.checked_mul(in_pad / g).ok_or(Error::Truncated("cq norms"))? {
                    return Err(Error::Truncated("cq norms"));
                }
                Tensor::Cq(CqMatrix {
                    out,
                    inp,
                    packed: blob[..n_packed].to_vec(),
                    norms,
                    group_size: g,
                    bits,
                })
            }
            DTYPE_RAW => Tensor::Raw(blob.to_vec()),
            other => return Err(Error::BadDtype(other)),
        };
        records.push(Record {
            dtype,
            shape,
            offset,
            nbytes,
            group_size,
            bits,
        });
        tensors.push(tensor);
    }

    Ok(Archive {
        config,
        codebook,
        records,
        tensors,
        raw,
    })
}

pub fn packed_row_bytes(in_pad: usize, bits: u32) -> usize {
    if bits == TERNARY_RECORD_BITS {
        in_pad * 2 / 8
    } else {
        in_pad * bits as usize / 8
    }
}

/// Unpack the LSB-first bitstream of code indices (one continuous stream per row:
/// index `k` occupies bits `[k*bits, (k+1)*bits)`).
pub fn unpack_lsb(packed: &[u8], bits: u32, out: usize, in_pad: usize) -> Vec<u8> {
    let bits = bits as usize;
    let row_bytes = in_pad * bits / 8;
    let mut idx = vec![0u8; out * in_pad];
    let mask = (1u64 << bits) - 1;
    for row in 0..out {
        let base = row * row_bytes;
        for k in 0..in_pad {
            let bitpos = k * bits;
            let byte = base + bitpos / 8;
            let shift = (bitpos % 8) as u32;
            let mut word = 0u64;
            for j in 0..8usize {
                if byte + j < packed.len() {
                    word |= (packed[byte + j] as u64) << (8 * j);
                }
            }
            idx[row * in_pad + k] = ((word >> shift) & mask) as u8;
        }
    }
    idx
}

/// The analytic ternary codebook `{-c, 0, +c} / sqrt(group)`, `c = 1.2240064`.
pub fn ternary_codebook(group: usize) -> Vec<f32> {
    let c = 1.2240064f32;
    let inv = 1.0 / (group as f32).sqrt();
    vec![-c * inv, 0.0, c * inv]
}

/// The analytic binary codebook `{-s, +s} / sqrt(group)`, `s = sqrt(2/pi)`.
pub fn binary_codebook(group: usize) -> Vec<f32> {
    let s = (2.0f32 / std::f32::consts::PI).sqrt();
    let inv = 1.0 / (group as f32).sqrt();
    vec![-s * inv, s * inv]
}

impl Archive {
    /// Codebook for a logical width: header cb2/cb3/cb4 for 2/3/4, analytic for
    /// 1 (binary) and ternary crumbs (record bits == 5). Errs on short books
    /// instead of silently truncating.
    pub fn codebook_for(&self, bits: u32, group: usize) -> Result<Vec<f32>> {
        let header_book = |start: usize, len: usize| -> Result<Vec<f32>> {
            if self.codebook.len() < start + len {
                return Err(Error::BadGeometry("codebook too short for this CQ width"));
            }
            Ok(self.codebook[start..start + len].to_vec())
        };
        match bits {
            2 => header_book(0, CB2),
            3 => header_book(CB2, CB3),
            4 => header_book(CB2 + CB3, CB4),
            1 => Ok(binary_codebook(group)),
            TERNARY_RECORD_BITS => Ok(ternary_codebook(group)),
            other => Err(Error::BadBits(other)),
        }
    }

    /// The embedded tokenizer attachment, if present.
    pub fn tokenizer_blob(&self) -> Option<&[u8]> {
        self.tensors.iter().find_map(|t| match t {
            Tensor::Raw(b) => Some(b.as_slice()),
            _ => None,
        })
    }
}

/// Normalized Walsh–Hadamard matrix of size `n` (n a power of two), `H / sqrt(n)`.
pub fn walsh_hadamard(n: usize) -> Vec<f32> {
    let mut h = vec![1.0f32];
    let mut m = 1usize;
    while m < n {
        let mut next = vec![0.0f32; 4 * m * m];
        for i in 0..m {
            for j in 0..m {
                let v = h[i * m + j];
                next[i * 2 * m + j] = v;
                next[i * 2 * m + m + j] = v;
                next[(m + i) * 2 * m + j] = v;
                next[(m + i) * 2 * m + m + j] = -v;
            }
        }
        h = next;
        m *= 2;
    }
    let inv = 1.0 / (n as f32).sqrt();
    for v in h.iter_mut() {
        *v *= inv;
    }
    h
}

/// In-place fast Walsh–Hadamard transform with normalization `1/sqrt(n)`.
pub fn fwht(x: &mut [f32]) {
    let n = x.len();
    let mut h = 1usize;
    while h < n {
        let mut i = 0usize;
        while i < n {
            for j in i..i + h {
                let a = x[j];
                let b = x[j + h];
                x[j] = a + b;
                x[j + h] = a - b;
            }
            i += 2 * h;
        }
        h *= 2;
    }
    let inv = 1.0 / (n as f32).sqrt();
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// Dequantize a CQ matrix to row-major f32 `[out, inp]`:
/// `w_group = (codebook[idx] * norm) @ H` per group.
pub fn dequant_cq(mat: &CqMatrix, codebook: &[f32]) -> Result<Vec<f32>> {
    // Re-validate: `mat` may come from anywhere, not just `read_archive_from`.
    if !matches!(mat.bits, 1..=5) {
        return Err(Error::BadBits(mat.bits));
    }
    let g = mat.group_size;
    if g == 0 || !g.is_power_of_two() {
        return Err(Error::BadGeometry("CQ group_size must be a power of two >= 1"));
    }
    let in_pad = mat.in_pad();
    let idx = if mat.bits == TERNARY_RECORD_BITS {
        let crumbs = unpack_lsb(&mat.packed, 2, mat.out, in_pad);
        crumbs
            .iter()
            .map(|&c| if c == 3 { 0 } else { c + 1 })
            .collect::<Vec<u8>>()
    } else {
        unpack_lsb(&mat.packed, mat.bits, mat.out, in_pad)
    };
    let groups_per_row = in_pad / g;
    let mut w = vec![0.0f32; mat.out * in_pad];
    let mut tmp = vec![0.0f32; g];
    for row in 0..mat.out {
        for gr in 0..groups_per_row {
            let norm = mat.norms[row * groups_per_row + gr].to_f32();
            let base = row * in_pad + gr * g;
            for k in 0..g {
                let ci = idx[row * in_pad + gr * g + k] as usize;
                let c = codebook
                    .get(ci)
                    .ok_or(Error::BadGeometry("codebook index out of range"))?;
                tmp[k] = c * norm;
            }
            fwht(&mut tmp);
            w[base..base + g].copy_from_slice(&tmp);
        }
    }
    if in_pad == mat.inp {
        Ok(w)
    } else {
        let mut out = vec![0.0f32; mat.out * mat.inp];
        for row in 0..mat.out {
            out[row * mat.inp..(row + 1) * mat.inp]
                .copy_from_slice(&w[row * in_pad..row * in_pad + mat.inp]);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn models_dir() -> std::path::PathBuf {
        std::env::var("NEEDLE_MODELS_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| {
                std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models")
            })
    }

    #[test]
    fn parses_published_needle3() {
        let path = models_dir().join("needle3.cact");
        if !path.exists() {
            eprintln!("skip: {} not downloaded", path.display());
            return;
        }
        let ar = read_archive(&path).unwrap();
        let c = &ar.config;
        assert_eq!(c.vocab_size, 8192);
        assert_eq!(c.d_model, 768);
        assert_eq!(c.num_heads, 12);
        assert_eq!(c.num_kv_heads, 2);
        assert_eq!(c.num_layers, 20);
        assert_eq!(c.qk_head_dim, 48);
        assert_eq!(c.v_head_dim, 64);
        assert_eq!(c.hada_n, 1024);
        assert_eq!(c.mhc_lanes, 4);
        assert_eq!(c.sliding_window, 1024);
        assert_eq!(c.global_layers, vec![4, 9, 14, 19]);
        assert_eq!(c.qkv_conv_taps, 3);
        assert_eq!(c.engram_slots, 18432);
        assert_eq!(c.engram_sub_dim, 128);
        assert_eq!(c.num_engram_tables, 6);
        assert_eq!(c.engram_conv_dilation, 3);
        assert_eq!(c.engram_orders, vec![2, 3]);
        assert_eq!(c.engram_layers, vec![3, 7, 11, 15, 19]);
        assert!((c.rope_theta - 100000.0).abs() < 1.0);
        assert_eq!(c.kv_bits, 8);
        assert_eq!(ar.codebook.len(), 28);
        assert!(ar.tokenizer_blob().is_some());
        println!("num_tensors = {}", ar.tensors.len());
        println!("kv_window = {}", c.kv_window);
        println!("max_seq_len = {}", c.max_seq_len);
        println!("out_vocab = {}", c.out_vocab);
    }

    #[test]
    fn lsb_roundtrip() {
        let out = 3usize;
        let in_pad = 32usize;
        let bits = 4u32;
        let idx: Vec<u8> = (0..(out * in_pad) as u8).map(|i| i % 16).collect();
        let mut packed = vec![0u8; out * in_pad * bits as usize / 8];
        for (i, &iv) in idx.iter().enumerate() {
            let v = iv as u64;
            let bitpos = i * bits as usize;
            for j in 0..bits as usize {
                if (v >> j) & 1 == 1 {
                    packed[(bitpos + j) / 8] |= 1 << ((bitpos + j) % 8);
                }
            }
        }
        let back = unpack_lsb(&packed, bits, out, in_pad);
        assert_eq!(back, idx);
    }

    #[test]
    fn fwht_matches_matrix() {
        for n in [4usize, 32, 128] {
            let h = walsh_hadamard(n);
            let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin()).collect();
            let mut got = x.clone();
            fwht(&mut got);
            for i in 0..n {
                let expect: f32 = (0..n).map(|j| h[i * n + j] * x[j]).sum();
                assert!((got[i] - expect).abs() < 1e-4, "n={n} i={i}");
            }
        }
    }

    #[test]
    fn cq_roundtrip_small() {
        let g = 8usize;
        let cb = ternary_codebook(g);
        let out = 2usize;
        let inp = 16usize;
        let w: Vec<f32> = (0..out * inp)
            .map(|i| ((i as f32) * 0.311).sin() * 0.8)
            .collect();
        let in_pad = inp;
        let mut norms = vec![f16::ZERO; out * (in_pad / g)];
        let mut packed = vec![0u8; out * in_pad * 2 / 8];
        for row in 0..out {
            for gr in 0..in_pad / g {
                let mut rot = vec![0f32; g];
                rot.copy_from_slice(&w[row * inp + gr * g..row * inp + gr * g + g]);
                fwht(&mut rot);
                let norm: f32 = rot.iter().map(|v| v * v).sum::<f32>().sqrt();
                norms[row * (in_pad / g) + gr] = f16::from_f32(norm);
                for (k, &rv) in rot.iter().enumerate() {
                    let unit = rv / norm.max(1e-12);
                    let mut best = 0usize;
                    let mut bd = f32::MAX;
                    for (ci, &c) in cb.iter().enumerate() {
                        let d = (unit - c).abs();
                        if d < bd {
                            bd = d;
                            best = ci;
                        }
                    }
                    let crumb = if best == 0 { 3u8 } else { (best - 1) as u8 };
                    let bitpos = (row * in_pad + gr * g + k) * 2;
                    for j in 0..2u32 {
                        if (crumb >> j) & 1 == 1 {
                            packed[(bitpos + j as usize) / 8] |=
                                1 << ((bitpos + j as usize) % 8);
                        }
                    }
                }
            }
        }
        let mat = CqMatrix {
            out,
            inp,
            packed,
            norms,
            group_size: g,
            bits: TERNARY_RECORD_BITS,
        };
        let back = dequant_cq(&mat, &cb).unwrap();
        for row in 0..out {
            for gr in 0..in_pad / g {
                let a = &w[row * inp + gr * g..row * inp + gr * g + g];
                let b = &back[row * inp + gr * g..row * inp + gr * g + g];
                let num: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt();
                let den: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
                assert!(num / den < 0.6, "row={row} gr={gr} rel={}", num / den);
            }
        }
    }
}
