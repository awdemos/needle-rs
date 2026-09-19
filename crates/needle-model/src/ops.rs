//! Small f32 compute kernels used by the SAN forward pass. Row-major everywhere;
//! matrices are `[out, in]` so `y = W x` is a row-parallel GEMV.
#![allow(dead_code)]

use rayon::prelude::*;

/// `(1 + scale) * x / rms(x)` — the ZCRMSNorm.
pub fn zc_rms_norm(x: &[f32], scale: &[f32], eps: f32, out: &mut [f32]) {
    let n = x.len();
    let mut ss = 0.0f32;
    for v in x {
        ss += v * v;
    }
    let rms = (ss / n as f32 + eps).sqrt();
    for i in 0..n {
        out[i] = (1.0 + scale[i]) * x[i] / rms;
    }
}

/// Unit RMS normalization in f32 (no learnable scale).
pub fn rms_unit(x: &[f32], eps: f32, out: &mut [f32]) {
    let n = x.len();
    let mut ss = 0.0f32;
    for v in x {
        ss += v * v;
    }
    let inv = 1.0 / (ss / n as f32 + eps).sqrt();
    for i in 0..n {
        out[i] = x[i] * inv;
    }
}

/// `y = W x` with W row-major `[out, in]`.
pub fn matvec(w: &[f32], x: &[f32], out: &mut [f32]) {
    let n_in = x.len();
    out.par_iter_mut().enumerate().for_each(|(i, y)| {
        let row = &w[i * n_in..(i + 1) * n_in];
        let mut acc = 0.0f32;
        for (a, b) in row.iter().zip(x) {
            acc += a * b;
        }
        *y = acc;
    });
}

/// `Y = X W^T` with X `[t, in]` row-major and W `[out, in]`; writes Y `[t, out]`.
pub fn matmul_rows(x: &[f32], w: &[f32], t: usize, n_in: usize, n_out: usize, y: &mut [f32]) {
    debug_assert_eq!(x.len(), t * n_in);
    debug_assert_eq!(w.len(), n_out * n_in);
    debug_assert_eq!(y.len(), t * n_out);
    for ti in 0..t {
        matvec(w, &x[ti * n_in..(ti + 1) * n_in], &mut y[ti * n_out..(ti + 1) * n_out]);
    }
}

pub fn softmax(x: &[f32]) -> Vec<f32> {
    let m = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut out = Vec::with_capacity(x.len());
    let mut sum = 0.0f32;
    for &v in x {
        let e = (v - m).exp();
        out.push(e);
        sum += e;
    }
    let inv = 1.0 / sum.max(1e-30);
    for v in out.iter_mut() {
        *v *= inv;
    }
    out
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// 2D softmax along the last axis for `[rows, cols]` logits.
pub fn softmax_rows(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0; x.len()];
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for (o, &v) in out[r * cols..(r + 1) * cols].iter_mut().zip(row) {
            let e = (v - m).exp();
            *o = e;
            sum += e;
        }
        let inv = 1.0 / sum.max(1e-30);
        for o in &mut out[r * cols..(r + 1) * cols] {
            *o *= inv;
        }
    }
    out
}

/// Sinkhorn: 20 rounds of row/col logsumexp centering, then exp.
pub fn sinkhorn(logits: &[f32], n: usize) -> Vec<f32> {
    let mut k = logits.to_vec();
    for _ in 0..20 {
        // rows: k -= logsumexp(k, axis=-1)
        for i in 0..n {
            let row = &k[i * n..(i + 1) * n];
            let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = row.iter().map(|&v| (v - m).exp()).sum();
            let logsum = m + sum.ln();
            for v in k[i * n..(i + 1) * n].iter_mut() {
                *v -= logsum;
            }
        }
        // cols: k -= logsumexp(k, axis=-2)
        for j in 0..n {
            let mut m = f32::NEG_INFINITY;
            for i in 0..n {
                m = m.max(k[i * n + j]);
            }
            let mut sum = 0.0f32;
            for i in 0..n {
                sum += (k[i * n + j] - m).exp();
            }
            let logsum = m + sum.ln();
            for i in 0..n {
                k[i * n + j] -= logsum;
            }
        }
    }
    for v in k.iter_mut() {
        *v = v.exp();
    }
    k
}

/// Per-head A8 (symmetric int8) quantization matching `fake_quant(x, head_dim, 8)`:
/// scale = absmax/127 over each head vector; returns (values i8, scales f32).
pub fn quantize_a8_per_group(x: &[f32], groups: usize, group_len: usize) -> (Vec<i8>, Vec<f32>) {
    let mut q = vec![0i8; x.len()];
    let mut scales = vec![0.0f32; groups];
    for g in 0..groups {
        let seg = &x[g * group_len..(g + 1) * group_len];
        let absmax = seg.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = if absmax > 0.0 { absmax / 127.0 } else { 1.0 };
        scales[g] = scale;
        for (i, &v) in seg.iter().enumerate() {
            q[g * group_len + i] = (v / scale).round().clamp(-127.0, 127.0) as i8;
        }
    }
    (q, scales)
}

pub fn dequant_a8(q: &[i8], scales: &[f32], group_len: usize, out: &mut [f32]) {
    for (i, &qv) in q.iter().enumerate() {
        out[i] = qv as f32 * scales[i / group_len];
    }
}

/// RoPE tables: `cos[t*half + i]`, `sin[t*half + i]` for head_dim `2*half`.
pub struct Rope {
    pub half: usize,
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
}

impl Rope {
    pub fn new(head_dim: usize, seq_len: usize, theta: f32) -> Rope {
        let half = head_dim / 2;
        let mut cos = vec![0.0; seq_len * half];
        let mut sin = vec![0.0; seq_len * half];
        for t in 0..seq_len {
            for i in 0..half {
                let freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32);
                let a = t as f32 * freq;
                cos[t * half + i] = a.cos();
                sin[t * half + i] = a.sin();
            }
        }
        Rope { half, cos, sin }
    }

    /// Rotate-half: out = [x1*cos - x2*sin, x2*cos + x1*sin].
    pub fn apply(&self, x: &[f32], t: usize, out: &mut [f32]) {
        let h = self.half;
        for i in 0..h {
            let c = self.cos[t * h + i];
            let s = self.sin[t * h + i];
            let x1 = x[i];
            let x2 = x[h + i];
            out[i] = x1 * c - x2 * s;
            out[h + i] = x2 * c + x1 * s;
        }
    }
}
