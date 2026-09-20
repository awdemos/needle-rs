//! Incremental SAN forward pass: one unified single-position `step` drives both
//! prefill and decode. Cross-time state lives in `Cache` (attention KV, causal
//! conv tap rings, engram raw-value rings, and the short token window the
//! engram hash reads).
//!
//! Attention KV is cached in f32 with unit scales, matching the JAX reference
//! and the shipped checkpoint's training path. The archive header's `kv_bits`
//! and `kv_window` fields are parsed into `Config` but currently unused here —
//! int8 KV caching with windowed eviction is a deployed-engine memory
//! optimization layered on top, not part of this port's numerics.

use crate::ops::*;
use crate::{engram_indices, Config, Error, Model, Result};
use rayon::prelude::*;
use std::cell::RefCell;

const EPS: f32 = 1e-6;

/// Per-position hidden cells `[t, rows, d]` (rows = layers+1) for the probe heads.
pub struct Cells {
    pub data: Vec<f32>,
    pub t: usize,
    pub rows: usize,
    pub d: usize,
}

impl Cells {
    pub fn new(t: usize, rows: usize, d: usize) -> Cells {
        Cells {
            data: vec![0.0; t * rows * d],
            t,
            rows,
            d,
        }
    }

    fn set_row(&mut self, t: usize, row: usize, x: &[f32]) {
        debug_assert_eq!(x.len(), self.d);
        self.data[(t * self.rows + row) * self.d..(t * self.rows + row + 1) * self.d]
            .copy_from_slice(x);
    }
}

/// Per-layer, per-site cross-time state.
pub struct Cache {
    pub len: usize,
    k_q: Vec<Vec<f32>>, // per layer: [pos][K*qh] flat (scales kept for the int8 option)
    k_s: Vec<Vec<f32>>, // per layer: [pos][K]
    v_q: Vec<Vec<f32>>, // per layer: [pos][K*vh]
    v_s: Vec<Vec<f32>>,
    rq: Vec<Vec<f32>>, // ring (taps-1) * H*qh — raw q for the causal conv
    rk: Vec<Vec<f32>>, // ring (taps-1) * K*qh
    rv: Vec<Vec<f32>>, // ring (taps-1) * K*vh
    ev_raw: Vec<Vec<f32>>, // per site: ring ((etaps-1)*dil) * d — raw engram v
    tok_ring: Vec<u32>,    // last max_order-1 tokens, for the engram hash window
}

const fn max(a: usize, b: usize) -> usize {
    if a > b {
        a
    } else {
        b
    }
}

struct Buffers {
    x: Vec<f32>,
    nx: Vec<f32>,
    hpre: Vec<f32>,
    hpost: Vec<f32>,
    hres: Vec<f32>,
    u: Vec<f32>,
    y: Vec<f32>,
    stream2: Vec<f32>,
    q: Vec<f32>,     // H*qh
    k: Vec<f32>,     // K*qh
    v: Vec<f32>,     // K*vh
    qc: Vec<f32>,    // conv-tap accumulators
    kc: Vec<f32>,
    vc: Vec<f32>,
    qn: Vec<f32>,    // normed+roped q (H*qh)
    kn: Vec<f32>,    // normed+roped k (K*qh)
    attn_out: Vec<f32>, // H*vh
    d_k: Vec<f32>,   // dequant k for one position (K*qh)
    d_v: Vec<f32>,
    e: Vec<f32>,     // engram embedding (num_tables*sub)
    k_tmp: Vec<f32>, // engram k (d) — current site
    v_tmp: Vec<f32>, // engram raw v (d) — current site
    ek: Vec<f32>,    // per-site engram k (sites*d)
    ev: Vec<f32>,    // per-site engram conv v (sites*d)
    cond: Vec<f32>,  // hadamard conditioner (hada_n)
    gate: Vec<f32>,  // attention output gate (H*vh)
    tmp: Vec<f32>,
    hada: Vec<f32>,
    logits: Vec<f32>,
}

impl Buffers {
    fn new(cap: usize) -> Buffers {
        let z = vec![0.0; cap];
        Buffers {
            x: z.clone(),
            nx: z.clone(),
            hpre: z.clone(),
            hpost: z.clone(),
            hres: z.clone(),
            u: z.clone(),
            y: z.clone(),
            stream2: z.clone(),
            q: z.clone(),
            k: z.clone(),
            v: z.clone(),
            qc: z.clone(),
            kc: z.clone(),
            vc: z.clone(),
            qn: z.clone(),
            kn: z.clone(),
            attn_out: z.clone(),
            d_k: z.clone(),
            d_v: z.clone(),
            e: z.clone(),
            k_tmp: z.clone(),
            v_tmp: z.clone(),
            ek: z.clone(),
            ev: z.clone(),
            cond: z.clone(),
            gate: z.clone(),
            tmp: z.clone(),
            hada: z.clone(),
            logits: z.clone(),
        }
    }
}

thread_local! {
    static BUF: RefCell<Buffers> = RefCell::new(Buffers::new(0));
}

fn with_buffers<R>(cap: usize, f: impl FnOnce(&mut Buffers) -> R) -> R {
    BUF.with(|b| {
        let mut slot = b.borrow_mut();
        if slot.x.len() < cap {
            *slot = Buffers::new(cap);
        }
        f(&mut slot)
    })
}

impl Model {
    pub fn new_cache(&self) -> Cache {
        let cfg = &self.config;
        let kdim = cfg.num_kv_heads * cfg.qk_head_dim;
        let vdim = cfg.num_kv_heads * cfg.v_head_dim;
        let qdim = cfg.num_heads * cfg.qk_head_dim;
        let t = cfg.qkv_conv_taps.saturating_sub(1);
        let layers: Vec<_> = (0..cfg.num_layers)
            .map(|_| {
                (
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    vec![0.0; t * qdim],
                    vec![0.0; t * kdim],
                    vec![0.0; t * vdim],
                )
            })
            .collect();
        let (k_q, k_s, v_q, v_s, rq, rk, rv) = layers.into_iter().fold(
            (vec![], vec![], vec![], vec![], vec![], vec![], vec![]),
            |(mut a, mut b, mut c, mut d, mut e, mut f, mut g),
             (x1, x2, x3, x4, x5, x6, x7)| {
                a.push(x1);
                b.push(x2);
                c.push(x3);
                d.push(x4);
                e.push(x5);
                f.push(x6);
                g.push(x7);
                (a, b, c, d, e, f, g)
            },
        );
        let ring = (cfg.engram_conv_taps.saturating_sub(1)) * max(cfg.engram_conv_dilation, 1);
        let ev_raw = vec![vec![0.0; ring * cfg.d_model]; cfg.engram_layers.len()];
        let max_order = cfg.engram_orders.iter().copied().max().unwrap_or(1);
        Cache {
            len: 0,
            k_q,
            k_s,
            v_q,
            v_s,
            rq,
            rk,
            rv,
            ev_raw,
            tok_ring: vec![0; max_order.saturating_sub(1)],
        }
    }

    /// Run one position through the network; returns logits `[out_vocab]`.
    pub fn step(
        &self,
        token: u32,
        pos: usize,
        cache: &mut Cache,
        mut cells: Option<&mut Cells>,
    ) -> Result<Vec<f32>> {
        let cfg = &self.config;
        // `step` appends exactly one position to the cache; anything other than
        // the next contiguous position would silently desync the position→slot
        // mapping (or index past the KV rows), so refuse it up front.
        if pos != cache.len {
            return Err(Error::Shape("non-contiguous position: expected cache.len"));
        }
        if pos >= cfg.max_seq_len {
            return Err(Error::Shape("sequence position exceeds max_seq_len"));
        }
        if token as usize >= cfg.vocab_size {
            return Err(Error::Shape("token id out of range"));
        }
        let d = cfg.d_model;
        let n = cfg.mhc_lanes;
        let l_count = cfg.num_layers;
        let qh = cfg.qk_head_dim;
        let _vh = cfg.v_head_dim;
        let h = cfg.num_heads;
        let _kvh = cfg.num_kv_heads;
        let qdim = h * qh;
        let n_c = n * d;
        let etaps = cfg.engram_conv_taps;
        let dil = max(cfg.engram_conv_dilation, 1);
        let orders_n = cfg.engram_orders.len();
        // from_archive rejects num_engram_tables not divisible by orders_n.
        let heads = cfg.num_engram_tables.checked_div(orders_n).unwrap_or(0);
        let num_tables = cfg.num_engram_tables;
        let sub = cfg.engram_sub_dim;
        let table_dim = num_tables * sub;
        let max_order = cfg.engram_orders.iter().copied().max().unwrap_or(1);
        let sites = cfg.engram_layers.len();
        let cap = max(max(max(n_c, sites * d), table_dim), qdim + d);

        let logits = with_buffers(cap, |b| -> Result<Vec<f32>> {
            // ---- input embedding ----
            b.x[..d].copy_from_slice(&self.embedding[token as usize * d..(token as usize + 1) * d]);
            for v in b.x[..d].iter_mut() {
                *v *= self.embed_scale;
            }
            if let Some(c) = cells.as_deref_mut() {
                c.set_row(pos, 0, &b.x[..d]);
            }
            let mut stream: Vec<f32> = vec![0.0; n * d];
            for i in 0..n {
                stream[i * d..(i + 1) * d].copy_from_slice(&b.x[..d]);
            }

            // ---- engram ek/ev for this position ----
            // hash window: last max_order-1 tokens + current
            let win = max_order; // window length incl. current
            let mut idx_window = vec![0u32; win];
            let hist = cache.tok_ring.len();
            idx_window[..hist].copy_from_slice(&cache.tok_ring[..hist]);
            idx_window[hist] = token;
            let idx = engram_indices(
                &idx_window,
                &cfg.engram_orders,
                heads,
                cfg.engram_slots,
                cfg.engram_seed_heads,
            );
            let cur_row = (win - 1) * num_tables;
            for s in 0..sites {
                let eg = &self.engrams[s];
                for t_i in 0..num_tables {
                    let o = cfg.engram_orders[t_i / heads];
                    let ok = if pos + 1 >= o { 1.0 } else { 0.0 };
                    let slot = idx[cur_row + t_i] as usize;
                    let base = (t_i * cfg.engram_slots + slot) * sub;
                    for j in 0..sub {
                        b.e[t_i * sub + j] = eg.tables[base + j] * ok;
                    }
                }
                matvec(&eg.key_proj, &b.e[..table_dim], &mut b.k_tmp[..d]);
                matvec(&eg.value_proj, &b.e[..table_dim], &mut b.v_tmp[..d]);
                // conv v over the raw ring: ev = sum_j taps[j] * raw[pos - j*dil]
                for i in 0..d {
                    b.ev[s * d + i] = eg.taps[i] * b.v_tmp[i];
                }
                for j in 1..etaps {
                    let back = j * dil;
                    if pos >= back {
                        let ring_len = cache.ev_raw[s].len() / d;
                        let slot = (pos + ring_len - back) % ring_len;
                        for i in 0..d {
                            b.ev[s * d + i] += eg.taps[j * d + i] * cache.ev_raw[s][slot * d + i];
                        }
                    }
                }
                // with engram_conv_taps <= 1 there is no history ring, exactly
                // like the attention conv rings below
                let ring_len = cache.ev_raw[s].len() / d;
                if ring_len > 0 {
                    let slot = pos % ring_len;
                    cache.ev_raw[s][slot * d..(slot + 1) * d].copy_from_slice(&b.v_tmp[..d]);
                }
                b.ek[s * d..(s + 1) * d].copy_from_slice(&b.k_tmp[..d]);
            }
            // shift the token ring left, append current
            if hist > 0 {
                for j in 0..hist - 1 {
                    cache.tok_ring[j] = cache.tok_ring[j + 1];
                }
                cache.tok_ring[hist - 1] = token;
            }

            // ---- layers ----
            for li in 0..l_count {
                // nx = rms_unit(stream flattened over lanes)
                {
                    let mut ss = 0.0f32;
                    for v in &stream {
                        ss += v * v;
                    }
                    let inv = 1.0 / (ss / n_c as f32 + EPS).sqrt();
                    for (i, &v) in stream.iter().enumerate() {
                        b.nx[i] = v * inv;
                    }
                }
                {
                    let phi_pre = &self.mhc.phi_pre[li * n_c * n..(li + 1) * n_c * n];
                    let phi_post = &self.mhc.phi_post[li * n_c * n..(li + 1) * n_c * n];
                    let lane = li % n;
                    for i in 0..n {
                        let mut sp = 0.0f32;
                        let mut sq = 0.0f32;
                        for c in 0..n_c {
                            sp += b.nx[c] * phi_pre[c * n + i];
                            sq += b.nx[c] * phi_post[c * n + i];
                        }
                        let pre_off = if i == lane { 4.0 } else { -4.0 };
                        let post_off = if i == lane { 0.0 } else { -4.0 };
                        b.hpre[i] = sigmoid(
                            self.mhc.a_pre[li] * sp + self.mhc.b_pre[li * n + i] + pre_off,
                        );
                        b.hpost[i] = 2.0
                            * sigmoid(
                                self.mhc.a_post[li] * sq + self.mhc.b_post[li * n + i] + post_off,
                            );
                    }
                }
                for i in 0..d {
                    b.u[i] = 0.0;
                }
                for lane_i in 0..n {
                    let w = b.hpre[lane_i];
                    let seg = &stream[lane_i * d..(lane_i + 1) * d];
                    for (i, &sv) in seg.iter().enumerate() {
                        b.u[i] += w * sv;
                    }
                }

                // y = block(u) - u, written into b.y
                let u_copy: Vec<f32> = b.u[..d].to_vec();
                self.block(li, pos, cache, &u_copy, b)?;

                // hres = sinkhorn(a_res * (nx @ phi_res) + b_res)
                {
                    let phi_res =
                        &self.mhc.phi_res[li * n_c * n * n..(li + 1) * n_c * n * n];
                    let mut logits = vec![0.0f32; n * n];
                    for i in 0..n {
                        for j in 0..n {
                            let mut acc = 0.0f32;
                            for c in 0..n_c {
                                acc += b.nx[c] * phi_res[c * n * n + i * n + j];
                            }
                            logits[i * n + j] =
                                self.mhc.a_res[li] * acc + self.mhc.b_res[li * n * n + i * n + j];
                        }
                    }
                    let sh = sinkhorn(&logits, n);
                    b.hres[..n * n].copy_from_slice(&sh);
                }
                for lane_i in 0..n {
                    for i in 0..d {
                        let mut acc = 0.0f32;
                        for j in 0..n {
                            acc += b.hres[lane_i * n + j] * stream[j * d + i];
                        }
                        b.stream2[lane_i * d + i] = acc + b.hpost[lane_i] * b.y[i];
                    }
                }
                std::mem::swap(&mut stream, &mut b.stream2);

                if let Some(c) = cells.as_deref_mut() {
                    for i in 0..d {
                        let mut m = 0.0f32;
                        for lane_i in 0..n {
                            m += stream[lane_i * d + i];
                        }
                        b.tmp[i] = m / n as f32;
                    }
                    c.set_row(pos, li + 1, &b.tmp[..d]);
                }
            }

            // ---- final norm + tied head on the lane mean ----
            for i in 0..d {
                let mut m = 0.0f32;
                for lane_i in 0..n {
                    m += stream[lane_i * d + i];
                }
                b.x[i] = m / n as f32;
            }
            zc_rms_norm(&b.x[..d], &self.final_norm, EPS, &mut b.tmp[..d]);
            let vocab = self.out_vocab;
            b.logits.clear();
            b.logits.resize(vocab, 0.0);
            b.logits
                .par_iter_mut()
                .enumerate()
                .for_each(|(vi, y)| {
                    let row = &self.embedding[vi * d..(vi + 1) * d];
                    let mut acc = 0.0f32;
                    for (i, &tv) in b.tmp[..d].iter().enumerate() {
                        acc += row[i] * tv;
                    }
                    *y = acc;
                });
            Ok(b.logits[..vocab].to_vec())
        })?;
        cache.len = cache.len.max(pos + 1);
        Ok(logits)
    }
    /// `y = block(u) - u` for one position; u is the MHC-merged lane input.
    fn block(&self, li: usize, pos: usize, cache: &mut Cache, u: &[f32], b: &mut Buffers) -> Result<()> {
        let cfg = &self.config;
        let d = cfg.d_model;
        let qh = cfg.qk_head_dim;
        let _vh = cfg.v_head_dim;
        let h = cfg.num_heads;
        let _kvh = cfg.num_kv_heads;
        let _qdim = h * qh;
        let sites = cfg.engram_layers.len();
        let lw = &self.layers[li];

        // ---- engram add: x = u + sum_s flag[l][s] * alpha_s * ev_s ----
        b.x[..d].copy_from_slice(u);
        if sites > 0 {
            rms_unit(&b.x[..d], EPS, &mut b.tmp[..d]);
            let xu = b.tmp[..d].to_vec();
            for s in 0..sites {
                if cfg.engram_layers[s] != li {
                    continue;
                }
                let ek = &b.ek[s * d..(s + 1) * d];
                // rms_unit(ek) into k_tmp
                rms_unit(ek, EPS, &mut b.k_tmp[..d]);
                let mut dot = 0.0f32;
                for (i, &xv) in xu.iter().enumerate() {
                    dot += xv * b.k_tmp[i];
                }
                let alpha = sigmoid(dot / (d as f32).sqrt());
                let ev = &b.ev[s * d..(s + 1) * d];
                for (i, &evv) in ev.iter().enumerate() {
                    b.x[i] += alpha * evv;
                }
            }
        }

        // ---- attention sub-block ----
        // h = attn(zc_rms_norm(x, norm_in)); then skip + attn_gate * zc_rms_norm(h, post_norm)
        zc_rms_norm(&b.x[..d], &lw.norm_in, EPS, &mut b.tmp[..d]);
        let attn_in = b.tmp[..d].to_vec();
        self.attention(li, pos, cache, &attn_in, b)?;

        zc_rms_norm(&b.attn_out[..d], &lw.post_norm, EPS, &mut b.tmp[..d]);
        let gate = sigmoid(lw.attn_gate);
        for i in 0..d {
            b.x[i] += gate * b.tmp[i];
        }

        // ---- hadamard sub-block: skip + hadamard(zc_rms_norm(x, pre_hada)) ----
        zc_rms_norm(&b.x[..d], &lw.pre_hada, EPS, &mut b.tmp[..d]);
        let xin: Vec<f32> = b.tmp[..d].to_vec();
        self.hadamard(lw, &xin, b)?;
        for (i, &uv) in u.iter().enumerate() {
            b.y[i] = (b.x[i] + b.hada[i]) - uv;
        }
        Ok(())
    }

    /// GQA attention for one query position; writes H*vh into `b.attn_out`.
    /// The attention input is `attn_in` (normed x); the gate uses the same input.
    fn attention(
        &self,
        li: usize,
        pos: usize,
        cache: &mut Cache,
        attn_in: &[f32],
        b: &mut Buffers,
    ) -> Result<()> {
        let cfg = &self.config;
        let d = cfg.d_model;
        let qh = cfg.qk_head_dim;
        let vh = cfg.v_head_dim;
        let h = cfg.num_heads;
        let kvh = cfg.num_kv_heads;
        let kdim = kvh * qh;
        let vdim = kvh * vh;
        let qdim = h * qh;
        let odim = h * vh;
        let taps = cfg.qkv_conv_taps;
        let lw = &self.layers[li];

        matvec(&lw.q_proj, attn_in, &mut b.q[..qdim]);
        matvec(&lw.k_proj, attn_in, &mut b.k[..kdim]);
        matvec(&lw.v_proj, attn_in, &mut b.v[..vdim]);

        // causal depthwise conv over the last `taps` raw positions; tap 0 scales
        // the current position (trained, not necessarily 1)
        if taps > 0 {
            let ring = taps - 1;
            let qslot = pos % ring.max(1);
            let kslot = pos % ring.max(1);
            let vslot = pos % ring.max(1);
            for i in 0..qdim {
                b.qc[i] = lw.q_taps[i] * b.q[i];
            }
            for i in 0..kdim {
                b.kc[i] = lw.k_taps[i] * b.k[i];
            }
            for i in 0..vdim {
                b.vc[i] = lw.v_taps[i] * b.v[i];
            }
            for j in 1..taps {
                if pos >= j {
                    let qs = (pos + ring - j) % ring;
                    let ks = (pos + ring - j) % ring;
                    let vs = (pos + ring - j) % ring;
                    for i in 0..qdim {
                        b.qc[i] += lw.q_taps[j * qdim + i] * cache.rq[li][qs * qdim + i];
                    }
                    for i in 0..kdim {
                        b.kc[i] += lw.k_taps[j * kdim + i] * cache.rk[li][ks * kdim + i];
                    }
                    for i in 0..vdim {
                        b.vc[i] += lw.v_taps[j * vdim + i] * cache.rv[li][vs * vdim + i];
                    }
                }
            }
            if ring > 0 {
                cache.rq[li][qslot * qdim..(qslot + 1) * qdim].copy_from_slice(&b.q[..qdim]);
                cache.rk[li][kslot * kdim..(kslot + 1) * kdim].copy_from_slice(&b.k[..kdim]);
                cache.rv[li][vslot * vdim..(vslot + 1) * vdim].copy_from_slice(&b.v[..vdim]);
            }
            std::mem::swap(&mut b.q, &mut b.qc);
            std::mem::swap(&mut b.k, &mut b.kc);
            std::mem::swap(&mut b.v, &mut b.vc);
        }

        // q/k norm (per head over qh) + rope
        for hh in 0..h {
            let qseg = &mut b.q[hh * qh..(hh + 1) * qh];
            let mut normed = vec![0.0f32; qh];
            zc_rms_norm(qseg, &lw.q_norm, EPS, &mut normed);
            qseg.copy_from_slice(&normed);
        }
        for kk in 0..kvh {
            let kseg = &mut b.k[kk * qh..(kk + 1) * qh];
            let mut normed = vec![0.0f32; qh];
            zc_rms_norm(kseg, &lw.k_norm, EPS, &mut normed);
            kseg.copy_from_slice(&normed);
        }
        let rope = &self.rope;
        for hh in 0..h {
            let mut out = vec![0.0f32; qh];
            rope.apply(&b.q[hh * qh..(hh + 1) * qh], pos, &mut out);
            b.qn[hh * qh..(hh + 1) * qh].copy_from_slice(&out);
        }
        for kk in 0..kvh {
            let mut out = vec![0.0f32; qh];
            rope.apply(&b.k[kk * qh..(kk + 1) * qh], pos, &mut out);
            b.kn[kk * qh..(kk + 1) * qh].copy_from_slice(&out);
        }

        // cache current k/v (post conv + norm + rope) as f32: the JAX reference and
        // the shipped checkpoint's training path keep f32 KV; int8 KV caching
        // (kv_bits) is a deployed-engine memory optimization layered on top.
        {
            cache.k_q[li].extend_from_slice(&b.kn[..kdim]);
            cache.k_s[li].extend_from_slice(&vec![1.0f32; kvh]);
            cache.v_q[li].extend_from_slice(&b.v[..vdim]);
            cache.v_s[li].extend_from_slice(&vec![1.0f32; kvh]);
        }

        // attend
        let is_global = cfg.global_layers.contains(&li);
        let lo = if !is_global && cfg.sliding_window > 0 && pos + 1 > cfg.sliding_window {
            pos + 1 - cfg.sliding_window
        } else {
            0
        };
        let scale = 1.0 / (qh as f32).sqrt();
        let group = h / kvh;
        let span = pos + 1 - lo;
        let mut scores: Vec<Vec<f32>> = vec![vec![0.0f32; span]; h];
        for (si, p) in (lo..=pos).enumerate() {
            let kq = &cache.k_q[li][p * kdim..(p + 1) * kdim];
            let kscales = &cache.k_s[li][p * kvh..(p + 1) * kvh];
            for kk in 0..kvh {
                for i in 0..qh {
                    b.d_k[kk * qh + i] = kq[kk * qh + i] * kscales[kk];
                }
            }
            for (hh, sc) in scores.iter_mut().enumerate() {
                let kk = hh / group;
                let mut dot = 0.0f32;
                for i in 0..qh {
                    dot += b.qn[hh * qh + i] * b.d_k[kk * qh + i];
                }
                sc[si] = dot * scale;
            }
        }
        for (hh, sc) in scores.iter().enumerate() {
            let kk = hh / group;
            let w = softmax(sc);
            for i in 0..vh {
                b.attn_out[hh * vh + i] = 0.0;
            }
            for (si, p) in (lo..=pos).enumerate() {
                let vq = &cache.v_q[li][p * vdim..(p + 1) * vdim];
                let vscales = &cache.v_s[li][p * kvh..(p + 1) * kvh];
                for i in 0..vh {
                    b.d_v[kk * vh + i] = vq[kk * vh + i] * vscales[kk];
                }
                let wv = w[si];
                for i in 0..vh {
                    b.attn_out[hh * vh + i] += wv * b.d_v[kk * vh + i];
                }
            }
        }

        // output gate: out = attn_out * sigmoid(gate_proj(attn_in))
        matvec(&lw.gate_proj, attn_in, &mut b.gate[..odim]);
        for i in 0..odim {
            b.attn_out[i] *= sigmoid(b.gate[i]);
        }
        // out projection → b.attn_out becomes d-dim
        let mut out = vec![0.0f32; d];
        matvec(&lw.out_proj, &b.attn_out[..odim], &mut out);
        b.attn_out[..d].copy_from_slice(&out);
        Ok(())
    }

    fn hadamard(&self, lw: &crate::LayerWeights, x: &[f32], b: &mut Buffers) -> Result<()> {
        let cfg = &self.config;
        let d = cfg.d_model;
        let n_hada = cfg.hada_n;
        let (ba, bb) = Config::hada_blocks(n_hada);
        let rank = 8; // HADA_COND_RANK

        // cond = 1 + softmax(x @ cond_v) @ cond_u  → kept in b.cond [n_hada]
        let mut logits = [0.0f32; 8];
        for (r, slot) in logits.iter_mut().enumerate().take(rank) {
            let mut acc = 0.0f32;
            for (i, &xv) in x.iter().enumerate() {
                acc += xv * lw.cond_v[i * rank + r];
            }
            *slot = acc;
        }
        let sm = softmax(&logits);
        for j in 0..n_hada {
            let mut acc = 0.0f32;
            for (r, &smv) in sm.iter().enumerate().take(rank) {
                acc += smv * lw.cond_u[r * n_hada + j];
            }
            b.cond[j] = 1.0 + acc;
        }
        // z = pad(x) to n_hada, d1 * z
        for (j, &d1j) in lw.d1.iter().enumerate().take(n_hada) {
            let xv = if j < d { x[j] } else { 0.0 };
            b.hada[j] = d1j * xv;
        }
        // kron with (w1a, w1b), then permute p1
        kron_apply(&mut b.hada[..n_hada], ba, bb, &lw.w1a, &lw.w1b);
        permute(&mut b.hada[..n_hada], &self.hada_p1);
        // silu(d2 * cond * z + b2)
        for j in 0..n_hada {
            b.hada[j] = silu(lw.d2[j] * b.cond[j] * b.hada[j] + lw.b2[j]);
        }
        kron_apply(&mut b.hada[..n_hada], ba, bb, &lw.w2a, &lw.w2b);
        permute(&mut b.hada[..n_hada], &self.hada_p2);
        for j in 0..n_hada {
            b.hada[j] *= lw.d3[j];
        }
        kron_apply(&mut b.hada[..n_hada], ba, bb, &lw.w3a, &lw.w3b);
        for j in 0..d {
            b.hada[j] *= lw.d4[j];
        }
        Ok(())
    }

    /// Run `ids` through the network; returns last-position logits and,
    /// when `record_cells` is set, cells for every position.
    pub fn forward(
        &self,
        ids: &[u32],
        cache: &mut Cache,
        record_cells: bool,
    ) -> Result<(Vec<f32>, Option<Cells>)> {
        let start = cache.len;
        let mut logits = Vec::new();
        let mut cells = if record_cells {
            Some(Cells::new(
                start + ids.len(),
                self.config.num_layers + 1,
                self.config.d_model,
            ))
        } else {
            None
        };
        for (i, &id) in ids.iter().enumerate() {
            let pos = start + i;
            let last = i + 1 == ids.len();
            let out = self.step(id, pos, cache, cells.as_mut())?;
            if last {
                logits = out;
            }
        }
        Ok((logits, cells))
    }

    /// Full recompute over `ids` to obtain cells for the probe heads.
    pub fn cells_for(&self, ids: &[u32]) -> Result<Cells> {
        let mut cache = self.new_cache();
        let (_, cells) = self.forward(ids, &mut cache, true)?;
        cells.ok_or(Error::Shape("cells"))
    }
}

/// `z = (A ⊗ B) z` for z `[..., a*b]` row-major: out[k,l] = Σ_ij z[i,j] A[i,k] B[j,l],
/// i.e. `Aᵀ z B` — the matrices are trained (asymmetric), so multiply by the transpose.
fn kron_apply(z: &mut [f32], a: usize, b_sz: usize, wa: &[f32], wb: &[f32]) {
    // column transform (axis 0 of the a×b matrix): z1[k,j] = Σ_i wa[i,k] z[i,j]
    for j in 0..b_sz {
        let mut col: Vec<f32> = (0..a).map(|i| z[i * b_sz + j]).collect();
        matvec_t(&mut col, wa);
        for i in 0..a {
            z[i * b_sz + j] = col[i];
        }
    }
    // row transform (axis 1): out[k,l] = Σ_j z1[k,j] wb[j,l]
    for i in 0..a {
        matvec_t(&mut z[i * b_sz..(i + 1) * b_sz], wb);
    }
}

/// In-place `x ← Wᵀ x` with W row-major `n×n`: out[i] = Σ_j W[j,i] x[j].
fn matvec_t(x: &mut [f32], w: &[f32]) {
    let n = x.len();
    let src: Vec<f32> = x.to_vec();
    for i in 0..n {
        let mut acc = 0.0f32;
        for j in 0..n {
            acc += w[j * n + i] * src[j];
        }
        x[i] = acc;
    }
}

/// `out[i] = x[perm[i]]`
fn permute(x: &mut [f32], perm: &[f32]) {
    let n = x.len();
    let src: Vec<f32> = x.to_vec();
    for i in 0..n {
        x[i] = src[perm[i] as usize];
    }
}
