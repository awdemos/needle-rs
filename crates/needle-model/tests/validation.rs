//! Regression tests for validation and error paths: `step` refuses
//! non-contiguous positions and out-of-range token ids, and `from_archive`
//! rejects degenerate header geometries / malformed tensors with `Err`
//! instead of panicking later in the forward pass. All archives are built
//! in memory; no model file is required.
use needle_format::{Archive, Config, Tensor};
use needle_model::Model;

/// A small but complete geometry: every invariant `validate_geometry`
/// enforces holds here.
fn test_config() -> Config {
    Config {
        vocab_size: 16,
        out_vocab: 0,
        d_model: 8,
        num_heads: 2,
        num_kv_heads: 1,
        num_layers: 1,
        qk_head_dim: 4,
        v_head_dim: 4,
        max_seq_len: 32,
        hada_n: 8,
        mhc_lanes: 2,
        sliding_window: 0,
        global_layers: vec![],
        qkv_conv_taps: 2,
        engram_slots: 8,
        engram_sub_dim: 4,
        num_engram_tables: 2,
        engram_conv_taps: 2,
        engram_conv_dilation: 2,
        engram_seed_heads: 0,
        engram_orders: vec![2],
        engram_layers: vec![0],
        rope_theta: 10000.0,
        kv_window: 0,
        kv_bits: 8,
    }
}

fn t(n: usize) -> Tensor {
    // deterministic small values; never NaN/inf
    let data: Vec<f32> = (0..n).map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.01).collect();
    Tensor::Fp32 { shape: vec![n], data }
}

/// The canonical `_tensors` tensor order for `test_config`, with `qh` and
/// `etaps` overridable (they change q/k/v and engram tensor sizes).
fn tensors_for(cfg: &Config) -> Vec<Tensor> {
    let d = cfg.d_model;
    let h = cfg.num_heads;
    let kvh = cfg.num_kv_heads;
    let qh = cfg.qk_head_dim;
    let vh = cfg.v_head_dim;
    let hada_n = cfg.hada_n;
    let (ba, bb) = Config::hada_blocks(hada_n);
    let taps_n = cfg.qkv_conv_taps;
    let table_dim = cfg.num_engram_tables * cfg.engram_sub_dim;
    let mut ts = vec![t(cfg.vocab_size * d)];
    for _ in 0..cfg.num_layers {
        ts.push(t(d)); // norm_in
        ts.push(t(h * qh * d)); // q_proj
        ts.push(t(kvh * qh * d)); // k_proj
        ts.push(t(kvh * vh * d)); // v_proj
        if taps_n > 0 {
            ts.push(t(taps_n * h * qh)); // q_taps
            ts.push(t(taps_n * kvh * qh)); // k_taps
            ts.push(t(taps_n * kvh * vh)); // v_taps
        }
        ts.push(t(qh)); // q_norm
        ts.push(t(qh)); // k_norm
        ts.push(t(h * vh * d)); // gate_proj
        ts.push(t(d * h * vh)); // out_proj
        ts.push(t(d)); // post_norm
        ts.push(t(1)); // attn_gate
        ts.push(t(d)); // pre_hada
        for _ in 0..5 {
            ts.push(t(hada_n)); // d1, d2, b2, d3, d4
        }
        ts.push(t(ba * ba)); // w1a
        ts.push(t(bb * bb)); // w1b
        ts.push(t(ba * ba)); // w2a
        ts.push(t(bb * bb)); // w2b
        ts.push(t(ba * ba)); // w3a
        ts.push(t(bb * bb)); // w3b
        ts.push(t(d * 8)); // cond_v
        ts.push(t(8 * hada_n)); // cond_u
    }
    let n = cfg.mhc_lanes;
    let l = cfg.num_layers;
    ts.push(t(l)); // a_pre
    ts.push(t(l)); // a_post
    ts.push(t(l)); // a_res
    ts.push(t(l * n)); // b_pre
    ts.push(t(l * n)); // b_post
    ts.push(t(l * n * n)); // b_res
    ts.push(t(l * n * n * d)); // phi_pre
    ts.push(t(l * n * n * d)); // phi_post
    ts.push(t(l * n * n * n * d)); // phi_res
    let ident: Vec<f32> = (0..hada_n).map(|i| i as f32).collect();
    ts.push(Tensor::Fp32 { shape: vec![hada_n], data: ident.clone() }); // hada_p1
    ts.push(Tensor::Fp32 { shape: vec![hada_n], data: ident }); // hada_p2
    for _ in 0..cfg.engram_layers.len() {
        ts.push(t(cfg.num_engram_tables * cfg.engram_slots * cfg.engram_sub_dim)); // tables
        ts.push(t(d * table_dim)); // key_proj
        ts.push(t(d * table_dim)); // value_proj
        ts.push(t(cfg.engram_conv_taps * d)); // taps
    }
    ts.push(t(d)); // final_norm
    ts
}

fn archive(cfg: Config, tensors: Vec<Tensor>) -> Archive {
    Archive { config: cfg, codebook: vec![], records: vec![], tensors, raw: vec![] }
}

/// `Model` has no `Debug`, so `.unwrap_err()` is unavailable; `.err()` works.
fn err_of(r: needle_model::Result<Model>) -> String {
    r.err().expect("from_archive should have failed").to_string()
}

fn load_err(cfg: Config) -> String {
    // geometry validation runs before any tensor is read, so an empty
    // tensor list still surfaces the geometry error
    err_of(Model::from_archive(&archive(cfg, vec![])))
}

#[test]
fn rejects_bad_kv_heads() {
    let mut cfg = test_config();
    cfg.num_kv_heads = 0;
    assert!(load_err(cfg).contains("num_kv_heads"));
    let mut cfg = test_config();
    cfg.num_kv_heads = cfg.num_heads + 1;
    assert!(load_err(cfg).contains("num_kv_heads"));
    let mut cfg = test_config();
    cfg.num_heads = 3;
    cfg.num_kv_heads = 2;
    assert!(load_err(cfg).contains("divisible"));
}

#[test]
fn rejects_bad_engram_geometry() {
    let mut cfg = test_config();
    cfg.engram_orders = vec![];
    assert!(load_err(cfg).contains("engram_orders"));
    let mut cfg = test_config();
    cfg.num_engram_tables = 0;
    assert!(load_err(cfg).contains("multiple"));
    let mut cfg = test_config();
    cfg.num_engram_tables = 3; // not a multiple of len(orders) = 2
    cfg.engram_orders = vec![2, 2];
    assert!(load_err(cfg).contains("multiple"));
    let mut cfg = test_config();
    cfg.engram_slots = 0;
    assert!(load_err(cfg).contains("engram_slots"));
    let mut cfg = test_config();
    cfg.engram_conv_taps = 0;
    assert!(load_err(cfg).contains("engram_conv_taps"));
    let mut cfg = test_config();
    cfg.engram_layers = vec![cfg.num_layers];
    assert!(load_err(cfg).contains("engram_layers"));
    let mut cfg = test_config();
    cfg.engram_conv_dilation = 1; // != max(orders) = 2
    assert!(load_err(cfg).contains("engram_conv_dilation"));
}

#[test]
fn rejects_zero_mhc_lanes() {
    let mut cfg = test_config();
    cfg.mhc_lanes = 0;
    assert!(load_err(cfg).contains("mhc_lanes"));
}

#[test]
fn rejects_bad_tensor_lengths() {
    let cfg = test_config();
    let mut ts = tensors_for(&cfg);
    ts[1] = t(3); // norm_in must be d = 8
    assert!(err_of(Model::from_archive(&archive(cfg, ts))).contains("norm_in"));

    let cfg = test_config();
    let mut ts = tensors_for(&cfg);
    let bad_gate = ts.len() - 1; // final_norm
    ts[bad_gate] = t(2);
    assert!(err_of(Model::from_archive(&archive(cfg, ts))).contains("final_norm"));

    let cfg = test_config();
    let mut ts = tensors_for(&cfg);
    let n = ts.len();
    ts[n - 2] = t(4); // engram taps must be engram_conv_taps * d = 16
    assert!(err_of(Model::from_archive(&archive(cfg, ts))).contains("engram.taps"));
}

#[test]
fn rejects_empty_attn_gate() {
    let cfg = test_config();
    let mut ts = tensors_for(&cfg);
    // find attn_gate: norm_in(1) + q/k/v_proj(3) + taps(3) + q/k_norm(2)
    // + gate/out_proj(2) + post_norm(1) = 12th tensor (0-based index 12... )
    // after embedding; locate by its unique length-1 size among layer tensors
    let pos = ts
        .iter()
        .position(|t| matches!(t, Tensor::Fp32 { shape, data } if shape == &vec![1] && data.len() == 1))
        .unwrap();
    ts[pos] = Tensor::Fp32 { shape: vec![0], data: vec![] };
    assert!(err_of(Model::from_archive(&archive(cfg, ts))).contains("attn_gate"));
}

#[test]
fn rejects_bad_permutations() {
    // with one engram site the tail is [tables, key_proj, value_proj, taps,
    // final_norm]; hada_p1/hada_p2 sit just before it
    let tail = 6;
    let cfg = test_config();
    let mut ts = tensors_for(&cfg);
    let n = ts.len();
    // hada_p1: out-of-range value would panic in permute()
    ts[n - tail - 1] = Tensor::Fp32 {
        shape: vec![cfg.hada_n],
        data: vec![99.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    };
    assert!(err_of(Model::from_archive(&archive(cfg, ts))).contains("hada_p1"));

    let cfg = test_config();
    let mut ts = tensors_for(&cfg);
    let n = ts.len();
    // hada_p2: duplicate value (not a bijection)
    ts[n - tail] = Tensor::Fp32 {
        shape: vec![cfg.hada_n],
        data: vec![0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    };
    assert!(err_of(Model::from_archive(&archive(cfg, ts))).contains("hada_p2"));
}

#[test]
fn valid_archive_loads_and_steps() {
    let cfg = test_config();
    let model = Model::from_archive(&archive(cfg.clone(), tensors_for(&cfg))).unwrap();
    let mut cache = model.new_cache();
    let (logits, _) = model.forward(&[1, 2, 3], &mut cache, false).unwrap();
    assert_eq!(logits.len(), cfg.vocab_size);
    assert!(logits.iter().all(|v| v.is_finite()));
}

#[test]
fn step_rejects_non_contiguous_positions() {
    let cfg = test_config();
    let model = Model::from_archive(&archive(cfg.clone(), tensors_for(&cfg))).unwrap();
    let mut cache = model.new_cache();
    model.step(1, 0, &mut cache, None).unwrap();
    // gap: position 2 with only one cached position
    let err = model.step(2, 2, &mut cache, None).unwrap_err().to_string();
    assert!(err.contains("non-contiguous"), "{err}");
    // duplicate: position 0 again
    let err = model.step(1, 0, &mut cache, None).unwrap_err().to_string();
    assert!(err.contains("non-contiguous"), "{err}");
    // the failed calls must not have advanced the cache
    assert_eq!(cache.len, 1);
    model.step(2, 1, &mut cache, None).unwrap();
}

#[test]
fn step_rejects_out_of_range_token() {
    let cfg = test_config();
    let model = Model::from_archive(&archive(cfg.clone(), tensors_for(&cfg))).unwrap();
    let mut cache = model.new_cache();
    let err = model.step(cfg.vocab_size as u32, 0, &mut cache, None).unwrap_err().to_string();
    assert!(err.contains("token id out of range"), "{err}");
    let err = model.step(u32::MAX, 0, &mut cache, None).unwrap_err().to_string();
    assert!(err.contains("token id out of range"), "{err}");
}

/// `engram_conv_taps == 1` leaves no history ring; the ring write must be
/// skipped (as the attention conv rings already do), not divide by zero.
#[test]
fn engram_conv_taps_one_steps() {
    let mut cfg = test_config();
    cfg.engram_conv_taps = 1;
    let model = Model::from_archive(&archive(cfg.clone(), tensors_for(&cfg))).unwrap();
    let mut cache = model.new_cache();
    for pos in 0..3 {
        let logits = model.step(1, pos, &mut cache, None).unwrap();
        assert!(logits.iter().all(|v| v.is_finite()));
    }
}

/// `qk_head_dim > 64` must work, not overflow a fixed `[f32; 64]` stack array.
#[test]
fn qk_head_dim_128_steps() {
    let mut cfg = test_config();
    cfg.qk_head_dim = 128;
    let model = Model::from_archive(&archive(cfg.clone(), tensors_for(&cfg))).unwrap();
    let mut cache = model.new_cache();
    for pos in 0..2 {
        let logits = model.step(1, pos, &mut cache, None).unwrap();
        assert!(logits.iter().all(|v| v.is_finite()));
    }
}
