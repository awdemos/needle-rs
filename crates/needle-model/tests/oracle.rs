//! Differential test against the JAX reference on the exact archive bytes:
//! a tiny random model exported via Python `write_export`, then run in JAX from
//! the archive-reconstructed weights (`*_a`) — tight tolerance — plus the
//! tree-quantized references (`*_q`) as loose sanity (f16 storage noise on a
//! random net reaches ~0.13 on logits, so those stay coarse).
use needle_format::read_archive;
use needle_model::Model;

fn models_dir() -> std::path::PathBuf {
    std::env::var("NEEDLE_MODELS_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models")
        })
}

fn worst_diff(want: &serde_json::Value, got: &[f32], t: usize, n: usize) -> f32 {
    let row = want[t].as_array().unwrap();
    let mut worst = 0.0f32;
    for v in 0..n {
        let w = row[v].as_f64().unwrap() as f32;
        worst = worst.max((w - got[v]).abs());
    }
    worst
}

#[test]
fn matches_jax_tiny_model() {
    let dir = models_dir();
    let prefix = std::env::var("NEEDLE_TINY_PREFIX").unwrap_or_else(|_| "tiny".into());
    let oracle_path = dir.join(format!("{prefix}_oracle.json"));
    let cact = dir.join(format!("{prefix}.cact"));
    if !oracle_path.exists() || !cact.exists() {
        eprintln!("skip: run scripts/make_oracle.py first");
        return;
    }
    let oracle: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&oracle_path).unwrap()).unwrap();
    let tokens: Vec<u32> = oracle["tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let vocab = oracle["logits_a"][0].as_array().unwrap().len();

    let ar = read_archive(&cact).unwrap();
    let model = Model::from_archive(&ar).unwrap();

    // logits at every position
    let mut cache = model.new_cache();
    let mut got_logits: Vec<Vec<f32>> = Vec::new();
    let mut argmiss_a = 0;
    for (i, &t) in tokens.iter().enumerate() {
        let logits = model.step(t, i, &mut cache, None).unwrap();
        let want = &oracle["logits_a"][i];
        let want_am = want
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.as_f64().unwrap().partial_cmp(&b.1.as_f64().unwrap()).unwrap())
            .unwrap()
            .0;
        let got_am = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        if want_am != got_am {
            argmiss_a += 1;
        }
        got_logits.push(logits);
    }

    let mut worst_a = 0.0f32;
    let mut worst_q = 0.0f32;
    for t in 0..tokens.len() {
        worst_a = worst_a.max(worst_diff(&oracle["logits_a"], &got_logits[t], t, vocab));
        worst_q = worst_q.max(worst_diff(&oracle["logits_q"], &got_logits[t], t, vocab));
    }
    println!("logits: worst |diff| vs archive-ref = {worst_a:.6}, vs tree-ref = {worst_q:.6}, argmax misses vs archive-ref = {argmiss_a}/{}", tokens.len());
    assert!(worst_a < 2e-3, "logits diverge from the archive-exact reference: {worst_a}");
    assert!(argmiss_a == 0, "argmax mismatches vs archive reference");

    let cells = model.cells_for(&tokens).unwrap();
    let cells_a = oracle["cells_a"].as_array().unwrap();
    let mut worst_ca = 0.0f32;
    for t in 0..tokens.len() {
        for r in 0..cells.rows {
            let want_row = cells_a[t].as_array().unwrap()[r].as_array().unwrap();
            for i in 0..cells.d {
                let w = want_row[i].as_f64().unwrap() as f32;
                let g = cells.data[(t * cells.rows + r) * cells.d + i];
                worst_ca = worst_ca.max((w - g).abs());
            }
        }
    }
    println!("cells: worst |diff| vs archive-ref = {worst_ca:.6}");
    assert!(worst_ca < 2e-3, "cells diverge from the archive-exact reference: {worst_ca}");
}
