//! Differential dequant check against Python `read_export` on the real archive.
use needle_format::{dequant_cq, read_archive, Tensor};

#[test]
fn dequant_matches_python_on_real_archive() {
    let dir = std::env::var("NEEDLE_MODELS_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models"));
    let dump_path = dir.join("cq_dump.json");
    let cact = dir.join("needle3.cact");
    if !dump_path.exists() || !cact.exists() {
        eprintln!("skip: need needle3.cact + models/cq_dump.json");
        return;
    }
    let dump: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&dump_path).unwrap()).unwrap();
    let ar = read_archive(&cact).unwrap();
    let mut worst_all = 0.0f32;
    for entry in dump.as_array().unwrap() {
        let idx = entry["index"].as_u64().unwrap() as usize;
        let bits = entry["bits"].as_u64().unwrap() as u32;
        let shape: Vec<usize> = entry["shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        let Tensor::Cq(m) = &ar.tensors[idx] else { panic!("not cq") };
        assert_eq!(m.bits, bits, "bits mismatch at {idx}");
        let cb = ar.codebook_for(bits, m.group_size).unwrap();
        let w = dequant_cq(m, &cb).unwrap();
        assert_eq!(w.len(), shape[0] * shape[1]);
        for (row_s, vals) in entry["values"].as_object().unwrap() {
            let r: usize = row_s.parse().unwrap();
            let mut worst = 0.0f32;
            for (c, v) in vals.as_array().unwrap().iter().enumerate() {
                let want = v.as_f64().unwrap() as f32;
                worst = worst.max((want - w[r * shape[1] + c]).abs());
            }
            worst_all = worst_all.max(worst);
            assert!(worst < 2e-2, "tensor {idx} row {r} bits {bits}: worst {worst}");
        }
    }
    println!("all CQ tensors: worst sample diff = {worst_all:.6}");
}
