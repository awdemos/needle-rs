use needle_model::Model;
use needle_format::read_archive;
use needle_tokenizer::{Tokenizer, BOS_ID, EOS_ID, IM_START, IM_END, TOOLS_START, TOOLS_END, TOOL_CALL_START, TOOL_CALL_END, THINK_START};

fn models_dir() -> std::path::PathBuf {
    std::env::var("NEEDLE_MODELS_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models"))
}

#[test]
fn loads_real_weights() {
    let path = models_dir().join("needle3.cact");
    if !path.exists() {
        eprintln!("skip: needle3.cact not downloaded");
        return;
    }
    let t0 = std::time::Instant::now();
    let ar = read_archive(&path).unwrap();
    let model = Model::from_archive(&ar).unwrap();
    println!("loaded in {:?}; heads: {:?}", t0.elapsed(), model.heads.iter().map(|h| h.kind).collect::<Vec<_>>());
    assert_eq!(model.embedding.len(), ar.config.vocab_size * ar.config.d_model);
    assert_eq!(model.layers.len(), 20);
    assert_eq!(model.engrams.len(), 5);
}

#[test]
fn greedy_tool_call() {
    let path = models_dir().join("needle3.cact");
    if !path.exists() {
        eprintln!("skip: needle3.cact not downloaded");
        return;
    }
    let ar = read_archive(&path).unwrap();
    let model = Model::from_archive(&ar).unwrap();
    let tok = Tokenizer::from_blob(ar.tokenizer_blob().unwrap()).unwrap();

    let tools = r#"[{"name":"set_lights","description":"Turn a room's lights on/off and set brightness","parameters":{"type":"object","properties":{"room":{"type":"string"},"on":{"type":"boolean"},"brightness":{"type":"integer","minimum":0,"maximum":100}},"required":["room","on"]}}]"#;
    let prompt = format!(
        "{IM_START}user\n{TOOLS_START}{tools}{TOOLS_END}\ndim the living room lights to 30{IM_END}\n{IM_START}assistant\n"
    );
    let mut ids = vec![BOS_ID];
    ids.extend(tok.encode(&prompt));

    let t0 = std::time::Instant::now();
    let mut cache = model.new_cache();
    let (mut logits, _) = model.forward(&ids, &mut cache, false).unwrap();
    let prefill_t = t0.elapsed();
    println!("prefill {} tokens in {:?} ({:.0} t/s)", ids.len(), prefill_t, ids.len() as f64 / prefill_t.as_secs_f64());

    let mut generated: Vec<u32> = Vec::new();
    let t1 = std::time::Instant::now();
    for _ in 0..96 {
        let (next, _) = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        if next == EOS_ID as usize {
            break;
        }
        generated.push(next as u32);
        logits = model.step(next as u32, cache.len, &mut cache, None).unwrap();
    }
    let dec_t = t1.elapsed();
    println!("decode {} tokens in {:?} ({:.0} t/s)", generated.len(), dec_t, generated.len() as f64 / dec_t.as_secs_f64());
    let text = tok.decode(&generated);
    println!("generated: {text:?}");
    assert!(text.contains(TOOL_CALL_START), "expected a tool_call block, got {text:?}");
    assert!(text.contains(TOOL_CALL_END), "expected closed tool_call block");
    // the reasoning may be absent or present; the JSON must name the tool
    assert!(text.contains("set_lights"), "expected the right tool, got {text:?}");
}
