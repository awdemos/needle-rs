use needle_build::Checkpoint;

fn main() {
    let ckpt = Checkpoint::load(std::path::Path::new("models/needle3.safetensors")).unwrap();
    let ar = needle_format::read_archive(std::path::Path::new("models/needle3.cact")).unwrap();
    let blob = ar.tokenizer_blob().unwrap().to_vec();
    ckpt.write_cact(std::path::Path::new("models/needle3-rebuilt.cact"), Some(&blob), 4, 128).unwrap();
    println!("rebuilt: {} bytes", std::fs::metadata("models/needle3-rebuilt.cact").unwrap().len());

    let ar2 = needle_format::read_archive(std::path::Path::new("models/needle3-rebuilt.cact")).unwrap();
    let model = needle_model::Model::from_archive(&ar2).unwrap();
    let tok = needle_tokenizer::Tokenizer::from_blob(ar2.tokenizer_blob().unwrap()).unwrap();
    let tools: Vec<serde_json::Value> = serde_json::from_str(
        r#"[{"name":"set_lights","description":"Turn a room's lights on/off and set brightness","parameters":{"type":"object","properties":{"room":{"type":"string"},"on":{"type":"boolean"},"brightness":{"type":"integer","minimum":0,"maximum":100}},"required":["room","on"]}}]"#,
    ).unwrap();
    let engine = needle_engine::Engine::new(&model, tok, tools);
    let mut cache = model.new_cache();
    let mut history: Vec<u32> = Vec::new();
    let r = engine.complete_turn("turn on the living room lights", &mut cache, &mut history, None, 512).unwrap();
    println!("response: {}", serde_json::to_string(&r.to_json()).unwrap());
}
