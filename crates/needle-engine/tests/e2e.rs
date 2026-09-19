//! End-to-end engine test against the real published model.
use needle_engine::needle_model::Model;
use needle_engine::{Engine, CONFIDENCE_FLOOR};
use needle_format::read_archive;
use needle_tokenizer::Tokenizer;

fn models_dir() -> std::path::PathBuf {
    std::env::var("NEEDLE_MODELS_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models")
        })
}

fn load() -> Option<(Model, Tokenizer, Vec<serde_json::Value>)> {
    let path = models_dir().join("needle3.cact");
    if !path.exists() {
        return None;
    }
    let ar = read_archive(&path).unwrap();
    let model = Model::from_archive(&ar).unwrap();
    let tok = Tokenizer::from_blob(ar.tokenizer_blob().unwrap()).unwrap();
    let tools: Vec<serde_json::Value> = serde_json::from_str(
        r#"[
        {"name":"set_lights","description":"Turn a room's lights on/off and set brightness","parameters":{"type":"object","properties":{"room":{"type":"string"},"on":{"type":"boolean"},"brightness":{"type":"integer","minimum":0,"maximum":100}},"required":["room","on"]}},
        {"name":"set_thermostat","description":"Set the thermostat temperature and mode","parameters":{"type":"object","properties":{"temperature":{"type":"integer"},"mode":{"type":"string","enum":["heat","cool","auto"]}},"required":["temperature"]}},
        {"name":"lock_door","description":"Lock or unlock a door","parameters":{"type":"object","properties":{"door":{"type":"string"},"locked":{"type":"boolean"}},"required":["door","locked"]}}
    ]"#,
    )
    .unwrap();
    Some((model, tok, tools))
}

#[test]
fn complete_produces_tool_call() {
    let Some((model, tok, tools)) = load() else {
        eprintln!("skip: needle3.cact not downloaded");
        return;
    };
    let engine = Engine::new(&model, tok, tools);
    let mut cache = model.new_cache();
    let mut history: Vec<u32> = Vec::new();
    let r = engine
        .complete_turn(
            "turn on the living room lights",
            &mut cache,
            &mut history,
            None,
            512,
        )
        .unwrap();
    println!("response: {}", serde_json::to_string_pretty(&r.to_json()).unwrap());
    assert_eq!(r.r#type, "call");
    assert_eq!(r.function_calls.len(), 1);
    let call = &r.function_calls[0];
    assert_eq!(call.name, "set_lights");
    assert_eq!(call.arguments["room"], "living room");
    assert_eq!(call.arguments["on"], true);
    assert!(!r.reasoning.is_empty(), "expected reasoning");
    let c = r.confidence.expect("confidence");
    assert!((0.0..=1.0).contains(&c), "confidence in range: {c}");
    assert!(c >= CONFIDENCE_FLOOR, "expected unsuppressed: {c}");
}

#[test]
fn off_topic_is_refusal() {
    let Some((model, tok, tools)) = load() else {
        eprintln!("skip: needle3.cact not downloaded");
        return;
    };
    let engine = Engine::new(&model, tok, tools);
    let mut cache = model.new_cache();
    let mut history: Vec<u32> = Vec::new();
    let r = engine
        .complete_turn(
            "what's the capital of France?",
            &mut cache,
            &mut history,
            None,
            512,
        )
        .unwrap();
    println!("response: {}", serde_json::to_string_pretty(&r.to_json()).unwrap());
    assert!(r.function_calls.is_empty(), "expected refusal, got {:?}", r.function_calls);
}

#[test]
fn grammar_constrains_enum_and_second_call() {
    let Some((model, tok, tools)) = load() else {
        eprintln!("skip: needle3.cact not downloaded");
        return;
    };
    let engine = Engine::new(&model, tok, tools);
    let mut cache = model.new_cache();
    let mut history: Vec<u32> = Vec::new();
    let r = engine
        .complete_turn(
            "set the thermostat to 22 degrees and lock the front door",
            &mut cache,
            &mut history,
            None,
            512,
        )
        .unwrap();
    println!("response: {}", serde_json::to_string_pretty(&r.to_json()).unwrap());
    assert_eq!(r.function_calls.len(), 2, "expected two calls");
    assert_eq!(r.function_calls[0].name, "set_thermostat");
    assert_eq!(r.function_calls[1].name, "lock_door");
}
