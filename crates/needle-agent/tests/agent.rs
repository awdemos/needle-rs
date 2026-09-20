//! Agent-level tests against the real model.
use needle_agent::Needle;
use serde_json::{json, Value};

fn models_dir() -> std::path::PathBuf {
    std::env::var("NEEDLE_MODELS_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models")
        })
}

fn smart_home_tools() -> Vec<Value> {
    serde_json::from_str(
        r#"[
        {"name":"set_lights","description":"Turn a room's lights on/off and set brightness","parameters":{"type":"object","properties":{"room":{"type":"string"},"on":{"type":"boolean"},"brightness":{"type":"integer","minimum":0,"maximum":100}},"required":["room","on"]}},
        {"name":"set_thermostat","description":"Set the thermostat temperature and mode","parameters":{"type":"object","properties":{"temperature":{"type":"integer"},"mode":{"type":"string","enum":["heat","cool","auto"]}},"required":["temperature"]}},
        {"name":"lock_door","description":"Lock or unlock a door","parameters":{"type":"object","properties":{"door":{"type":"string"},"locked":{"type":"boolean"}},"required":["door","locked"]}}
    ]"#,
    )
    .unwrap()
}

#[test]
fn complete_and_reset() {
    let path = models_dir().join("needle3.cact");
    if !path.exists() {
        eprintln!("skip");
        return;
    }
    let mut agent = Needle::new(&path, smart_home_tools(), None).unwrap();
    let r = agent.complete("turn on the kitchen lights").unwrap();
    assert_eq!(r.function_calls.len(), 1);
    assert_eq!(r.function_calls[0].name, "set_lights");
    agent.reset();
    let r2 = agent.complete("lock the back door").unwrap();
    assert_eq!(r2.function_calls.len(), 1);
    assert_eq!(r2.function_calls[0].name, "lock_door");
}

#[test]
fn run_loop_executes_tools() {
    let path = models_dir().join("needle3.cact");
    if !path.exists() {
        eprintln!("skip");
        return;
    }
    let mut agent = Needle::new(&path, smart_home_tools(), None).unwrap();
    let mut exec = |name: &str, args: &Value| -> Result<Value, String> {
        Ok(json!({"ok": true, "tool": name, "args": args}))
    };
    let r = agent.run("turn on the porch lights", &mut exec).unwrap();
    println!("run response: {}", serde_json::to_string_pretty(&r.to_json()).unwrap());
    assert!(r.results.is_some(), "expected executed results attached");
}

#[test]
fn extract_record() {
    let path = models_dir().join("needle3.cact");
    if !path.exists() {
        eprintln!("skip");
        return;
    }
    let mut agent = Needle::new(&path, Vec::new(), None).unwrap();
    let record = json!({
        "name": "invoice",
        "parameters": {
            "type": "object",
            "properties": {
                "vendor": {"type": "string"},
                "total": {"type": "number"}
            },
            "required": ["vendor", "total"]
        }
    });
    let out = agent
        .extract("Invoice from Acme Corp, total $1,200.00", &record, false)
        .unwrap();
    let Some(v) = out else { panic!("expected extraction") };
    println!("extracted: {v}");
    assert_eq!(v["vendor"], "Acme Corp");
}

#[test]
fn auto_date_fact() {
    let s = needle_agent::with_date_fact("");
    assert!(s.starts_with("date: 20"), "got {s}");
    let s2 = needle_agent::with_date_fact("date: 2026-01-01 Mon 09:00; locale: en-US");
    assert_eq!(s2, "date: 2026-01-01 Mon 09:00; locale: en-US");
}
