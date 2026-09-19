//! The smart_home environment — a Rust port of `needle/environments/smart_home.py`.
//! Run: `cargo run --release --example-bin smart_home -- --model ../../models/needle3.cact`
//! (or `cargo run --release` from this directory with the model path as an arg).

use needle_agent::Needle;
use serde_json::{json, Value};

pub const SYSTEM: &str = "A small apartment: living room, kitchen, bedroom, study. Doors: front, back.";

pub fn tools() -> Vec<Value> {
    serde_json::from_str(
        r#"[
        {"name":"set_lights","description":"Turn a room's lights on/off and set brightness","parameters":{"type":"object","properties":{"room":{"type":"string"},"on":{"type":"boolean"},"brightness":{"type":"integer","minimum":0,"maximum":100}},"required":["room","on"]}},
        {"name":"set_thermostat","description":"Set the thermostat temperature and mode","parameters":{"type":"object","properties":{"temperature":{"type":"integer"},"mode":{"type":"string","enum":["heat","cool","auto"]}},"required":["temperature"]}},
        {"name":"lock_door","description":"Lock or unlock a door","parameters":{"type":"object","properties":{"door":{"type":"string"},"locked":{"type":"boolean"}},"required":["door","locked"]}},
        {"name":"play_music","description":"Play music by genre in a room","parameters":{"type":"object","properties":{"genre":{"type":"string","enum":["rock","jazz","classical","pop"]},"room":{"type":"string"}},"required":["genre"]}},
        {"name":"close_blinds","description":"Open or close the blinds in a room","parameters":{"type":"object","properties":{"room":{"type":"string"},"closed":{"type":"boolean"}},"required":["room","closed"]}}
    ]"#,
    )
    .unwrap()
}

pub const TEST_CASES: &[(&str, &str)] = &[
    ("dim the living room lights to 30", "set_lights"),
    ("turn off the kitchen lights", "set_lights"),
    ("set the thermostat to 22 degrees", "set_thermostat"),
    ("lock the front door", "lock_door"),
    ("play some jazz in the study", "play_music"),
    ("close the bedroom blinds", "close_blinds"),
    ("what's the weather like", ""), // refusal
];

fn main() {
    let model = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "../../models/needle3.cact".to_string());
    let mut agent = Needle::new(std::path::Path::new(&model), tools(), Some(SYSTEM.into()))
        .expect("agent");

    let mut exec = |name: &str, args: &Value| -> Result<Value, String> {
        println!("  -> executing {name} {args}");
        Ok(json!({"ok": true, "tool": name}))
    };

    let mut passed = 0;
    for (query, want_tool) in TEST_CASES {
        let r = agent.run(query, &mut exec).unwrap();
        // executed tool names surface in the attached results (the final turn
        // after result feedback carries no further calls)
        let executed_names: Vec<String> = r
            .results
            .as_ref()
            .and_then(|v| v.as_array().cloned())
            .unwrap_or_default()
            .iter()
            .filter_map(|v| v.get("tool").and_then(|t| t.as_str()).map(String::from))
            .collect();
        let got = r
            .function_calls
            .first()
            .map(|c| c.name.as_str())
            .or_else(|| r.suppressed_calls.first().map(|c| c.name.as_str()))
            .or_else(|| executed_names.first().map(|s| s.as_str()))
            .unwrap_or("");
        let ok = got == *want_tool;
        passed += ok as usize;
        println!(
            "[{}] {query:?}\n    want {want_tool:?} got {got:?} conf {:?} type {}\n    reasoning: {}",
            if ok { "PASS" } else { "FAIL" },
            r.confidence,
            r.r#type,
            r.reasoning.lines().next().unwrap_or("")
        );
        agent.reset();
    }
    println!("smart_home acceptance: {passed}/{}", TEST_CASES.len());
}
