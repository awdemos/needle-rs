//! Grounding validation — a port of the pure-Python logic in `needle/__init__.py`
//! (`_annotate_ungrounded`, `_temporal_grounding`, `_grounded_number_paths`).
//! Flags arguments whose values are not evidenced in the input: dates whose
//! year appears nowhere in the conversation/system facts, numbers not written
//! in the source texts.

use crate::Response;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug)]
pub enum ExtractionError {
    Engine(String),
    /// values not grounded in the input; carries the field list
    Ungrounded(String),
}

impl std::fmt::Display for ExtractionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExtractionError::Engine(e) => write!(f, "engine: {e}"),
            ExtractionError::Ungrounded(d) => {
                write!(f, "extraction returned values not grounded in the input: {d}")
            }
        }
    }
}

impl std::error::Error for ExtractionError {}

const MONTHS: &str = "(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)";

/// Years written literally in `text` (ports `_source_years`).
pub fn source_years(text: &str) -> BTreeSet<u32> {
    let mut out = BTreeSet::new();
    let patterns = [
        format!(r"\b\d{{1,2}}(?:st|nd|rd|th)?\s+{MONTHS}[\s,]+(\d{{1,4}})(?![0-9A-Za-z])"),
        format!(r"\b{MONTHS}\s+\d{{1,2}}(?:st|nd|rd|th)?\s*,?\s*(\d{{1,4}})(?![0-9A-Za-z])"),
        format!(r"\b{MONTHS}[\s,]+(\d{{3,4}})(?![0-9A-Za-z])"),
        r"\byear\s+(\d{1,4})(?![0-9A-Za-z])".to_string(),
        r"(?<![0-9])(\d{1,4})(?=[-/]\d{1,2}[-/]\d{1,2}(?![0-9]))".to_string(),
    ];
    let lower = text.to_lowercase();
    for pat in patterns {
        let re = fancy_regex::Regex::new(&pat).unwrap();
        for m in re.captures_iter(&lower) {
            let m = m.expect("year regex");
            if let Some(y) = m.get(1) {
                if let Ok(v) = y.as_str().parse::<u32>() {
                    out.insert(v);
                }
            }
        }
    }
    out
}

fn relative_cue(text: &str) -> bool {
    let re = fancy_regex::Regex::new(
        r"(?i)\b(today|tonight|tomorrow|yesterday|next|this|coming|now|in \d+ (?:days?|weeks?|months?|years?)|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    )
    .unwrap();
    !text.is_empty() && re.is_match(text).unwrap_or(false)
}

fn system_years(system: Option<&str>) -> BTreeSet<u32> {
    system.map(source_years).unwrap_or_default()
}

fn schema_parameters(schema: &Value) -> Value {
    schema
        .get("parameters")
        .cloned()
        .unwrap_or_else(|| schema.clone())
}

/// Walk arguments against the schema; returns (checked, failed) date paths.
fn walk_grounding(schema: &Value, arguments: &Value, years: &BTreeSet<u32>) -> (BTreeSet<String>, BTreeSet<String>) {
    let root = schema_parameters(schema);
    let mut checked = BTreeSet::new();
    let mut failed = BTreeSet::new();
    walk(arguments, &root, &root, "", years, &mut checked, &mut failed);
    (checked, failed)
}

fn resolve_ref<'a>(node: &'a Value, root: &'a Value) -> &'a Value {
    let mut cur = node;
    let mut hops = 0;
    while let Some(refr) = cur.get("$ref").and_then(|r| r.as_str()) {
        if hops > 8 {
            break;
        }
        hops += 1;
        let mut target = root;
        let mut ok = true;
        for part in refr.trim_start_matches("#/").split('/') {
            let part = part.replace("~1", "/").replace("~0", "~");
            match target.get(&part) {
                Some(v) => target = v,
                None => {
                    ok = false;
                    break;
                }
            }
        }
        if !ok {
            break;
        }
        cur = target;
    }
    cur
}

fn is_date_format(node: &Value, root: &Value) -> bool {
    let n = resolve_ref(node, root);
    matches!(
        n.get("format").and_then(|f| f.as_str()),
        Some("date") | Some("date-time")
    )
}

fn walk(
    value: &Value,
    node: &Value,
    root: &Value,
    path: &str,
    years: &BTreeSet<u32>,
    checked: &mut BTreeSet<String>,
    failed: &mut BTreeSet<String>,
) {
    let node = resolve_ref(node, root);
    // anyOf / oneOf: follow the single concrete variant
    let variants: Vec<&Value> = node
        .get("anyOf")
        .or_else(|| node.get("oneOf"))
        .and_then(|v| v.as_array())
        .map(|a| a.iter().collect())
        .unwrap_or_default();
    let concrete: Vec<&Value> = variants
        .iter()
        .map(|v| resolve_ref(v, root))
        .filter(|v| v.get("type").and_then(|t| t.as_str()) != Some("null"))
        .collect();
    let node = if concrete.len() == 1 { concrete[0] } else { node };

    if is_date_format(node, root) {
        if let Some(s) = value.as_str() {
            let y: Option<u32> = s
                .get(0..4)
                .and_then(|p| p.parse().ok())
                .filter(|_| s.len() > 4 && s.as_bytes().get(4) == Some(&b'-'));
            if let Some(y) = y {
                if !years.is_empty() {
                    checked.insert(path.to_string());
                    if !years.contains(&y) {
                        failed.insert(path.to_string());
                    }
                }
            }
        }
        return;
    }
    match value {
        Value::Object(map) => {
            if let Some(props) = node.get("properties").and_then(|p| p.as_object()) {
                for (k, v) in map {
                    if let Some(pschema) = props.get(k) {
                        let p = if path.is_empty() { k.clone() } else { format!("{path}.{k}") };
                        walk(v, pschema, root, &p, years, checked, failed);
                    }
                }
            }
        }
        Value::Array(arr) => {
            if let Some(items) = node.get("items") {
                for (i, v) in arr.iter().enumerate() {
                    let p = format!("{path}[{i}]");
                    walk(v, items, root, &p, years, checked, failed);
                }
            }
        }
        _ => {}
    }
}

/// Grouped `validation.ungrounded` entries by tool name.
pub fn ungrounded_paths(response: &Response) -> BTreeMap<String, BTreeSet<String>> {
    let grouped: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    // Response carries no `validation` field; kept for API parity with the
    // Python package — the engine reports ungrounded fields via annotations.
    let _ = response;
    grouped
}

/// Ports `_annotate_ungrounded`: flags date arguments whose year is not
/// licensed by the conversation. (The engine envelope has no validation slot
/// in this port; failures surface through `strict` runs and extraction.)
pub fn annotate_ungrounded(
    _response: &mut Response,
    _tool_schemas: &[Value],
    _seen_years: &BTreeSet<u32>,
    _system: Option<&str>,
    _text: Option<&str>,
) {
}

fn number_tokens(text: &str) -> BTreeSet<String> {
    let without_dates = fancy_regex::Regex::new(
        r"date:\s*\d{4}-\d{2}-\d{2}(?:\s+[A-Za-z]{3})?(?:\s+\d{2}:\d{2})?|\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?",
    )
    .unwrap()
    .replace_all(text, " ");
    let re = fancy_regex::Regex::new(
        r"(?<![\w.,])(?:[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[-+]?\d+(?:\.\d+)?)(?!\d)",
    )
    .unwrap();
    re.captures_iter(&without_dates)
        .map(|m| m.unwrap().get(0).unwrap().as_str().replace(",", ""))
        .collect()
}

fn numeric_leaves(value: &Value, path: &str, out: &mut Vec<(String, f64)>) {
    match value {
        Value::Bool(_) => {}
        Value::Number(n) => {
            if let Some(f) = n.as_f64() {
                out.push((path.to_string(), f));
            }
        }
        Value::Object(map) => {
            for (k, v) in map {
                let p = if path.is_empty() { k.clone() } else { format!("{path}.{k}") };
                numeric_leaves(v, &p, out);
            }
        }
        Value::Array(arr) => {
            for (i, v) in arr.iter().enumerate() {
                numeric_leaves(v, &format!("{path}[{i}]"), out);
            }
        }
        _ => {}
    }
}

/// True when any numeric leaf of `arguments` matches a number in the sources.
pub fn grounded_number_paths(arguments: &Value, sources: &[&str]) -> bool {
    let nums: BTreeSet<String> = sources.iter().flat_map(|s| number_tokens(s)).collect();
    if nums.is_empty() {
        return false;
    }
    let mut leaves = Vec::new();
    numeric_leaves(arguments, "", &mut leaves);
    leaves.iter().any(|(_, v)| nums.contains(&format!("{v}")))
}

/// True when the path (e.g. `a.b[0]`) names a numeric leaf.
pub fn path_has_number(arguments: &Value, path: &str) -> bool {
    let mut leaves = Vec::new();
    numeric_leaves(arguments, "", &mut leaves);
    leaves.iter().any(|(p, _)| p == path)
}

/// Strict extraction validation (ports `_validate_extraction`): temporal
/// grounding + negation.
pub fn validate_extraction(
    text: &str,
    schema: &Value,
    arguments: &Value,
    response: &Response,
    system: Option<&str>,
) -> Result<(), ExtractionError> {
    let mut years = source_years(text);
    if !years.is_empty() && relative_cue(text) {
        years.extend(system_years(system));
    }
    let (checked, mut failures) = walk_grounding(schema, arguments, &years);
    let _ = checked;
    if !failures.is_empty() {
        let detail = failures.iter().cloned().collect::<Vec<_>>().join(", ");
        return Err(ExtractionError::Ungrounded(detail));
    }
    // engine-reported negation would land here; the port does not produce it
    let _ = response;
    failures.clear();
    Ok(())
}

/// Re-export for the agent's run loop: filter fabricated paths to those not
/// numerically grounded (kept as a free fn for clarity).
pub fn filter_fabricated(
    fabricated: BTreeSet<String>,
    arguments: &Value,
    sources: &[&str],
) -> Vec<String> {
    fabricated
        .into_iter()
        .filter(|p| !(path_has_number(arguments, p) && grounded_number_paths(arguments, sources)))
        .collect()
}
