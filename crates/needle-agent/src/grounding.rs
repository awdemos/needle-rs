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

/// Grouped `validation.ungrounded` entries by tool name (ports
/// `_ungrounded_paths`): each `tool.path` entry lands under `tool`, with an
/// entry lacking a dot grouping under its own name.
pub fn ungrounded_paths(response: &Response) -> BTreeMap<String, BTreeSet<String>> {
    let mut grouped: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let Some(list) = response
        .validation
        .as_ref()
        .and_then(|v| v.get("ungrounded"))
        .and_then(|u| u.as_array())
    else {
        return grouped;
    };
    for name in list {
        let Some(name) = name.as_str() else { continue };
        let (tool, path) = name.split_once('.').unwrap_or((name, name));
        grouped.entry(tool.to_string()).or_default().insert(path.to_string());
    }
    grouped
}

/// Licensed years for date arguments: years written in the input, plus the
/// system date's year when the input reasons relatively (ports
/// `_licensed_years`).
fn licensed_years(seen_years: &BTreeSet<u32>, system: Option<&str>, relative: bool) -> BTreeSet<u32> {
    let mut years = seen_years.clone();
    if !years.is_empty() && relative {
        if let Some(system) = system {
            years.extend(source_years(system));
        }
    }
    years
}

/// Ports `_annotate_ungrounded`: flags date arguments whose year is not
/// licensed by the conversation, appending `tool.path` names to
/// `response.validation.ungrounded` (merged with any engine-reported names).
pub fn annotate_ungrounded(
    response: &mut Response,
    tool_schemas: &[Value],
    seen_years: &BTreeSet<u32>,
    system: Option<&str>,
    text: Option<&str>,
) {
    if response.function_calls.is_empty() {
        return;
    }
    let relative = text.map(relative_cue).unwrap_or(true);
    let years = licensed_years(seen_years, system, relative);
    if years.is_empty() {
        return;
    }
    let schemas: BTreeMap<&str, &Value> = tool_schemas
        .iter()
        .filter_map(|e| e.get("name").and_then(|n| n.as_str()).map(|n| (n, e)))
        .collect();
    let mut found: Vec<String> = Vec::new();
    for call in &response.function_calls {
        let Some(schema) = schemas.get(call.name.as_str()) else { continue };
        let (_checked, failures) = walk_grounding(schema, &call.arguments, &years);
        for path in &failures {
            found.push(format!("{}.{}", call.name, path));
        }
    }
    if found.is_empty() {
        return;
    }
    let validation = response.validation.get_or_insert_with(|| serde_json::json!({}));
    let mut ungrounded = validation
        .get("ungrounded")
        .and_then(|u| u.as_array())
        .cloned()
        .unwrap_or_default();
    for name in found {
        let entry = Value::String(name);
        if !ungrounded.contains(&entry) {
            ungrounded.push(entry);
        }
    }
    validation["ungrounded"] = Value::Array(ungrounded);
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

/// JSON number rendered the way Python's `str(value)` would render it, so
/// decimal equality matches `decimal.Decimal(str(value))`: integers keep
/// their exact digits, floats use their shortest round-trip form.
fn number_string(n: &serde_json::Number) -> String {
    if let Some(i) = n.as_i64() {
        return i.to_string();
    }
    if let Some(u) = n.as_u64() {
        return u.to_string();
    }
    format!("{}", n.as_f64().unwrap_or(f64::NAN))
}

fn numeric_leaves(value: &Value, path: &str, out: &mut Vec<(String, String)>) {
    match value {
        Value::Bool(_) => {}
        Value::Number(n) => out.push((path.to_string(), number_string(n))),
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

/// Canonical decimal form (sign, significant digits, base-10 exponent) for
/// Decimal-style numeric equality: "1,200.00", "1200" and 1200.0 all
/// normalize to the same triple, so grounding compares values, not strings.
fn canonical_decimal(s: &str) -> Option<(bool, String, i32)> {
    let s = s.trim().replace(',', "");
    let (neg, s) = match s.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, s.strip_prefix('+').unwrap_or(&s)),
    };
    let (mantissa, exp10) = match s.split_once(['e', 'E']) {
        Some((m, e)) => (m, e.parse::<i32>().ok()?),
        None => (s, 0),
    };
    let mut digits = String::new();
    let mut frac: i32 = 0;
    let mut seen_dot = false;
    for c in mantissa.chars() {
        match c {
            '0'..='9' => {
                digits.push(c);
                if seen_dot {
                    frac -= 1;
                }
            }
            '.' if !seen_dot => seen_dot = true,
            _ => return None,
        }
    }
    if digits.is_empty() {
        return None;
    }
    // value = digits (as integer) * 10^(frac + exp10)
    let mut exp = frac.checked_add(exp10)?;
    let significant = digits.trim_start_matches('0');
    let trimmed = significant.trim_end_matches('0');
    exp = exp.checked_add((significant.len() - trimmed.len()) as i32)?;
    if trimmed.is_empty() {
        // zero (possibly negative zero, which Decimal treats as 0)
        return Some((false, String::new(), 0));
    }
    Some((neg, trimmed.to_string(), exp))
}

/// Per-path numeric grounding (ports `_grounded_number_paths`): the set of
/// numeric argument paths whose value is written literally in the sources.
pub fn grounded_number_paths(arguments: &Value, sources: &[&str]) -> BTreeSet<String> {
    let numbers: std::collections::HashSet<(bool, String, i32)> = sources
        .iter()
        .flat_map(|s| number_tokens(s))
        .filter_map(|t| canonical_decimal(&t))
        .collect();
    let mut grounded = BTreeSet::new();
    if numbers.is_empty() {
        return grounded;
    }
    let mut leaves = Vec::new();
    numeric_leaves(arguments, "", &mut leaves);
    for (path, value) in leaves {
        if let Some(d) = canonical_decimal(&value) {
            if numbers.contains(&d) {
                grounded.insert(path);
            }
        }
    }
    grounded
}

/// Strict-run filtering (ports the `run` loop): drop fabricated paths whose
/// numeric value is grounded in the sources; the rest block the call.
pub fn filter_fabricated(
    fabricated: BTreeSet<String>,
    arguments: &Value,
    sources: &[&str],
) -> Vec<String> {
    if fabricated.is_empty() {
        return Vec::new();
    }
    let grounded = grounded_number_paths(arguments, sources);
    fabricated.into_iter().filter(|p| !grounded.contains(p)).collect()
}

/// Strict extraction validation (ports `_validate_extraction`): temporal
/// grounding, plus numeric grounding and negation for engine-reported flags.
pub fn validate_extraction(
    text: &str,
    schema: &Value,
    arguments: &Value,
    response: &Response,
    system: Option<&str>,
) -> Result<(), ExtractionError> {
    let years = licensed_years(&source_years(text), system, relative_cue(text));
    let (checked, mut failures) = walk_grounding(schema, arguments, &years);
    if let Some(validation) = response.validation.as_ref() {
        let flagged = validation
            .get("ungrounded")
            .and_then(|u| u.as_array())
            .cloned()
            .unwrap_or_default();
        if !flagged.is_empty() {
            let sources = [text, system.unwrap_or_default()];
            let grounded = grounded_number_paths(arguments, &sources);
            for name in &flagged {
                let Some(name) = name.as_str() else { continue };
                let path = name.split_once('.').map(|(_, p)| p).unwrap_or(name);
                if grounded.contains(path) {
                    continue;
                }
                if !checked.contains(path) || failures.contains(path) {
                    failures.insert(path.to_string());
                }
            }
        }
        if validation
            .get("negation")
            .and_then(|n| n.as_bool())
            .unwrap_or(false)
        {
            failures.insert("negated request".to_string());
        }
    }
    if !failures.is_empty() {
        let detail = failures.iter().cloned().collect::<Vec<_>>().join(", ");
        return Err(ExtractionError::Ungrounded(detail));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    //! Mirrors tests/test_grounding.py: annotation flags fabricated dates,
    //! the strict run loop blocks ungrounded calls unless the value is
    //! written in the sources (Decimal-style numeric equality, per path).
    use super::*;
    use needle_engine::FunctionCall;
    use serde_json::json;

    const DATED_SYSTEM: &str = "date: 2026-07-21 Tue 14:30";

    fn invoice_schema() -> Value {
        json!({"name": "Invoice",
               "parameters": {"type": "object",
                              "properties": {"vendor": {"type": "string"},
                                             "due_date": {"type": "string", "format": "date"},
                                             "total": {"type": "number"}},
                              "required": ["vendor", "due_date"]}})
    }

    fn invoice_response(due_date: &str) -> Response {
        Response {
            r#type: "call".into(),
            success: true,
            error: None,
            error_code: None,
            function_calls: vec![FunctionCall {
                name: "Invoice".into(),
                arguments: json!({"vendor": "Acme", "due_date": due_date}),
            }],
            suppressed_calls: vec![],
            reasoning: String::new(),
            confidence: None,
            prefill_tps: 0.0,
            decode_tps: 0.0,
            results: None,
            validation: None,
        }
    }

    fn ungrounded_names(response: &Response) -> Vec<String> {
        response
            .validation
            .as_ref()
            .and_then(|v| v.get("ungrounded"))
            .and_then(|u| u.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default()
    }

    fn annotate(response: &mut Response, seen: &[u32], system: Option<&str>, text: Option<&str>) {
        annotate_ungrounded(
            response,
            std::slice::from_ref(&invoice_schema()),
            &seen.iter().copied().collect(),
            system,
            text,
        );
    }

    #[test]
    fn flags_call_dates_that_contradict_the_input() {
        // mirrors test_complete_flags_call_dates_that_contradict_the_input
        let mut r = invoice_response("2026-09-05");
        annotate(&mut r, &[2031], Some(DATED_SYSTEM), Some("Send an invoice to Acme due on 5th September 2031"));
        assert_eq!(ungrounded_names(&r), vec!["Invoice.due_date".to_string()]);
    }

    #[test]
    fn leaves_grounded_dates_alone() {
        let mut r = invoice_response("2031-09-05");
        annotate(&mut r, &[2031], Some(DATED_SYSTEM), Some("Send an invoice to Acme due on 5th September 2031"));
        assert!(r.validation.is_none());
    }

    #[test]
    fn input_without_a_year_is_not_checked() {
        let mut r = invoice_response("2026-09-05");
        annotate(&mut r, &[], Some(DATED_SYSTEM), Some("Send Acme an invoice due next Friday"));
        assert!(r.validation.is_none());
    }

    #[test]
    fn system_facts_license_relative_dates() {
        let mut r = invoice_response("2026-07-22");
        annotate(&mut r, &[2019], Some(DATED_SYSTEM), Some("invoice Acme tomorrow for the 2019 reunion"));
        assert!(r.validation.is_none());
    }

    #[test]
    fn years_carry_across_turns_until_reset() {
        // follow-up tool feedback carries no new years; the seen set still licenses 2031
        let mut r = invoice_response("2031-09-05");
        annotate(&mut r, &[2031], Some(DATED_SYSTEM), Some("{\"id\": 42, \"since\": \"2019-03-01\"}"));
        assert!(r.validation.is_none());
        // after reset only the new turn's years license dates
        let mut r = invoice_response("2031-09-05");
        annotate(&mut r, &[2019], Some(DATED_SYSTEM), Some("customer since March 2019, bill them"));
        assert_eq!(ungrounded_names(&r), vec!["Invoice.due_date".to_string()]);
    }

    #[test]
    fn engine_reported_fabrications_are_kept_and_block_execution() {
        // mirrors test_engine_reported_fabrications_are_kept_and_block_execution:
        // the engine-reported flag survives annotation (the date itself is
        // licensed by the query's year) and resolves to a blocking path
        let mut r = invoice_response("2031-09-05");
        r.validation = Some(json!({"ungrounded": ["Invoice.vendor"], "negation": false}));
        annotate(&mut r, &[2031], Some(DATED_SYSTEM), Some("bill someone on 5th September 2031"));
        assert_eq!(ungrounded_names(&r), vec!["Invoice.vendor".to_string()]);
        let grouped = ungrounded_paths(&r);
        let fabricated: BTreeSet<String> = grouped.get("Invoice").cloned().unwrap_or_default();
        let kept = filter_fabricated(fabricated, &r.function_calls[0].arguments, &["bill someone on 5th September 2031", DATED_SYSTEM]);
        assert_eq!(kept, vec!["vendor".to_string()]);
    }

    #[test]
    fn run_refuses_ungrounded_calls() {
        // mirrors test_run_refuses_ungrounded_calls_unless_strict_is_off
        let mut r = invoice_response("2026-09-05");
        annotate(&mut r, &[2031], Some(DATED_SYSTEM), Some("Send an invoice to Acme due on 5th September 2031"));
        let fabricated = ungrounded_paths(&r).get("Invoice").cloned().unwrap_or_default();
        let kept = filter_fabricated(fabricated, &r.function_calls[0].arguments, &["Send an invoice to Acme due on 5th September 2031", DATED_SYSTEM]);
        assert_eq!(kept, vec!["due_date".to_string()]);
        // strict-off runs execute the call as-is
        assert_eq!(r.function_calls[0].arguments["due_date"], "2026-09-05");
    }

    #[test]
    fn run_executes_a_grounded_number_and_refuses_an_ungrounded_sibling() {
        // mirrors test_run_executes_a_grounded_number_and_refuses_an_ungrounded_sibling
        let query = "make it 21 and cool the room";
        let mut fabricated: BTreeSet<String> = BTreeSet::new();
        fabricated.insert("temperature".to_string());
        let grounded = filter_fabricated(fabricated.clone(), &json!({"temperature": 21}), &[query, DATED_SYSTEM]);
        assert!(grounded.is_empty(), "21 is written in the query");
        let kept = filter_fabricated(fabricated, &json!({"temperature": 22}), &[query, DATED_SYSTEM]);
        assert_eq!(kept, vec!["temperature".to_string()]);
    }

    #[test]
    fn run_executes_a_separated_number_grounded_in_the_query() {
        // mirrors test_run_executes_a_separated_number_grounded_in_the_query
        let mut fabricated: BTreeSet<String> = BTreeSet::new();
        fabricated.insert("temperature".to_string());
        let kept = filter_fabricated(fabricated, &json!({"temperature": 1200}), &["set it to 1,200", DATED_SYSTEM]);
        assert!(kept.is_empty());
    }

    #[test]
    fn run_refuses_a_number_absent_from_the_query() {
        let mut fabricated: BTreeSet<String> = BTreeSet::new();
        fabricated.insert("temperature".to_string());
        let kept = filter_fabricated(fabricated, &json!({"temperature": 99}), &["make it 21 and cool the room", DATED_SYSTEM]);
        assert_eq!(kept, vec!["temperature".to_string()]);
    }

    #[test]
    fn run_still_refuses_a_non_numeric_engine_flag() {
        // a flagged string path is never numerically grounded, even beside a grounded number
        let mut fabricated: BTreeSet<String> = BTreeSet::new();
        fabricated.insert("mode".to_string());
        let kept = filter_fabricated(fabricated, &json!({"temperature": 21}), &["make it 21 and cool the room", DATED_SYSTEM]);
        assert_eq!(kept, vec!["mode".to_string()]);
    }

    #[test]
    fn ungrouped_flags_group_under_their_own_name() {
        let mut r = Response {
            r#type: "call".into(),
            success: true,
            error: None,
            error_code: None,
            function_calls: vec![],
            suppressed_calls: vec![],
            reasoning: String::new(),
            confidence: None,
            prefill_tps: 0.0,
            decode_tps: 0.0,
            results: None,
            validation: Some(json!({"ungrounded": ["set_thermostat", "a.b.c"]})),
        };
        let grouped = ungrounded_paths(&r);
        assert_eq!(grouped["set_thermostat"], ["set_thermostat"].into_iter().map(String::from).collect());
        assert_eq!(grouped["a"], ["b.c"].into_iter().map(String::from).collect());
        r.validation = None;
        assert!(ungrounded_paths(&r).is_empty());
    }

    #[test]
    fn decimal_grounding_matches_separators_and_signs() {
        // mirrors the test_extract_clears_* number cases, via grounded_number_paths
        let cases: &[(&str, f64, bool)] = &[
            ("Invoice from Acme Corp, $1,200.00, due 2026-09-01", 1200.0, true),
            ("Invoice from Acme Corp, $1,200, due 2026-09-01", 1200.0, true),
            ("Invoice from Acme Corp, $1200.00, due 2026-09-01", 1200.0, true),
            ("Invoice from Acme Corp, -1,200.00, due 2026-09-01", -1200.0, true),
            ("Invoice from Acme Corp, $0.00, due 2026-09-01", 0.0, true),
            // digit runs inside a longer number do not ground a prefix value
            ("Invoice from Acme Corp, $12,000.00, due 2026-09-01", 1200.0, false),
            ("Invoice from Acme Corp, $11,200.00, due 2026-09-01", 1200.0, false),
            ("Invoice from Acme Corp, $1,2000.00, due 2026-09-01", 1200.0, false),
            // absent from the source
            ("Invoice from Acme Corp, due 2026-09-01", 1200.0, false),
            // date hyphens are not minus signs
            ("Invoice from Acme Corp, $1,200.00, due 2026-09-01", -9.0, false),
            ("Invoice from Acme Corp, $1,200.00-1,400.00, due 2026-09-01", -1400.0, false),
        ];
        for (text, value, want) in cases {
            let grounded = grounded_number_paths(&json!({"total": value}), &[text, ""]);
            assert_eq!(grounded.contains("total"), *want, "text={text} value={value}");
        }
        // integers ground the same way as floats (Decimal equality)
        let grounded = grounded_number_paths(&json!({"total": 1200}), &["Invoice from Acme Corp, $1,200.00, due 2026-09-01"]);
        assert!(grounded.contains("total"));
        // nested numeric paths carry their index
        let grounded = grounded_number_paths(&json!({"items": [{"price": 1100.0}]}), &["Order with one item priced $1,100.00"]);
        assert!(grounded.contains("items[0].price"));
    }

    #[test]
    fn extract_validation_clears_and_rejects() {
        let schema = invoice_schema();
        let response = |ungrounded: Value, negation: bool| Response {
            r#type: "call".into(),
            success: true,
            error: None,
            error_code: None,
            function_calls: vec![],
            suppressed_calls: vec![],
            reasoning: String::new(),
            confidence: None,
            prefill_tps: 0.0,
            decode_tps: 0.0,
            results: None,
            validation: Some(json!({"ungrounded": ungrounded, "negation": negation})),
        };
        let text = "Invoice from Acme Corp, $1,200.00, due 2026-09-01";
        let args = json!({"vendor": "Acme Corp", "total": 1200.0, "due_date": "2026-09-01"});

        // mirrors test_extract_clears_a_thousands_separated_number
        let ok = response(json!(["Invoice.total"]), false);
        assert!(validate_extraction(text, &schema, &args, &ok, None).is_ok());

        // mirrors test_extract_still_rejects_a_fabricated_number
        let bad = response(json!(["Invoice.total"]), false);
        let mut bad_args = args.clone();
        bad_args["total"] = json!(9999.0);
        let err = validate_extraction(text, &schema, &bad_args, &bad, None).unwrap_err();
        assert!(err.to_string().contains("total"), "{err}");

        // mirrors test_extract_still_rejects_a_non_numeric_engine_flag
        let bad = response(json!(["Invoice.vendor"]), false);
        let err = validate_extraction(text, &schema, &args, &bad, None).unwrap_err();
        assert!(err.to_string().contains("vendor"), "{err}");

        // nested separated numbers clear (test_extract_clears_a_nested_separated_number)
        let order_schema = json!({"name": "Order", "parameters": {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "object", "properties": {"price": {"type": "number"}}}}}}});
        let ok = response(json!(["Order.items[0].price"]), false);
        assert!(validate_extraction("Order with one item priced $1,100.00", &order_schema, &json!({"items": [{"price": 1100.0}]}), &ok, None).is_ok());

        // negation reports land in the error detail
        let negated = response(json!([]), true);
        let err = validate_extraction(text, &schema, &args, &negated, None).unwrap_err();
        assert!(err.to_string().contains("negated request"), "{err}");

        // model-fabricated dates raise through the annotation path itself:
        // the model invents 2031 while the input only licenses 2026
        let mut r = invoice_response("2031-09-05");
        annotate(&mut r, &[2026], Some(DATED_SYSTEM), Some("Send an invoice to Acme due on 5th September 2026"));
        let err = validate_extraction(
            "Send an invoice to Acme due on 5th September 2026",
            &schema,
            &r.function_calls[0].arguments,
            &r,
            None,
        )
        .unwrap_err();
        assert!(err.to_string().contains("due_date"), "{err}");
    }
}
