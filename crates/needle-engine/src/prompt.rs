//! Prompt rendering — a port of `render_example` in `needle/model/finetune.py`.
//! One turn is:
//! `<|im_start|>system\n{system}<|im_end|>\n` (optional) +
//! `<|im_start|>user\n<tools>{json}</tools>\n{query}<|im_end|>\n<|im_start|>assistant\n`
//! and the model continues with an optional think block +
//! `<tool_call>{json}</tool_call><|im_end|>`.

use needle_tokenizer::{IM_END, IM_START, TOOLS_END, TOOLS_START};

/// Render the prompt for one turn. `tools_json` is the compact JSON schema list.
pub fn render(tools_json: &str, query: &str, system: Option<&str>) -> String {
    // finetune.py::render_example strips the system string and omits the
    // system turn entirely when nothing remains after trimming
    let prefix = match system.map(str::trim) {
        Some(s) if !s.is_empty() => format!("{IM_START}system\n{s}{IM_END}\n"),
        _ => String::new(),
    };
    format!("{prefix}{IM_START}user\n{TOOLS_START}{tools_json}{TOOLS_END}\n{query}{IM_END}\n{IM_START}assistant\n")
}

/// The training template for multi-turn: assistant answers with an optional
/// think block then the tool-call JSON, closed by `<|im_end|>`.
pub fn render_target(reasoning: Option<&str>, answers_json: &str) -> String {
    let think = match reasoning {
        Some(r) if !r.trim().is_empty() => {
            format!("{}\n{}\n{}\n", needle_tokenizer::THINK_START, r.trim(), needle_tokenizer::THINK_END)
        }
        _ => String::new(),
    };
    format!(
        "{think}{}{answers_json}{}{IM_END}",
        needle_tokenizer::TOOL_CALL_START,
        needle_tokenizer::TOOL_CALL_END
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use needle_tokenizer::{IM_START, THINK_END, THINK_START, TOOL_CALL_END, TOOL_CALL_START, TOOLS_START};

    #[test]
    fn matches_training_template() {
        // mirrors tests/test_render.py::test_render_example_matches_training_template
        let p = render("", "q", None);
        assert!(p.starts_with(&format!("{IM_START}user\n{TOOLS_START}")));
        assert!(p.ends_with(&format!("q<|im_end|>\n{IM_START}assistant\n")));
        let t = render_target(Some("'q' -> f"), r#"[{"name":"f","arguments":{}}]"#);
        assert_eq!(
            t,
            format!("{THINK_START}\n'q' -> f\n{THINK_END}\n{TOOL_CALL_START}[{{\"name\":\"f\",\"arguments\":{{}}}}]{TOOL_CALL_END}<|im_end|>")
        );
        let t2 = render_target(None, "[]");
        assert_eq!(t2, format!("{TOOL_CALL_START}[]{TOOL_CALL_END}<|im_end|>"));
    }

    #[test]
    fn system_prefix() {
        let p = render("[]", "q", Some("date: 2026-07-21 Tue 14:30"));
        assert!(p.starts_with(&format!("{IM_START}system\ndate: 2026-07-21 Tue 14:30<|im_end|>\n")));
    }

    #[test]
    fn whitespace_only_system_is_omitted() {
        // mirrors finetune.py: a stripped-empty system renders no system turn
        let p = render("[]", "q", Some("   \n\t  "));
        assert!(p.starts_with(&format!("{IM_START}user\n")));
        assert!(!p.contains(&format!("{IM_START}system")));
    }

    #[test]
    fn system_prompt_is_trimmed() {
        let p = render("[]", "q", Some("  date: today  "));
        assert!(p.starts_with(&format!("{IM_START}system\ndate: today{IM_END}\n")));
    }
}
