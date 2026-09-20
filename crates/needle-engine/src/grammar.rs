//! Byte-level grammar for Needle tool calls, compiled from JSON Schemas.
//!
//! The training target is `<tool_call>[{"name":"X","arguments":{...}},...]</tool_call>`.
//! A streaming byte acceptor: `step(byte)` advances the parse; `valid_next()`
//! lists the bytes that keep the parse alive so the decoder can mask the
//! vocabulary to tokens whose every byte is acceptable.
//!
//! Guarantees: well-formed JSON in the trained shape — call objects with a
//! `name` (one of the declared tools) and `arguments` matching the schema
//! (required keys present, value types, string enums). Objects with no
//! declared properties (or truthy `additionalProperties`) are free-form: any
//! key is accepted and its value parses as unrestricted JSON. Numeric ranges,
//! string patterns and container sizes are NOT enforced (validate post-hoc).

use std::collections::BTreeSet;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct ToolSpec {
    pub name: String,
    pub props: Vec<(String, VSpec)>,
    pub required: BTreeSet<String>,
    /// free-form arguments object (no declared properties, or
    /// `additionalProperties` set): undeclared keys are accepted
    pub additional: bool,
}

#[derive(Debug, Clone)]
pub enum VSpec {
    Str { en: Option<Arc<Vec<String>>>, max_len: Option<usize> },
    Int,
    Num,
    Bool,
    Arr { items: Box<VSpec>, max_items: Option<usize> },
    Obj { props: Vec<(String, VSpec)>, required: BTreeSet<String>, additional: bool },
    Any,
}

/// An object schema shared by the object value frames.
#[derive(Debug, Clone)]
struct ObjSpec {
    props: Vec<(String, VSpec)>,
    required: BTreeSet<String>,
    /// free-form object (no declared properties, or `additionalProperties`
    /// set): undeclared keys are accepted with unrestricted JSON values
    additional: bool,
}

type ObjRc = Arc<ObjSpec>;

/// What a completed value frame turns into. Argument keys ride inside the
/// continuation (not in shared parse state) so nested objects cannot
/// clobber the key an outer value still needs to record.
#[derive(Debug, Clone)]
enum Cont {
    /// an argument value in the current tool's arguments object; carries the key
    Arg { key: String },
    /// an array element; carries the parent array state and the array's own cont
    ArrItem { items: Box<VSpec>, max: Option<usize>, count: usize, cont: Box<Cont> },
    /// an object value; carries parent object state + key to record
    ObjValue { spec: ObjRc, seen: BTreeSet<String>, key: String, cont: Box<Cont> },
}

/// Escape state inside a string value: plain, just after `\`, or mid-`\uXXXX`
/// (`n` hex digits read so far).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StrEsc {
    None,
    Esc,
    Hex { val: u32, n: u8 },
}

#[derive(Debug, Clone)]
enum Frame {
    /// answers array opens with '['
    ArrOpen,
    /// inside the array: '{' starts a call, ']' ends (empty = refusal)
    Arr,
    /// reading the literal key `"name"`
    CallKeyName { buf: String },
    CallColon,
    /// reading the tool-name string (prefix-matched against the tool set)
    CallName { buf: String, in_str: bool },
    /// expecting ',' after the name value
    CallNameComma,
    /// reading the literal key `"arguments"`
    CallArgsKey { buf: String },
    CallArgsColon,
    /// expecting '{' to open the arguments object
    ArgsOpen,
    /// after '{': the first property key or '}' (empty arguments object)
    ArgsFirst,
    /// reading a property-name string (prefix-matched against the tool's props)
    ArgsKey { buf: String, in_str: bool },
    /// expecting ':' after a key; carries the key + its value spec
    ArgsColon { key: String, spec: VSpec },
    /// after an argument value: ',' (next key) or '}' (close arguments)
    ArgsNext,
    /// after the arguments object: '}' closes the call object
    CallObjEnd,
    /// after a call: ',' (next call) or ']' (close array)
    CallEnd,
    Done,
    // ---- value frames (all carry their continuation) ----
    VStr { open: bool, esc: StrEsc, en: Option<Arc<Vec<String>>>, sbuf: Vec<u8>, cont: Cont },
    VInt { minus: bool, digits: usize, zero: bool, cont: Cont },
    VNum(Num, Cont),
    VBool { branch: Option<bool>, pos: usize, cont: Cont },
    VAny { kind: AnyKind, cont: Cont },
    VArrFirst { items: Box<VSpec>, max: Option<usize>, cont: Cont },
    VArrNext { items: Box<VSpec>, max: Option<usize>, count: usize, cont: Cont },
    VObjFirst { spec: ObjRc, cont: Cont },
    VObjKey { spec: ObjRc, seen: BTreeSet<String>, buf: String, in_str: bool, cont: Cont },
    /// expecting ':' after a key; carries the key + its value spec
    VObjColon { spec: ObjRc, seen: BTreeSet<String>, key: String, vspec: VSpec, cont: Cont },
    VObjNext { spec: ObjRc, seen: BTreeSet<String>, cont: Cont },
}

#[derive(Debug, Clone, Default)]
struct Num {
    minus: bool,
    int_digits: usize,
    /// the int part is a single leading '0' (no further int digits allowed)
    lead_zero: bool,
    dot: bool,
    frac_digits: usize,
    exp: bool,
    exp_sign: bool,
    exp_digits: usize,
}

#[derive(Debug, Clone)]
enum AnyKind {
    Start,
    Str { esc: bool },
    Num(Num),
    Null { pos: u8 },
}

impl VSpec {
    pub fn from_schema(node: &serde_json::Value) -> VSpec {
        let obj = node.as_object();
        if let Some(en) = obj.and_then(|o| o.get("enum")).and_then(|e| e.as_array()) {
            let vals: Vec<String> = en.iter().filter_map(|v| v.as_str().map(String::from)).collect();
            return VSpec::Str { en: Some(Arc::new(vals)), max_len: None };
        }
        if let Some(c) = obj.and_then(|o| o.get("const")).and_then(|c| c.as_str()) {
            return VSpec::Str { en: Some(Arc::new(vec![c.to_string()])), max_len: None };
        }
        let ty = obj
            .and_then(|o| o.get("type"))
            .and_then(|t| t.as_str())
            .unwrap_or("string");
        match ty {
            "integer" => VSpec::Int,
            "number" => VSpec::Num,
            "boolean" => VSpec::Bool,
            "array" => {
                let items = obj
                    .and_then(|o| o.get("items"))
                    .map(|i| Box::new(VSpec::from_schema(i)))
                    .unwrap_or_else(|| Box::new(VSpec::Any));
                let max_items = obj
                    .and_then(|o| o.get("maxItems"))
                    .and_then(|v| v.as_u64().map(|n| n as usize));
                VSpec::Arr { items, max_items }
            }
            "object" => {
                let mut props = Vec::new();
                let mut required = BTreeSet::new();
                if let Some(ps) = obj.and_then(|o| o.get("properties")).and_then(|p| p.as_object()) {
                    for (k, v) in ps {
                        props.push((k.clone(), VSpec::from_schema(v)));
                    }
                }
                if let Some(req) = obj.and_then(|o| o.get("required")).and_then(|r| r.as_array()) {
                    for r in req {
                        if let Some(s) = r.as_str() {
                            required.insert(s.to_string());
                        }
                    }
                }
                // objects with no declared properties (or truthy
                // additionalProperties) accept arbitrary keys
                let additional = props.is_empty()
                    || obj
                        .and_then(|o| o.get("additionalProperties"))
                        .is_some_and(|v| v.as_bool() != Some(false));
                VSpec::Obj { props, required, additional }
            }
            _ => VSpec::Str {
                en: None,
                max_len: obj
                    .and_then(|o| o.get("maxLength"))
                    .and_then(|v| v.as_u64().map(|n| n as usize)),
            },
        }
    }
}

/// The conversation's tool set + parse state.
#[derive(Debug, Clone)]
pub struct Grammar {
    tools: Vec<ToolSpec>,
    frames: Vec<Frame>,
    cur_tool: usize,
    /// keys recorded so far in the current tool's arguments object
    args_seen: BTreeSet<String>,
}

fn value_frame(spec: &VSpec, cont: Cont) -> Frame {
    match spec {
        VSpec::Str { en, .. } => Frame::VStr {
            open: false,
            esc: StrEsc::None,
            en: en.clone(),
            sbuf: Vec::new(),
            cont,
        },
        VSpec::Int => Frame::VInt { minus: false, digits: 0, zero: false, cont },
        VSpec::Num => Frame::VNum(Num::default(), cont),
        VSpec::Bool => Frame::VBool { branch: None, pos: 0, cont },
        VSpec::Arr { items, max_items } => Frame::VArrFirst { items: items.clone(), max: *max_items, cont },
        VSpec::Obj { props, required, additional } => Frame::VObjFirst {
            spec: Arc::new(ObjSpec {
                props: props.clone(),
                required: required.clone(),
                additional: *additional,
            }),
            cont,
        },
        VSpec::Any => Frame::VAny { kind: AnyKind::Start, cont },
    }
}

fn step_num(nm: &mut Num, byte: u8) -> bool {
    match byte {
        b'0'..=b'9' => {
            if nm.exp {
                nm.exp_digits += 1;
            } else if nm.dot {
                nm.frac_digits += 1;
            } else {
                // no leading zeros: a '0' int part may not take more digits
                if nm.lead_zero {
                    return false;
                }
                if nm.int_digits == 0 && byte == b'0' {
                    nm.lead_zero = true;
                }
                nm.int_digits += 1;
            }
            true
        }
        b'-' => {
            if nm.int_digits == 0 && !nm.minus && !nm.dot && !nm.exp {
                nm.minus = true;
                true
            } else if nm.exp && nm.exp_digits == 0 && !nm.exp_sign {
                nm.exp_sign = true;
                true
            } else {
                false
            }
        }
        b'+' => nm.exp && nm.exp_digits == 0 && !nm.exp_sign && replace(&mut nm.exp_sign, true),
        b'.' => !nm.dot && !nm.exp && nm.int_digits > 0 && replace(&mut nm.dot, true),
        b'e' | b'E' => {
            // an opened fraction must have digits before the exponent starts
            !nm.exp && nm.int_digits > 0 && (!nm.dot || nm.frac_digits > 0) && replace(&mut nm.exp, true)
        }
        _ => false,
    }
}

/// A JSON number may end only when every part that was opened has digits
/// (so `1e`, `1.` and bare `-` can never complete).
fn num_complete(nm: &Num) -> bool {
    nm.int_digits > 0 && (!nm.dot || nm.frac_digits > 0) && (!nm.exp || nm.exp_digits > 0)
}

fn hex_val(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

/// Enum-prefix check after appending a (plain or escaped) byte to the
/// decoded string buffer.
fn str_ok(en: &Option<Arc<Vec<String>>>, sbuf: &[u8]) -> bool {
    match en {
        Some(opts) => opts.iter().any(|o| o.as_bytes().starts_with(sbuf)),
        None => true,
    }
}

fn replace(slot: &mut bool, val: bool) -> bool {
    *slot = val;
    true
}

impl Grammar {
    pub fn compile(tools: &[serde_json::Value]) -> Grammar {
        let tools = tools
            .iter()
            .map(|t| {
                let name = t.get("name").and_then(|n| n.as_str()).unwrap_or("").to_string();
                let params = t
                    .get("parameters")
                    .cloned()
                    .unwrap_or_else(|| serde_json::json!({"type":"object"}));
                let (props, required, additional) = match VSpec::from_schema(&params) {
                    VSpec::Obj { props, required, additional } => (props, required, additional),
                    _ => (Vec::new(), BTreeSet::new(), true),
                };
                ToolSpec { name, props, required, additional }
            })
            .collect();
        Grammar {
            tools,
            frames: vec![Frame::ArrOpen],
            cur_tool: usize::MAX,
            args_seen: BTreeSet::new(),
        }
    }

    pub fn is_done(&self) -> bool {
        matches!(self.frames.last(), Some(Frame::Done))
    }

    /// The frame a completed value resolves to.
    fn resolve(&mut self, cont: Cont) -> Frame {
        match cont {
            Cont::Arg { key } => {
                self.args_seen.insert(key);
                Frame::ArgsNext
            }
            Cont::ArrItem { items, max, count, cont } => Frame::VArrNext { items, max, count, cont: *cont },
            Cont::ObjValue { spec, mut seen, key, cont } => {
                seen.insert(key);
                Frame::VObjNext { spec, seen, cont: *cont }
            }
        }
    }

    /// Replace the top frame with `next` (value completed, terminator consumed).
    fn complete(&mut self, cont: Cont) -> Frame {
        self.resolve(cont)
    }

    pub fn step(&mut self, byte: u8) -> bool {
        for _ in 0..128 {
            let depth = self.frames.len();
            let result: bool = match self.frames.last() {
                None => return false,
                Some(frame) => match frame {
                    Frame::ArrOpen => {
                        if byte == b'[' {
                            self.frames[depth - 1] = Frame::Arr;
                        }
                        byte == b'['
                    }
                    Frame::Arr => match byte {
                        b'{' => {
                            self.frames[depth - 1] = Frame::CallKeyName { buf: String::new() };
                            true
                        }
                        b']' => {
                            self.frames[depth - 1] = Frame::Done;
                            true
                        }
                        _ => false,
                    },
                    Frame::CallKeyName { buf } => {
                        let mut nb = buf.clone();
                        nb.push(byte as char);
                        if !"\"name\"".starts_with(nb.as_str()) {
                            false
                        } else {
                            if nb == "\"name\"" {
                                self.frames[depth - 1] = Frame::CallColon;
                            } else {
                                self.frames[depth - 1] = Frame::CallKeyName { buf: nb };
                            }
                            true
                        }
                    }
                    Frame::CallColon => {
                        if byte == b':' {
                            self.frames[depth - 1] = Frame::CallName { buf: String::new(), in_str: false };
                        }
                        byte == b':'
                    }
                    Frame::CallName { buf, in_str } => {
                        if !in_str {
                            if byte == b'"' {
                                self.frames[depth - 1] =
                                    Frame::CallName { buf: String::new(), in_str: true };
                                true
                            } else {
                                false
                            }
                        } else if byte == b'"' {
                            match self.tools.iter().position(|t| t.name == *buf) {
                                Some(ti) => {
                                    self.cur_tool = ti;
                                    self.frames[depth - 1] = Frame::CallNameComma;
                                    true
                                }
                                None => false,
                            }
                        } else if byte < 0x20 {
                            false
                        } else {
                            let mut nb = buf.clone();
                            nb.push(byte as char);
                            if self.tools.iter().any(|t| t.name.starts_with(nb.as_str())) {
                                self.frames[depth - 1] = Frame::CallName { buf: nb, in_str: true };
                                true
                            } else {
                                false
                            }
                        }
                    }
                    Frame::CallNameComma => {
                        if byte == b',' {
                            self.frames[depth - 1] = Frame::CallArgsKey { buf: String::new() };
                        }
                        byte == b','
                    }
                    Frame::CallArgsKey { buf } => {
                        let mut nb = buf.clone();
                        nb.push(byte as char);
                        if !"\"arguments\"".starts_with(nb.as_str()) {
                            false
                        } else {
                            if nb == "\"arguments\"" {
                                self.frames[depth - 1] = Frame::CallArgsColon;
                            } else {
                                self.frames[depth - 1] = Frame::CallArgsKey { buf: nb };
                            }
                            true
                        }
                    }
                    Frame::CallArgsColon => {
                        if byte == b':' {
                            self.frames[depth - 1] = Frame::ArgsOpen;
                        }
                        byte == b':'
                    }
                    Frame::ArgsOpen => {
                        if byte == b'{' {
                            self.args_seen.clear();
                            self.frames[depth - 1] = Frame::ArgsFirst;
                        }
                        byte == b'{'
                    }
                    Frame::ArgsFirst => match byte {
                        b'"' => {
                            self.frames[depth - 1] =
                                Frame::ArgsKey { buf: String::new(), in_str: true };
                            true
                        }
                        b'}' => {
                            // empty arguments object: legal only when no
                            // required keys remain (the trained no-arg shape)
                            let ok = self
                                .tools
                                .get(self.cur_tool)
                                .map(|t| t.required.iter().all(|r| self.args_seen.contains(r)))
                                .unwrap_or(true);
                            if ok {
                                self.frames[depth - 1] = Frame::CallObjEnd;
                            }
                            ok
                        }
                        _ => false,
                    },
                    Frame::ArgsKey { buf, in_str } => {
                        let (buf, in_str) = (buf.clone(), *in_str);
                        if !in_str {
                            if byte == b'"' {
                                self.frames[depth - 1] =
                                    Frame::ArgsKey { buf: String::new(), in_str: true };
                                true
                            } else {
                                false
                            }
                        } else if byte == b'"' {
                            if self.args_seen.contains(&buf) {
                                return false;
                            }
                            let tool = &self.tools[self.cur_tool];
                            let spec =
                                tool.props.iter().find(|(p, _)| p == &buf).map(|(_, s)| s.clone());
                            let spec = match spec {
                                Some(s) => s,
                                None if tool.additional => VSpec::Any,
                                None => return false,
                            };
                            self.frames[depth - 1] = Frame::ArgsColon { key: buf, spec };
                            true
                        } else if byte < 0x20 {
                            false
                        } else {
                            let mut nb = buf.clone();
                            nb.push(byte as char);
                            let tool = &self.tools[self.cur_tool];
                            if tool.props.iter().any(|(p, _)| p.starts_with(nb.as_str()))
                                || tool.additional
                            {
                                self.frames[depth - 1] = Frame::ArgsKey { buf: nb, in_str: true };
                                true
                            } else {
                                false
                            }
                        }
                    }
                    Frame::ArgsColon { key, spec } => {
                        if byte == b':' {
                            // the key rides in the continuation so nested
                            // objects cannot overwrite it before it records
                            self.frames[depth - 1] =
                                value_frame(spec, Cont::Arg { key: key.clone() });
                        }
                        byte == b':'
                    }
                    Frame::ArgsNext => match byte {
                        b',' => {
                            self.frames[depth - 1] =
                                Frame::ArgsKey { buf: String::new(), in_str: false };
                            true
                        }
                        b'}' => {
                            let ok = self
                                .tools
                                .get(self.cur_tool)
                                .map(|t| t.required.iter().all(|r| self.args_seen.contains(r)))
                                .unwrap_or(true);
                            if ok {
                                self.frames[depth - 1] = Frame::CallObjEnd;
                            }
                            ok
                        }
                        _ => false,
                    },
                    Frame::CallObjEnd => {
                        if byte == b'}' {
                            self.frames[depth - 1] = Frame::CallEnd;
                        }
                        byte == b'}'
                    }
                    Frame::CallEnd => match byte {
                        b',' => {
                            self.frames[depth - 1] = Frame::Arr;
                            true
                        }
                        b']' => {
                            self.frames[depth - 1] = Frame::Done;
                            true
                        }
                        _ => false,
                    },
                    Frame::Done => false,

                    // ---------- values ----------
                    Frame::VStr { open, esc, en, sbuf, cont } => {
                        let (open, esc, en, mut sbuf, cont) =
                            (*open, *esc, en.clone(), sbuf.clone(), cont.clone());
                        if !open {
                            if byte == b'"' {
                                self.frames[depth - 1] = Frame::VStr {
                                    open: true,
                                    esc: StrEsc::None,
                                    en,
                                    sbuf,
                                    cont,
                                };
                                return true;
                            }
                            return false;
                        }
                        if let StrEsc::Esc = esc {
                            // the escaped byte joins the decoded buffer, so
                            // enums and strings compare in decoded space
                            let decoded_byte: u8 = match byte {
                                b'"' => b'"',
                                b'\\' => b'\\',
                                b'/' => b'/',
                                b'b' => 0x08,
                                b'f' => 0x0c,
                                b'n' => b'\n',
                                b'r' => b'\r',
                                b't' => b'\t',
                                b'u' => {
                                    self.frames[depth - 1] = Frame::VStr {
                                        open: true,
                                        esc: StrEsc::Hex { val: 0, n: 0 },
                                        en,
                                        sbuf,
                                        cont,
                                    };
                                    return true;
                                }
                                _ => return false,
                            };
                            sbuf.push(decoded_byte);
                            if str_ok(&en, &sbuf) {
                                self.frames[depth - 1] = Frame::VStr {
                                    open: true,
                                    esc: StrEsc::None,
                                    en,
                                    sbuf,
                                    cont,
                                };
                                true
                            } else {
                                false
                            }
                        } else if let StrEsc::Hex { val, n } = esc {
                            // \u must be followed by exactly 4 hex digits
                            let Some(d) = hex_val(byte) else {
                                return false;
                            };
                            let v = val * 16 + d as u32;
                            let nn = n + 1;
                            if nn < 4 {
                                self.frames[depth - 1] = Frame::VStr {
                                    open: true,
                                    esc: StrEsc::Hex { val: v, n: nn },
                                    en,
                                    sbuf,
                                    cont,
                                };
                                return true;
                            }
                            // complete the escape: append the decoded
                            // codepoint as UTF-8 (lone surrogates are left to
                            // serde's post-hoc validation)
                            if let Some(c) = char::from_u32(v) {
                                let mut tmp = [0u8; 4];
                                sbuf.extend_from_slice(c.encode_utf8(&mut tmp).as_bytes());
                                if !str_ok(&en, &sbuf) {
                                    return false;
                                }
                            }
                            self.frames[depth - 1] = Frame::VStr {
                                open: true,
                                esc: StrEsc::None,
                                en,
                                sbuf,
                                cont,
                            };
                            true
                        } else {
                            let ok = match byte {
                                b'\\' => {
                                    self.frames[depth - 1] = Frame::VStr {
                                        open: true,
                                        esc: StrEsc::Esc,
                                        en,
                                        sbuf,
                                        cont,
                                    };
                                    true
                                }
                                b'"' => {
                                    let en_ok = match &en {
                                        Some(opts) => {
                                            opts.iter().any(|o| o.as_bytes() == sbuf.as_slice())
                                        }
                                        None => true,
                                    };
                                    if en_ok {
                                        self.frames[depth - 1] = self.complete(cont);
                                    }
                                    en_ok
                                }
                                0x00..=0x1f => false,
                                _ => {
                                    sbuf.push(byte);
                                    if str_ok(&en, &sbuf) {
                                        self.frames[depth - 1] = Frame::VStr {
                                            open: true,
                                            esc: StrEsc::None,
                                            en,
                                            sbuf,
                                            cont,
                                        };
                                        true
                                    } else {
                                        false
                                    }
                                }
                            };
                            return ok;
                        }
                    }
                    Frame::VInt { minus, digits, zero, cont } => {
                        let (minus, mut digits, mut zero, cont) =
                            (*minus, *digits, *zero, cont.clone());
                        match byte {
                            b'-' if !minus && digits == 0 => {
                                self.frames[depth - 1] =
                                    Frame::VInt { minus: true, digits, zero, cont };
                                true
                            }
                            b'0'..=b'9' => {
                                // JSON integers take no leading zeros
                                if zero {
                                    false
                                } else {
                                    if digits == 0 && byte == b'0' {
                                        zero = true;
                                    }
                                    digits += 1;
                                    self.frames[depth - 1] =
                                        Frame::VInt { minus, digits, zero, cont };
                                    true
                                }
                            }
                            _ if digits > 0 => {
                                self.frames[depth - 1] = self.complete(cont);
                                continue; // redispatch the terminator
                            }
                            _ => false,
                        }
                    }
                    Frame::VNum(nm, cont) => {
                        let (mut nm, cont) = (nm.clone(), cont.clone());
                        if step_num(&mut nm, byte) {
                            self.frames[depth - 1] = Frame::VNum(nm, cont);
                            true
                        } else if num_complete(&nm) {
                            self.frames[depth - 1] = self.complete(cont);
                            continue; // redispatch
                        } else {
                            false
                        }
                    }
                    Frame::VBool { branch, pos, cont } => {
                        let (branch, mut pos, cont) = (*branch, *pos, cont.clone());
                        if branch.is_none() && pos == 0 {
                            if byte == b't' {
                                self.frames[depth - 1] =
                                    Frame::VBool { branch: Some(true), pos: 1, cont };
                                return true;
                            }
                            if byte == b'f' {
                                self.frames[depth - 1] =
                                    Frame::VBool { branch: Some(false), pos: 1, cont };
                                return true;
                            }
                            return false;
                        }
                        let word: &[u8] = if branch == Some(true) { b"true" } else { b"false" };
                        if pos < word.len() && byte == word[pos] {
                            pos += 1;
                            if pos == word.len() {
                                self.frames[depth - 1] = self.complete(cont);
                            } else {
                                self.frames[depth - 1] = Frame::VBool { branch, pos, cont };
                            }
                            true
                        } else {
                            false
                        }
                    }
                    Frame::VAny { kind, cont } => {
                        let (kind, cont) = (kind.clone(), cont.clone());
                        match kind {
                            AnyKind::Start => {
                                let consume = matches!(byte, b'"' | b'[' | b'{');
                                let nf = match byte {
                                    b'"' => Some(Frame::VAny {
                                        kind: AnyKind::Str { esc: false },
                                        cont: cont.clone(),
                                    }),
                                    b'-' | b'0'..=b'9' => Some(Frame::VAny {
                                        kind: AnyKind::Num(Num::default()),
                                        cont: cont.clone(),
                                    }),
                                    b't' | b'f' => Some(value_frame(&VSpec::Bool, cont.clone())),
                                    b'n' => Some(Frame::VAny {
                                        kind: AnyKind::Null { pos: 0 },
                                        cont: cont.clone(),
                                    }),
                                    b'[' => Some(Frame::VArrNext {
                                        items: Box::new(VSpec::Any),
                                        max: None,
                                        count: 0,
                                        cont: cont.clone(),
                                    }),
                                    b'{' => Some(Frame::VObjKey {
                                        spec: Arc::new(ObjSpec {
                                            props: Vec::new(),
                                            required: BTreeSet::new(),
                                            additional: true,
                                        }),
                                        seen: BTreeSet::new(),
                                        buf: String::new(),
                                        in_str: false,
                                        cont: cont.clone(),
                                    }),
                                    _ => None,
                                };
                                match nf {
                                    Some(f) => {
                                        self.frames[depth - 1] = f;
                                        if consume {
                                            return true; // opening token consumed
                                        }
                                        continue; // redispatch into the new frame
                                    }
                                    None => return false,
                                }
                            }
                            AnyKind::Str { esc } => {
                                match (esc, byte) {
                                    (true, _) => {
                                        self.frames[depth - 1] = Frame::VAny {
                                            kind: AnyKind::Str { esc: false },
                                            cont,
                                        };
                                        true
                                    }
                                    (false, b'\\') => {
                                        self.frames[depth - 1] = Frame::VAny {
                                            kind: AnyKind::Str { esc: true },
                                            cont,
                                        };
                                        true
                                    }
                                    (false, b'"') => {
                                        self.frames[depth - 1] = self.complete(cont);
                                        true
                                    }
                                    (false, 0x00..=0x1f) => false,
                                    (false, _) => true,
                                }
                            }
                            AnyKind::Null { pos } => {
                                let word = b"null";
                                if (pos as usize) < word.len() && byte == word[pos as usize] {
                                    let np = pos + 1;
                                    if np as usize == word.len() {
                                        self.frames[depth - 1] = self.complete(cont);
                                    } else {
                                        self.frames[depth - 1] =
                                            Frame::VAny { kind: AnyKind::Null { pos: np }, cont };
                                    }
                                    true
                                } else {
                                    false
                                }
                            }
                            AnyKind::Num(mut nm) => {
                                if step_num(&mut nm, byte) {
                                    self.frames[depth - 1] =
                                        Frame::VAny { kind: AnyKind::Num(nm), cont };
                                    true
                                } else if num_complete(&nm) {
                                    self.frames[depth - 1] = self.complete(cont);
                                    continue; // redispatch
                                } else {
                                    false
                                }
                            }
                        }
                    }
                    Frame::VArrFirst { items, max, cont } => {
                        let (items, max, cont) = (items.clone(), *max, cont.clone());
                        if byte == b'[' {
                            self.frames[depth - 1] =
                                Frame::VArrNext { items, max, count: 0, cont };
                            true
                        } else {
                            false
                        }
                    }
                    Frame::VArrNext { items, max, count, cont } => {
                        let (items, max, count, cont) =
                            (items.clone(), *max, *count, cont.clone());
                        match byte {
                            b']' => {
                                self.frames[depth - 1] = self.complete(cont);
                                true
                            }
                            b',' if count > 0 && max.is_none_or(|m| count < m) => {
                                let new_count = count + 1;
                                let item_cont = Cont::ArrItem {
                                    items: items.clone(),
                                    max,
                                    count: new_count,
                                    cont: Box::new(cont.clone()),
                                };
                                self.frames[depth - 1] = Frame::VArrNext {
                                    items: items.clone(),
                                    max,
                                    count,
                                    cont,
                                };
                                self.frames.push(value_frame(&items, item_cont));
                                true
                            }
                            _ if count == 0 || max.is_none_or(|m| count < m) => {
                                // first byte of the next item: push and redispatch
                                let new_count = count + 1;
                                let item_cont = Cont::ArrItem {
                                    items: items.clone(),
                                    max,
                                    count: new_count,
                                    cont: Box::new(cont.clone()),
                                };
                                self.frames[depth - 1] = Frame::VArrNext {
                                    items: items.clone(),
                                    max,
                                    count,
                                    cont,
                                };
                                self.frames.push(value_frame(&items, item_cont));
                                continue;
                            }
                            _ => false,
                        }
                    }
                    Frame::VObjFirst { spec, cont } => {
                        let (spec, cont) = (spec.clone(), cont.clone());
                        if byte == b'{' {
                            self.frames[depth - 1] = Frame::VObjKey {
                                spec,
                                seen: BTreeSet::new(),
                                buf: String::new(),
                                in_str: false,
                                cont,
                            };
                            true
                        } else {
                            false
                        }
                    }
                    Frame::VObjKey { spec, seen, buf, in_str, cont } => {
                        let (spec, seen, buf, in_str, cont) =
                            (spec.clone(), seen.clone(), buf.clone(), *in_str, cont.clone());
                        if !in_str {
                            if byte == b'"' {
                                self.frames[depth - 1] = Frame::VObjKey {
                                    spec,
                                    seen,
                                    buf: String::new(),
                                    in_str: true,
                                    cont,
                                };
                                true
                            } else if byte == b'}' && seen.is_empty() && spec.required.is_empty() {
                                // empty nested object: only legal as the first
                                // byte, so `"a":1,` can never take this path
                                self.frames[depth - 1] = self.complete(cont);
                                true
                            } else {
                                false
                            }
                        } else if byte == b'"' {
                            if seen.contains(&buf) {
                                return false;
                            }
                            let vspec =
                                spec.props.iter().find(|(p, _)| p == &buf).map(|(_, s)| s.clone());
                            let vspec = match vspec {
                                Some(s) => s,
                                None if spec.additional => VSpec::Any,
                                None => return false,
                            };
                            self.frames[depth - 1] =
                                Frame::VObjColon { spec, seen, key: buf, vspec, cont };
                            true
                        } else if byte < 0x20 {
                            false
                        } else {
                            let mut nb = buf.clone();
                            nb.push(byte as char);
                            if spec.props.iter().any(|(p, _)| p.starts_with(nb.as_str()))
                                || spec.additional
                            {
                                self.frames[depth - 1] =
                                    Frame::VObjKey { spec, seen, buf: nb, in_str: true, cont };
                                true
                            } else {
                                false
                            }
                        }
                    }
                    Frame::VObjColon { spec, seen, key, vspec, cont } => {
                        if byte == b':' {
                            self.frames[depth - 1] = value_frame(
                                vspec,
                                Cont::ObjValue {
                                    spec: spec.clone(),
                                    seen: seen.clone(),
                                    key: key.clone(),
                                    cont: Box::new(cont.clone()),
                                },
                            );
                        }
                        byte == b':'
                    }
                    Frame::VObjNext { spec, seen, cont } => {
                        let (spec, seen, cont) = (spec.clone(), seen.clone(), cont.clone());
                        match byte {
                            b',' => {
                                self.frames[depth - 1] = Frame::VObjKey {
                                    spec,
                                    seen,
                                    buf: String::new(),
                                    in_str: false,
                                    cont,
                                };
                                true
                            }
                            b'}' => {
                                let ok = spec.required.iter().all(|r| seen.contains(r));
                                if ok {
                                    self.frames[depth - 1] = self.complete(cont);
                                }
                                ok
                            }
                            _ => false,
                        }
                    }
                },
            };
            return result;
        }
        false
    }

}

/// Bytes acceptable as the next step (for logit masking).
pub fn valid_next(g: &Grammar) -> [bool; 256] {
    let mut out = [false; 256];
    match g.frames.last() {
        Some(Frame::ArrOpen) => {
            out[b'[' as usize] = true;
        }
        Some(Frame::Arr) => {
            out[b'{' as usize] = true;
            out[b']' as usize] = true;
        }
        Some(Frame::CallKeyName { buf }) => literal_next(&mut out, buf, "\"name\""),
        Some(Frame::CallColon) => out[b':' as usize] = true,
        Some(Frame::CallName { buf, in_str }) => {
            if !in_str {
                out[b'"' as usize] = true;
            } else {
                for t in &g.tools {
                    if let Some(c) = t.name.as_bytes().get(buf.len()) {
                        if t.name.starts_with(buf.as_str()) {
                            out[*c as usize] = true;
                        }
                    }
                }
                if g.tools.iter().any(|t| t.name == *buf) {
                    out[b'"' as usize] = true;
                }
            }
        }
        Some(Frame::CallNameComma) => {
            out[b',' as usize] = true;
        }
        Some(Frame::CallArgsKey { buf }) => literal_next(&mut out, buf, "\"arguments\""),
        Some(Frame::CallArgsColon) => out[b':' as usize] = true,
        Some(Frame::ArgsOpen) => out[b'{' as usize] = true,
        Some(Frame::ArgsFirst) => {
            out[b'"' as usize] = true;
            let ok = g
                .tools
                .get(g.cur_tool)
                .map(|t| t.required.iter().all(|r| g.args_seen.contains(r)))
                .unwrap_or(true);
            if ok {
                out[b'}' as usize] = true;
            }
        }
        Some(Frame::ArgsKey { buf, in_str }) => {
            if !in_str {
                out[b'"' as usize] = true;
            } else if let Some(t) = g.tools.get(g.cur_tool) {
                for (p, _) in &t.props {
                    if g.args_seen.contains(p) {
                        continue;
                    }
                    if let Some(c) = p.as_bytes().get(buf.len()) {
                        if p.starts_with(buf.as_str()) {
                            out[*c as usize] = true;
                        }
                    }
                }
                let fresh = !g.args_seen.contains(buf);
                if t.props.iter().any(|(p, _)| p == buf) && fresh {
                    out[b'"' as usize] = true;
                }
                if t.additional && fresh {
                    for b in 0x20u8..=0xff {
                        out[b as usize] = true;
                    }
                    out[b'"' as usize] = true;
                }
            }
        }
        Some(Frame::ArgsColon { .. }) => out[b':' as usize] = true,
        Some(Frame::ArgsNext) => {
            out[b',' as usize] = true;
            let ok = g
                .tools
                .get(g.cur_tool)
                .map(|t| t.required.iter().all(|r| g.args_seen.contains(r)))
                .unwrap_or(true);
            if ok {
                out[b'}' as usize] = true;
            }
        }
        Some(Frame::CallObjEnd) => {
            out[b'}' as usize] = true;
        }
        Some(Frame::CallEnd) => {
            out[b',' as usize] = true;
            out[b']' as usize] = true;
        }
        Some(Frame::Done) => {}
        Some(Frame::VStr { open, esc, .. }) => {
            if !open {
                out[b'"' as usize] = true;
            } else {
                match esc {
                    StrEsc::None => {
                        for b in 0x20u8..=0xff {
                            out[b as usize] = true;
                        }
                        out[b'"' as usize] = true;
                        out[b'\\' as usize] = true;
                    }
                    StrEsc::Esc => {
                        for b in *b"\"\\/bfnrtu" {
                            out[b as usize] = true;
                        }
                    }
                    StrEsc::Hex { .. } => {
                        for b in b'0'..=b'9' {
                            out[b as usize] = true;
                        }
                        for b in b'a'..=b'f' {
                            out[b as usize] = true;
                        }
                        for b in b'A'..=b'F' {
                            out[b as usize] = true;
                        }
                    }
                }
            }
        }
        Some(Frame::VInt { minus, digits, zero, .. }) => {
            if !zero {
                for b in b'0'..=b'9' {
                    out[b as usize] = true;
                }
            }
            if !minus && *digits == 0 {
                out[b'-' as usize] = true;
            }
            if *digits > 0 {
                out[b',' as usize] = true;
                out[b'}' as usize] = true;
                out[b']' as usize] = true;
            }
        }
        Some(Frame::VNum(nm, _)) => {
            if !(nm.lead_zero && !nm.dot && !nm.exp) {
                for b in b'0'..=b'9' {
                    out[b as usize] = true;
                }
            }
            if !nm.minus && nm.int_digits == 0 {
                out[b'-' as usize] = true;
            }
            if nm.int_digits > 0 && !nm.dot && !nm.exp {
                out[b'.' as usize] = true;
                out[b'e' as usize] = true;
                out[b'E' as usize] = true;
            }
            if nm.exp && nm.exp_digits == 0 && !nm.exp_sign {
                out[b'+' as usize] = true;
                out[b'-' as usize] = true;
            }
            if num_complete(nm) {
                out[b',' as usize] = true;
                out[b'}' as usize] = true;
                out[b']' as usize] = true;
            }
        }
        Some(Frame::VBool { branch, pos, .. }) => {
            let word: &[u8] = match branch {
                Some(true) => b"true",
                Some(false) => b"false",
                None => {
                    out[b't' as usize] = true;
                    out[b'f' as usize] = true;
                    return out;
                }
            };
            if let Some(c) = word.get(*pos) {
                out[*c as usize] = true;
            }
        }
        Some(Frame::VAny { kind, .. }) => match kind {
            AnyKind::Start => {
                out[b'"' as usize] = true;
                out[b'-' as usize] = true;
                for b in b'0'..=b'9' {
                    out[b as usize] = true;
                }
                out[b't' as usize] = true;
                out[b'f' as usize] = true;
                out[b'n' as usize] = true;
                out[b'[' as usize] = true;
                out[b'{' as usize] = true;
            }
            AnyKind::Str { esc } => {
                if *esc {
                    for b in *b"\"\\/bfnrtu" {
                        out[b as usize] = true;
                    }
                } else {
                    for b in 0x20u8..=0xff {
                        out[b as usize] = true;
                    }
                    out[b'"' as usize] = true;
                    out[b'\\' as usize] = true;
                }
            }
            AnyKind::Null { pos } => {
                let word = b"null";
                if let Some(c) = word.get(*pos as usize) {
                    out[*c as usize] = true;
                }
            }
            AnyKind::Num(nm) => {
                if !(nm.lead_zero && !nm.dot && !nm.exp) {
                    for b in b'0'..=b'9' {
                        out[b as usize] = true;
                    }
                }
                if !nm.minus && nm.int_digits == 0 {
                    out[b'-' as usize] = true;
                }
                if nm.int_digits > 0 && !nm.dot && !nm.exp {
                    out[b'.' as usize] = true;
                    out[b'e' as usize] = true;
                    out[b'E' as usize] = true;
                }
                if nm.exp && nm.exp_digits == 0 && !nm.exp_sign {
                    out[b'+' as usize] = true;
                    out[b'-' as usize] = true;
                }
                if num_complete(nm) {
                    out[b',' as usize] = true;
                    out[b'}' as usize] = true;
                    out[b']' as usize] = true;
                }
            }
        },
        Some(Frame::VArrFirst { .. }) => {
            out[b'[' as usize] = true;
        }
        Some(Frame::VArrNext { .. }) => {
            out[b',' as usize] = true;
            out[b']' as usize] = true;
        }
        Some(Frame::VObjFirst { .. }) => {
            out[b'{' as usize] = true;
        }
        Some(Frame::VObjKey { spec, seen, buf, in_str, .. }) => {
            if !in_str {
                out[b'"' as usize] = true;
                if seen.is_empty() && spec.required.is_empty() {
                    out[b'}' as usize] = true;
                }
            } else {
                for (p, _) in &spec.props {
                    if seen.contains(p) {
                        continue;
                    }
                    if let Some(c) = p.as_bytes().get(buf.len()) {
                        if p.starts_with(buf.as_str()) {
                            out[*c as usize] = true;
                        }
                    }
                }
                let fresh = !seen.contains(buf);
                if spec.props.iter().any(|(p, _)| p == buf) && fresh {
                    out[b'"' as usize] = true;
                }
                if spec.additional && fresh {
                    for b in 0x20u8..=0xff {
                        out[b as usize] = true;
                    }
                    out[b'"' as usize] = true;
                }
            }
        }
        Some(Frame::VObjColon { .. }) => out[b':' as usize] = true,
        Some(Frame::VObjNext { spec, seen, .. }) => {
            out[b',' as usize] = true;
            let ok = spec.required.iter().all(|r| seen.contains(r));
            if ok {
                out[b'}' as usize] = true;
            }
        }
        None => {}
    }
    out
}

fn literal_next(out: &mut [bool; 256], buf: &str, want: &str) {
    if let Some(c) = want.as_bytes().get(buf.len()) {
        out[*c as usize] = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toolset() -> Vec<serde_json::Value> {
        serde_json::from_str(
            r#"[
            {"name":"set_lights","parameters":{"type":"object","properties":{"room":{"type":"string"},"on":{"type":"boolean"},"brightness":{"type":"integer"}},"required":["room","on"]}},
            {"name":"play_music","parameters":{"type":"object","properties":{"genre":{"type":"string","enum":["rock","jazz"]},"volume":{"type":"integer"}},"required":["genre"]}}
        ]"#,
        )
        .unwrap()
    }

    /// tools with an empty `properties` object (the trained no-arg shape)
    fn noarg_toolset() -> Vec<serde_json::Value> {
        serde_json::from_str(
            r#"[
            {"name":"get_time","parameters":{"type":"object","properties":{}}},
            {"name":"play_music","parameters":{"type":"object","properties":{"genre":{"type":"string","enum":["rock","jazz"]},"volume":{"type":"integer"}},"required":["genre"]}}
        ]"#,
        )
        .unwrap()
    }

    fn drive(g: &mut Grammar, s: &str) -> bool {
        s.bytes().all(|b| g.step(b))
    }

    #[test]
    fn accepts_valid_calls() {
        let mut g = Grammar::compile(&toolset());
        assert!(drive(&mut g, r#"[{"name":"set_lights","arguments":{"room":"kitchen","on":true,"brightness":30}}]"#));
        assert!(g.is_done());

        let mut g = Grammar::compile(&toolset());
        assert!(drive(&mut g, r#"[{"name":"play_music","arguments":{"genre":"jazz"}},{"name":"set_lights","arguments":{"on":false,"room":"hall"}}]"#));
        assert!(g.is_done());

        // refusal
        let mut g = Grammar::compile(&toolset());
        assert!(drive(&mut g, "[]"));
        assert!(g.is_done());
    }

    #[test]
    fn rejects_bad_input() {
        // unknown tool
        let mut g = Grammar::compile(&toolset());
        assert!(!drive(&mut g, r#"[{"name":"hack","arguments":{}}]"#));
        // missing required key
        let mut g = Grammar::compile(&toolset());
        assert!(!drive(&mut g, r#"[{"name":"set_lights","arguments":{"room":"k"}}]"#));
        // wrong type
        let mut g = Grammar::compile(&toolset());
        assert!(!drive(&mut g, r#"[{"name":"set_lights","arguments":{"room":"k","on":"yes"}}]"#));
        // bad enum
        let mut g = Grammar::compile(&toolset());
        assert!(!drive(&mut g, r#"[{"name":"play_music","arguments":{"genre":"pop"}}]"#));
        // malformed json
        let mut g = Grammar::compile(&toolset());
        assert!(!drive(&mut g, r#"[{"name":"set_lights","arguments":{"room":"k","on":true,}}]"#));
        // missing array bracket
        let mut g = Grammar::compile(&toolset());
        assert!(!drive(&mut g, r#"{"name":"set_lights","arguments":{}}"#));
    }

    #[test]
    fn valid_next_guides_decode() {
        let mut g = Grammar::compile(&toolset());
        assert!(drive(&mut g, r#"[{"name":"set_lights","arguments":{"room":"kitchen","on":true"#));
        let v = valid_next(&g);
        assert!(v[b',' as usize] && v[b'}' as usize]);
        assert!(!v[b'"' as usize]);
        // finish and close
        assert!(drive(&mut g, "}}]"));
        assert!(g.is_done());
        let v = valid_next(&g);
        assert!(v.iter().all(|b| !b));
    }

    #[test]
    fn nested_and_array_values() {
        let tools: Vec<serde_json::Value> = serde_json::from_str(
            r#"[{"name":"add_event","parameters":{"type":"object","properties":{"title":{"type":"string"},"days":{"type":"array","items":{"type":"integer"}},"alert":{"type":"object","properties":{"minutes":{"type":"integer"}},"required":["minutes"]}},"required":["title"]}}]"#,
        )
        .unwrap();
        let mut g = Grammar::compile(&tools);
        assert!(drive(&mut g, r#"[{"name":"add_event","arguments":{"title":"x","days":[1,2,3],"alert":{"minutes":5}}}]"#));
        assert!(g.is_done());
        // missing required in nested object
        let mut g = Grammar::compile(&tools);
        assert!(!drive(&mut g, r#"[{"name":"add_event","arguments":{"title":"x","alert":{}}}]"#));
    }

    #[test]
    fn empty_arguments_object_accepted() {
        // no-arg tools emit exactly `"arguments":{}` — the mask must offer
        // '}' right after '{' and the acceptor must reach Done
        let mut g = Grammar::compile(&noarg_toolset());
        assert!(drive(&mut g, r#"[{"name":"get_time","arguments":{}}]"#));
        assert!(g.is_done());

        let mut g = Grammar::compile(&noarg_toolset());
        assert!(drive(&mut g, r#"[{"name":"get_time","arguments":{"#));
        let v = valid_next(&g);
        assert!(v[b'"' as usize] && v[b'}' as usize]);

        // tools with required keys still reject an empty arguments object
        let mut g = Grammar::compile(&noarg_toolset());
        assert!(drive(&mut g, r#"[{"name":"play_music","arguments":{"#));
        let v = valid_next(&g);
        assert!(v[b'"' as usize] && !v[b'}' as usize]);
        assert!(!drive(&mut g, r#"}}]"#));

        // a trailing comma is not resurrected by the empty-object path
        let mut g = Grammar::compile(&noarg_toolset());
        assert!(!drive(&mut g, r#"[{"name":"get_time","arguments":{"genre":"rock",}}]"#));
    }

    #[test]
    fn object_arg_then_required_sibling() {
        // regression: a nested object value used to record its inner key at
        // arguments level, so a required sibling after it dead-ended the call
        let tools: Vec<serde_json::Value> = serde_json::from_str(
            r#"[{"name":"set_reminder","parameters":{"type":"object","properties":{"time":{"type":"string"},"repeat":{"type":"object","properties":{"days":{"type":"array","items":{"type":"string"}}},"required":["days"]}},"required":["time","repeat"]}}]"#,
        )
        .unwrap();
        let mut g = Grammar::compile(&tools);
        assert!(drive(&mut g, r#"[{"name":"set_reminder","arguments":{"repeat":{"days":["mon"]},"time":"now"}}]"#));
        assert!(g.is_done());

        // both orders work
        let mut g = Grammar::compile(&tools);
        assert!(drive(&mut g, r#"[{"name":"set_reminder","arguments":{"time":"now","repeat":{"days":[]}}}]"#));
        assert!(g.is_done());

        // missing required sibling / missing nested required still reject
        let mut g = Grammar::compile(&tools);
        assert!(!drive(&mut g, r#"[{"name":"set_reminder","arguments":{"time":"now"}}]"#));
        let mut g = Grammar::compile(&tools);
        assert!(!drive(&mut g, r#"[{"name":"set_reminder","arguments":{"repeat":{},"time":"now"}}]"#));
        // duplicated outer key is rejected even with a required sibling present
        let mut g = Grammar::compile(&tools);
        assert!(!drive(&mut g, r#"[{"name":"set_reminder","arguments":{"repeat":{"days":["mon"]},"time":"now","repeat":{"days":["tue"]}}}]"#));
    }

    #[test]
    fn duplicate_outer_keys_rejected() {
        // regression: nested keys were recorded at arguments level, so a
        // duplicated outer key slipped through when nothing was required
        let tools: Vec<serde_json::Value> = serde_json::from_str(
            r#"[{"name":"note","parameters":{"type":"object","properties":{"repeat":{"type":"object","properties":{"days":{"type":"array","items":{"type":"string"}}},"required":["days"]}}}}]"#,
        )
        .unwrap();
        let mut g = Grammar::compile(&tools);
        assert!(drive(&mut g, r#"[{"name":"note","arguments":{"repeat":{"days":["mon"]}}}]"#));
        let mut g = Grammar::compile(&tools);
        assert!(!drive(&mut g, r#"[{"name":"note","arguments":{"repeat":{"days":["mon"]},"repeat":{"days":["tue"]}}}]"#));
    }

    #[test]
    fn spaces_in_tool_names_keys_and_enums() {
        // the acceptor side of decoded-space matching: schema strings with
        // spaces must prefix-match and enum-compare in decoded space
        let tools: Vec<serde_json::Value> = serde_json::from_str(
            r#"[{"name":"set lights","parameters":{"type":"object","properties":{"living room":{"type":"string","enum":["bright white","dim red"]},"on":{"type":"boolean"}},"required":["living room","on"]}}]"#,
        )
        .unwrap();
        let mut g = Grammar::compile(&tools);
        assert!(drive(&mut g, r#"[{"name":"set lights","arguments":{"living room":"dim red","on":true}}]"#));
        assert!(g.is_done());
        // unknown enum value with spaces
        let mut g = Grammar::compile(&tools);
        assert!(!drive(&mut g, r#"[{"name":"set lights","arguments":{"living room":"bright blue","on":true}}]"#));
        // tool name without its space is not a prefix match
        let mut g = Grammar::compile(&tools);
        assert!(!drive(&mut g, r#"[{"name":"setlights","arguments":{"living room":"dim red","on":true}}]"#));
    }

    #[test]
    fn free_form_object_values() {
        // object with no declared properties accepts arbitrary keys/values
        let tools: Vec<serde_json::Value> = serde_json::from_str(
            r#"[{"name":"send","parameters":{"type":"object","properties":{"payload":{"type":"object"},"strict":{"type":"object","properties":{"code":{"type":"integer"}}}},"required":["payload"]}}]"#,
        )
        .unwrap();
        let mut g = Grammar::compile(&tools);
        assert!(drive(&mut g, r#"[{"name":"send","arguments":{"payload":{"any_key":[1,"two",true,null,{"deep":false}],"n":-1.5e2}}}]"#));
        assert!(g.is_done());
        // declared-property object without additionalProperties stays strict
        let mut g = Grammar::compile(&tools);
        assert!(!drive(&mut g, r#"[{"name":"send","arguments":{"payload":{},"strict":{"wrong":1}}}]"#));

        // additionalProperties: true accepts unknown keys beside declared ones
        let tools: Vec<serde_json::Value> = serde_json::from_str(
            r#"[{"name":"send","parameters":{"type":"object","properties":{"strict":{"type":"object","properties":{"code":{"type":"integer"}},"additionalProperties":true}},"required":["strict"]}}]"#,
        )
        .unwrap();
        let mut g = Grammar::compile(&tools);
        assert!(drive(&mut g, r#"[{"name":"send","arguments":{"strict":{"code":1,"other":[true,null]}}}]"#));
        assert!(g.is_done());
    }

    #[test]
    fn nested_empty_object_accepted() {
        // all-optional nested object may close with '{}'
        let tools: Vec<serde_json::Value> = serde_json::from_str(
            r#"[{"name":"remind","parameters":{"type":"object","properties":{"alert":{"type":"object","properties":{"minutes":{"type":"integer"}}},"title":{"type":"string"}},"required":["title"]}}]"#,
        )
        .unwrap();
        let mut g = Grammar::compile(&tools);
        assert!(drive(&mut g, r#"[{"name":"remind","arguments":{"title":"x","alert":{}}}]"#));
        assert!(g.is_done());
        let mut g = Grammar::compile(&tools);
        assert!(drive(&mut g, r#"[{"name":"remind","arguments":{"alert":{},"title":"x"}}]"#));
        assert!(g.is_done());

        // nested required keys are enforced wherever the object closes
        let tools: Vec<serde_json::Value> = serde_json::from_str(
            r#"[{"name":"remind","parameters":{"type":"object","properties":{"alert":{"type":"object","properties":{"minutes":{"type":"integer"}},"required":["minutes"]}}}}]"#,
        )
        .unwrap();
        let mut g = Grammar::compile(&tools);
        assert!(!drive(&mut g, r#"[{"name":"remind","arguments":{"alert":{}}}]"#));
        // ...including after a complete pair, and trailing commas stay illegal
        let mut g = Grammar::compile(&tools);
        assert!(!drive(&mut g, r#"[{"name":"remind","arguments":{"alert":{"minutes":5,}}}]"#));
    }

    #[test]
    fn escaped_chars_in_strings_and_enums() {
        // escaped bytes join the decoded buffer, so enums containing quotes
        // and escapes compare in decoded space
        let tools: Vec<serde_json::Value> = serde_json::from_str(
            r#"[{"name":"say","parameters":{"type":"object","properties":{"phrase":{"type":"string","enum":["say \"hi\"","a\\b","A"]}},"required":["phrase"]}}]"#,
        )
        .unwrap();
        let mut g = Grammar::compile(&tools);
        assert!(drive(&mut g, r#"[{"name":"say","arguments":{"phrase":"say \"hi\""}}]"#));
        assert!(g.is_done());
        let mut g = Grammar::compile(&tools);
        assert!(drive(&mut g, r#"[{"name":"say","arguments":{"phrase":"a\\b"}}]"#));
        // \u0041 decodes to 'A'
        let mut g = Grammar::compile(&tools);
        assert!(drive(&mut g, r#"[{"name":"say","arguments":{"phrase":"\u0041"}}]"#));
        assert!(g.is_done());
        // invalid escape, truncated \u escape, and wrong enum are rejected
        let mut g = Grammar::compile(&tools);
        assert!(!drive(&mut g, r#"[{"name":"say","arguments":{"phrase":"\x"}}]"#));
        let mut g = Grammar::compile(&tools);
        assert!(!drive(&mut g, r#"[{"name":"say","arguments":{"phrase":"\u12"}}]"#));
        let mut g = Grammar::compile(&tools);
        assert!(!drive(&mut g, r#"[{"name":"say","arguments":{"phrase":"say \"ho\""}}]"#));
    }

    #[test]
    fn rejects_invalid_numbers() {
        let tools: Vec<serde_json::Value> = serde_json::from_str(
            r#"[{"name":"calc","parameters":{"type":"object","properties":{"n":{"type":"number"},"i":{"type":"integer"}},"required":["n","i"]}}]"#,
        )
        .unwrap();
        for bad in ["1e", "1.", "007", "01", "1e+", "1.e3", "-", "1..2", "1ee2"] {
            let s = format!(r#"[{{"name":"calc","arguments":{{"n":{bad},"i":1}}}}]"#);
            let mut g = Grammar::compile(&tools);
            assert!(!drive(&mut g, &s), "accepted {bad}");
        }
        for good in ["0", "-0", "10", "-12", "0.5", "1e3", "1.5E-2", "0e0", "3.0"] {
            let s = format!(r#"[{{"name":"calc","arguments":{{"n":{good},"i":1}}}}]"#);
            let mut g = Grammar::compile(&tools);
            assert!(drive(&mut g, &s), "rejected {good}");
        }
        for bad in ["007", "01", "-"] {
            let s = format!(r#"[{{"name":"calc","arguments":{{"n":1,"i":{bad}}}}}]"#);
            let mut g = Grammar::compile(&tools);
            assert!(!drive(&mut g, &s), "accepted int {bad}");
        }
        let mut g = Grammar::compile(&tools);
        assert!(drive(&mut g, r#"[{"name":"calc","arguments":{"n":1,"i":-12}}]"#));
    }

    #[test]
    fn null_values_accepted() {
        // free-form values (no item schema) include JSON null
        let tools: Vec<serde_json::Value> = serde_json::from_str(
            r#"[{"name":"mix","parameters":{"type":"object","properties":{"payload":{"type":"object"},"tags":{"type":"array"}},"required":["payload"]}}]"#,
        )
        .unwrap();
        let mut g = Grammar::compile(&tools);
        assert!(drive(&mut g, r#"[{"name":"mix","arguments":{"payload":{"a":null},"tags":[null,1,"x",true,{"k":null}]}}]"#));
        assert!(g.is_done());
        // trailing garbage after null is still rejected
        let mut g = Grammar::compile(&tools);
        assert!(!drive(&mut g, r#"[{"name":"mix","arguments":{"payload":{"a":nullx}}}]"#));
    }
}
