//! Byte-level grammar for Needle tool calls, compiled from JSON Schemas.
//!
//! The training target is `<tool_call>[{"name":"X","arguments":{...}},...]</tool_call>`.
//! A streaming byte acceptor: `step(byte)` advances the parse; `valid_next()`
//! lists the bytes that keep the parse alive so the decoder can mask the
//! vocabulary to tokens whose every byte is acceptable.
//!
//! Guarantees: well-formed JSON in the trained shape — call objects with a
//! `name` (one of the declared tools) and `arguments` matching the schema
//! (required keys present, value types, string enums). Numeric ranges, string
//! patterns and container sizes are NOT enforced (validate post-hoc).

use std::collections::BTreeSet;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct ToolSpec {
    pub name: String,
    pub props: Vec<(String, VSpec)>,
    pub required: BTreeSet<String>,
}

#[derive(Debug, Clone)]
pub enum VSpec {
    Str { en: Option<Arc<Vec<String>>>, max_len: Option<usize> },
    Int,
    Num,
    Bool,
    Arr { items: Box<VSpec>, max_items: Option<usize> },
    Obj { props: Vec<(String, VSpec)>, required: BTreeSet<String> },
    Any,
}

type ObjRc = Arc<(Vec<(String, VSpec)>, BTreeSet<String>)>;

/// What a completed value frame turns into.
#[derive(Debug, Clone)]
enum Cont {
    /// an argument value in the current tool's arguments object
    Arg,
    /// an array element; carries the parent array state and the array's own cont
    ArrItem { items: Box<VSpec>, max: Option<usize>, count: usize, cont: Box<Cont> },
    /// an object value; carries parent object state + key to record
    ObjValue { spec: ObjRc, seen: BTreeSet<String>, key: String, cont: Box<Cont> },
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
    /// reading a property-name string (prefix-matched against the tool's props)
    ArgsKey { buf: String, in_str: bool },
    ArgsColon,
    /// after an argument value: ',' (next key) or '}' (close arguments)
    ArgsNext,
    /// after the arguments object: '}' closes the call object
    CallObjEnd,
    /// after a call: ',' (next call) or ']' (close array)
    CallEnd,
    Done,
    // ---- value frames (all carry their continuation) ----
    VStr { open: bool, esc: bool, en: Option<Arc<Vec<String>>>, sbuf: Vec<u8>, cont: Cont },
    VInt { minus: bool, digits: usize, cont: Cont },
    VNum(Num, Cont),
    VBool { branch: Option<bool>, pos: usize, cont: Cont },
    VAny { kind: AnyKind, cont: Cont },
    VArrFirst { items: Box<VSpec>, max: Option<usize>, cont: Cont },
    VArrNext { items: Box<VSpec>, max: Option<usize>, count: usize, cont: Cont },
    VObjFirst { spec: ObjRc, cont: Cont },
    VObjKey { spec: ObjRc, seen: BTreeSet<String>, buf: String, in_str: bool, cont: Cont },
    VObjColon { spec: ObjRc, seen: BTreeSet<String>, cont: Cont },
    VObjNext { spec: ObjRc, seen: BTreeSet<String>, cont: Cont },
}

#[derive(Debug, Clone, Default)]
struct Num {
    minus: bool,
    int_digits: usize,
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
                VSpec::Obj { props, required }
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
    /// property key completed by ArgsKey / VObjKey, awaiting ':' then value
    pending: Option<(String, VSpec)>,
    args_seen: BTreeSet<String>,
}

fn value_frame(spec: &VSpec, cont: Cont) -> Frame {
    match spec {
        VSpec::Str { en, .. } => Frame::VStr { open: false, esc: false, en: en.clone(), sbuf: Vec::new(), cont },
        VSpec::Int => Frame::VInt { minus: false, digits: 0, cont },
        VSpec::Num => Frame::VNum(Num::default(), cont),
        VSpec::Bool => Frame::VBool { branch: None, pos: 0, cont },
        VSpec::Arr { items, max_items } => Frame::VArrFirst { items: items.clone(), max: *max_items, cont },
        VSpec::Obj { props, required } => Frame::VObjFirst {
            spec: Arc::new((props.clone(), required.clone())),
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
        b'e' | b'E' => !nm.exp && nm.int_digits > 0 && replace(&mut nm.exp, true),
        _ => false,
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
                let (props, required) = match VSpec::from_schema(&params) {
                    VSpec::Obj { props, required } => (props, required),
                    _ => (Vec::new(), BTreeSet::new()),
                };
                ToolSpec { name, props, required }
            })
            .collect();
        Grammar {
            tools,
            frames: vec![Frame::ArrOpen],
            cur_tool: usize::MAX,
            pending: None,
            args_seen: BTreeSet::new(),
        }
    }

    pub fn is_done(&self) -> bool {
        matches!(self.frames.last(), Some(Frame::Done))
    }

    /// The frame a completed value resolves to.
    fn resolve(&mut self, cont: Cont) -> Frame {
        match cont {
            Cont::Arg => {
                if let Some((k, _)) = self.pending.take() {
                    self.args_seen.insert(k);
                }
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
                            self.frames[depth - 1] = Frame::ArgsKey { buf: String::new(), in_str: false };
                        }
                        byte == b'{'
                    }
                    Frame::ArgsKey { buf, in_str } => {
                        if !in_str {
                            if byte == b'"' {
                                self.frames[depth - 1] =
                                    Frame::ArgsKey { buf: String::new(), in_str: true };
                                true
                            } else {
                                false
                            }
                        } else if byte == b'"' {
                            let t = &self.tools[self.cur_tool];
                            match t.props.iter().find(|(p, _)| p == buf) {
                                Some((_, spec)) if !self.args_seen.contains(buf) => {
                                    self.pending = Some((buf.clone(), spec.clone()));
                                    self.frames[depth - 1] = Frame::ArgsColon;
                                    true
                                }
                                _ => false,
                            }
                        } else if byte < 0x20 {
                            false
                        } else {
                            let mut nb = buf.clone();
                            nb.push(byte as char);
                            let t = &self.tools[self.cur_tool];
                            if t.props.iter().any(|(p, _)| p.starts_with(nb.as_str())) {
                                self.frames[depth - 1] = Frame::ArgsKey { buf: nb, in_str: true };
                                true
                            } else {
                                false
                            }
                        }
                    }
                    Frame::ArgsColon => {
                        if byte == b':' {
                            let spec =
                                self.pending.as_ref().map(|(_, s)| s.clone()).unwrap_or(VSpec::Any);
                            self.frames[depth - 1] = value_frame(&spec, Cont::Arg);
                        }
                        byte == b':'
                    }
                    Frame::ArgsNext => match byte {
                        b',' => {
                            self.frames[depth - 1] = Frame::ArgsKey { buf: String::new(), in_str: false };
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
                                self.frames[depth - 1] =
                                    Frame::VStr { open: true, esc: false, en, sbuf, cont };
                                return true;
                            }
                            return false;
                        }
                        if esc {
                            let ok = matches!(
                                byte,
                                b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' | b'u'
                            );
                            if ok {
                                self.frames[depth - 1] =
                                    Frame::VStr { open: true, esc: false, en, sbuf, cont };
                            }
                            return ok;
                        }
                        let ok = match byte {
                            b'\\' => {
                                self.frames[depth - 1] =
                                    Frame::VStr { open: true, esc: true, en, sbuf, cont };
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
                                let en_ok = match &en {
                                    Some(opts) => opts
                                        .iter()
                                        .any(|o| o.as_bytes().starts_with(sbuf.as_slice())),
                                    None => true,
                                };
                                if en_ok {
                                    self.frames[depth - 1] =
                                        Frame::VStr { open: true, esc: false, en, sbuf, cont };
                                }
                                en_ok
                            }
                        };
                        return ok;
                    }
                    Frame::VInt { minus, digits, cont } => {
                        let (mut minus, mut digits, cont) = (*minus, *digits, cont.clone());
                        let _ = &mut minus;
                        match byte {
                            b'-' if !minus && digits == 0 => {
                                minus = true;
                                self.frames[depth - 1] = Frame::VInt { minus, digits, cont };
                                true
                            }
                            b'0'..=b'9' => {
                                digits += 1;
                                self.frames[depth - 1] = Frame::VInt { minus, digits, cont };
                                true
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
                        } else if nm.int_digits > 0 {
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
                                    b'[' => Some(Frame::VArrFirst {
                                        items: Box::new(VSpec::Any),
                                        max: None,
                                        cont: cont.clone(),
                                    }),
                                    b'{' => Some(Frame::VObjFirst {
                                        spec: Arc::new((Vec::new(), BTreeSet::new())),
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
                                let esc = esc;
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
                            AnyKind::Num(mut nm) => {
                                if step_num(&mut nm, byte) {
                                    self.frames[depth - 1] =
                                        Frame::VAny { kind: AnyKind::Num(nm), cont };
                                    true
                                } else if nm.int_digits > 0 {
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
                        match byte {
                            b'[' => {
                                self.frames[depth - 1] =
                                    Frame::VArrNext { items, max, count: 0, cont };
                                true
                            }
                            b']' => {
                                self.frames[depth - 1] = self.complete(cont);
                                true
                            }
                            _ => false,
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
                        match byte {
                            b'{' => {
                                self.frames[depth - 1] = Frame::VObjKey {
                                    spec,
                                    seen: BTreeSet::new(),
                                    buf: String::new(),
                                    in_str: false,
                                    cont,
                                };
                                true
                            }
                            b'}' => {
                                self.frames[depth - 1] = self.complete(cont);
                                true
                            }
                            _ => false,
                        }
                    }
                    Frame::VObjKey { spec, seen, buf, in_str, cont } => {
                        let (spec, seen, buf, in_str, cont) =
                            (spec.clone(), seen.clone(), buf.clone(), *in_str, cont.clone());
                        if !in_str {
                            if byte == b'"' {
                                self.frames[depth - 1] = Frame::VObjKey {
                                    spec, seen, buf: String::new(), in_str: true, cont,
                                };
                                true
                            } else {
                                false
                            }
                        } else if byte == b'"' {
                            if seen.contains(&buf) {
                                return false;
                            }
                            match spec.0.iter().find(|(p, _)| p == &buf) {
                                Some((_, vspec)) => {
                                    self.pending = Some((buf.clone(), vspec.clone()));
                                    self.frames[depth - 1] =
                                        Frame::VObjColon { spec, seen, cont };
                                    true
                                }
                                None => false,
                            }
                        } else if byte < 0x20 {
                            false
                        } else {
                            let mut nb = buf.clone();
                            nb.push(byte as char);
                            if spec.0.iter().any(|(p, _)| p.starts_with(nb.as_str())) {
                                self.frames[depth - 1] =
                                    Frame::VObjKey { spec, seen, buf: nb, in_str: true, cont };
                                true
                            } else {
                                false
                            }
                        }
                    }
                    Frame::VObjColon { spec, seen, cont } => {
                        let (spec, seen, cont) = (spec.clone(), seen.clone(), cont.clone());
                        if byte == b':' {
                            let vspec =
                                self.pending.as_ref().map(|(_, s)| s.clone()).unwrap_or(VSpec::Any);
                            let key =
                                self.pending.as_ref().map(|(k, _)| k.clone()).unwrap_or_default();
                            self.frames[depth - 1] = value_frame(
                                &vspec,
                                Cont::ObjValue { spec, seen, key, cont: Box::new(cont) },
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
                                self.frames[depth - 1] = self.complete(cont);
                                true
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
        Some(Frame::ArgsKey { buf, in_str }) => {
            if !in_str {
                out[b'"' as usize] = true;
            } else if let Some(t) = g.tools.get(g.cur_tool) {
                for (p, _) in &t.props {
                    if let Some(c) = p.as_bytes().get(buf.len()) {
                        if p.starts_with(buf.as_str()) {
                            out[*c as usize] = true;
                        }
                    }
                }
                if t.props.iter().any(|(p, _)| p == buf) {
                    out[b'"' as usize] = true;
                }
            }
        }
        Some(Frame::ArgsColon) => out[b':' as usize] = true,
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
            } else if *esc {
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
        Some(Frame::VInt { digits, .. }) => {
            for b in b'0'..=b'9' {
                out[b as usize] = true;
            }
            if *digits == 0 {
                out[b'-' as usize] = true;
            }
            if *digits > 0 {
                out[b',' as usize] = true;
                out[b'}' as usize] = true;
                out[b']' as usize] = true;
            }
        }
        Some(Frame::VNum(nm, _)) => {
            for b in b'0'..=b'9' {
                out[b as usize] = true;
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
            if nm.int_digits > 0 {
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
                for b in 0x20u8..=0xff {
                    out[b as usize] = true;
                }
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
            AnyKind::Num(nm) => {
                for b in b'0'..=b'9' {
                    out[b as usize] = true;
                }
                if !nm.minus && nm.int_digits == 0 {
                    out[b'-' as usize] = true;
                }
                if nm.int_digits > 0 && !nm.dot && !nm.exp {
                    out[b'.' as usize] = true;
                    out[b'e' as usize] = true;
                    out[b'E' as usize] = true;
                }
            }
        },
        Some(Frame::VArrFirst { .. }) => {
            out[b'[' as usize] = true;
            out[b']' as usize] = true;
        }
        Some(Frame::VArrNext { .. }) => {
            out[b',' as usize] = true;
            out[b']' as usize] = true;
        }
        Some(Frame::VObjFirst { .. }) => {
            out[b'{' as usize] = true;
            out[b'}' as usize] = true;
        }
        Some(Frame::VObjKey { spec, seen, buf, in_str, .. }) => {
            if !in_str {
                out[b'"' as usize] = true;
            } else {
                for (p, _) in &spec.0 {
                    if seen.contains(p) {
                        continue;
                    }
                    if let Some(c) = p.as_bytes().get(buf.len()) {
                        if p.starts_with(buf.as_str()) {
                            out[*c as usize] = true;
                        }
                    }
                }
                if spec.0.iter().any(|(p, _)| p == buf) {
                    out[b'"' as usize] = true;
                }
            }
        }
        Some(Frame::VObjColon { .. }) => out[b':' as usize] = true,
        Some(Frame::VObjNext { .. }) => {
            out[b',' as usize] = true;
            out[b'}' as usize] = true;
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
}
