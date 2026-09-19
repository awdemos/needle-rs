//! The reference tokenizer for Needle 3 — a port of `RefTokenizer` in
//! `needle/model/export.py`. It reads the self-contained SentencePiece BPE dump
//! embedded in every `.cact` archive (no SentencePiece dependency) and implements
//! marker-aware encode (`<|im_start|>`, `<tool_call>`, ...), greedy score-based BPE
//! merges, byte fallback, and decode.

use std::collections::HashMap;
use std::fmt;

pub const PAD_ID: u32 = 0;
pub const EOS_ID: u32 = 1;
pub const BOS_ID: u32 = 2;
pub const UNK_ID: u32 = 3;

pub const IM_START: &str = "<|im_start|>";
pub const IM_END: &str = "<|im_end|>";
pub const THINK_START: &str = "<think>";
pub const THINK_END: &str = "</think>";
pub const TOOLS_START: &str = "<tools>";
pub const TOOLS_END: &str = "</tools>";
pub const TOOL_CALL_START: &str = "<tool_call>";
pub const TOOL_CALL_END: &str = "</tool_call>";
pub const TOOL_RESULT_START: &str = "<tool_result>";
pub const TOOL_RESULT_END: &str = "</tool_result>";
pub const CONTEXT_START: &str = "<context>";
pub const CONTEXT_END: &str = "</context>";
pub const EXTRACT_START: &str = "<extract>";
pub const EXTRACT_END: &str = "</extract>";
pub const SCHEMA_START: &str = "<schema>";
pub const SCHEMA_END: &str = "</schema>";

pub const CHAT_MARKERS: [&str; 16] = [
    IM_START, IM_END, THINK_START, THINK_END, TOOLS_START, TOOLS_END, TOOL_CALL_START,
    TOOL_CALL_END, TOOL_RESULT_START, TOOL_RESULT_END, CONTEXT_START, CONTEXT_END,
    EXTRACT_START, EXTRACT_END, SCHEMA_START, SCHEMA_END,
];

pub const TK_NORMAL: u8 = 0;
pub const TK_UNKNOWN: u8 = 1;
pub const TK_CONTROL: u8 = 2;
pub const TK_USER_DEFINED: u8 = 3;
pub const TK_BYTE: u8 = 4;

const SP_META_SPACE: char = '\u{2581}'; // ▁

#[derive(Debug)]
pub enum Error {
    Truncated,
    BadUtf8,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Truncated => write!(f, "truncated tokenizer blob"),
            Error::BadUtf8 => write!(f, "invalid UTF-8 in tokenizer blob"),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone)]
pub struct Tokenizer {
    pub pieces: Vec<String>,
    pub scores: Vec<f32>,
    pub types: Vec<u8>,
    pub add_dummy: bool,
    pub byte_fallback: bool,
    pub unk_id: u32,
    p2id: HashMap<String, u32>,
    /// Byte piece for each byte value, e.g. the piece "<0x41>" for b'A'.
    byte_id: HashMap<u8, u32>,
    /// USER_DEFINED (marker) pieces, longest first.
    markers: Vec<String>,
}

fn le_u32(blob: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(blob[at..at + 4].try_into().unwrap())
}

impl Tokenizer {
    pub fn from_blob(blob: &[u8]) -> Result<Tokenizer> {
        if blob.len() < 24 {
            return Err(Error::Truncated);
        }
        let n = le_u32(blob, 0) as usize;
        let unk_id = le_u32(blob, 16);
        let add_dummy = blob[20] != 0;
        let byte_fallback = blob[21] != 0;
        let mut off = 24;
        let mut pieces = Vec::with_capacity(n);
        let mut scores = Vec::with_capacity(n);
        let mut types = Vec::with_capacity(n);
        for _ in 0..n {
            if blob.len() < off + 7 {
                return Err(Error::Truncated);
            }
            let score = f32::from_le_bytes(blob[off..off + 4].try_into().unwrap());
            let t = blob[off + 4];
            let len = u16::from_le_bytes(blob[off + 5..off + 7].try_into().unwrap()) as usize;
            off += 7;
            if blob.len() < off + len {
                return Err(Error::Truncated);
            }
            let piece = std::str::from_utf8(&blob[off..off + len])
                .map_err(|_| Error::BadUtf8)?
                .to_string();
            off += len;
            pieces.push(piece);
            scores.push(score);
            types.push(t);
        }
        let p2id: HashMap<String, u32> = pieces
            .iter()
            .enumerate()
            .map(|(i, p)| (p.clone(), i as u32))
            .collect();
        let byte_id: HashMap<u8, u32> = pieces
            .iter()
            .enumerate()
            .filter(|(_, p)| p.len() == 6 && p.starts_with("<0x") && p.ends_with('>'))
            .filter_map(|(i, p)| u8::from_str_radix(&p[3..5], 16).ok().map(|b| (b, i as u32)))
            .collect();
        let mut markers: Vec<String> = pieces
            .iter()
            .zip(&types)
            .filter(|(_, &t)| t == TK_USER_DEFINED)
            .map(|(p, _)| p.clone())
            .collect();
        markers.sort_by_key(|m| std::cmp::Reverse(m.len()));
        Ok(Tokenizer {
            pieces,
            scores,
            types,
            add_dummy,
            byte_fallback,
            unk_id,
            p2id,
            byte_id,
            markers,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.pieces.len()
    }

    pub fn piece(&self, id: u32) -> &str {
        &self.pieces[id as usize]
    }

    pub fn piece_id(&self, piece: &str) -> Option<u32> {
        self.p2id.get(piece).copied()
    }

    pub fn is_byte(&self, id: u32) -> bool {
        self.types[id as usize] == TK_BYTE
    }

    fn bpe(&self, seg: &str) -> Vec<u32> {
        let mut syms: Vec<String> = seg.chars().map(|c| c.to_string()).collect();
        while syms.len() > 1 {
            let mut best_score = f32::NEG_INFINITY;
            let mut best_j: Option<usize> = None;
            for j in 0..syms.len() - 1 {
                if let Some(&idx) = self.p2id.get(&format!("{}{}", syms[j], syms[j + 1])[..]) {
                    if best_j.is_none() || self.scores[idx as usize] > best_score {
                        best_score = self.scores[idx as usize];
                        best_j = Some(j);
                    }
                }
            }
            match best_j {
                Some(j) => {
                    let merged = format!("{}{}", syms[j], syms[j + 1]);
                    syms.splice(j..j + 2, [merged]);
                }
                None => break,
            }
        }
        let mut ids = Vec::new();
        for s in syms {
            match self.p2id.get(&s) {
                Some(&idx) => ids.push(idx),
                None if self.byte_fallback => {
                    for b in s.as_bytes() {
                        if let Some(&idx) = self.byte_id.get(b) {
                            ids.push(idx);
                        } else {
                            ids.push(self.unk_id);
                        }
                    }
                }
                None => ids.push(self.unk_id),
            }
        }
        ids
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }
        let mut esc: String = text.replace(' ', &SP_META_SPACE.to_string());
        if self.add_dummy {
            esc.insert(0, SP_META_SPACE);
        }
        let mut ids = Vec::new();
        let mut buf = String::new();
        let mut i = 0;
        while i < esc.len() {
            let rest = &esc[i..];
            let marker = self
                .markers
                .iter()
                .find(|m| rest.starts_with(m.as_str()));
            match marker {
                Some(m) => {
                    ids.extend(self.bpe(&buf));
                    buf.clear();
                    ids.push(self.p2id[m.as_str()]);
                    i += m.len();
                }
                None => {
                    let ch = rest.chars().next().unwrap();
                    buf.push(ch);
                    i += ch.len_utf8();
                }
            }
        }
        ids.extend(self.bpe(&buf));
        ids
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut buf: Vec<u8> = Vec::new();
        for &id in ids {
            let t = self.types[id as usize];
            if t == TK_BYTE {
                let p = &self.pieces[id as usize];
                if let Ok(b) = u8::from_str_radix(&p[3..5], 16) {
                    buf.push(b);
                }
            } else if t == TK_CONTROL || t == TK_UNKNOWN {
                continue;
            } else {
                buf.extend_from_slice(self.pieces[id as usize].as_bytes());
            }
        }
        let mut text = String::from_utf8_lossy(&buf).into_owned();
        text = text.replace(SP_META_SPACE, " ");
        if self.add_dummy && text.starts_with(' ') {
            text.remove(0);
        }
        text
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn models_dir() -> std::path::PathBuf {
        std::env::var("NEEDLE_MODELS_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| {
                std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models")
            })
    }

    fn load() -> Option<Tokenizer> {
        let path = models_dir().join("needle3.cact");
        if !path.exists() {
            return None;
        }
        let ar = needle_format::read_archive(&path).unwrap();
        Some(Tokenizer::from_blob(ar.tokenizer_blob().unwrap()).unwrap())
    }

    #[test]
    fn loads_and_roundtrips() {
        let Some(tok) = load() else {
            eprintln!("skip: needle3.cact not downloaded");
            return;
        };
        assert_eq!(tok.vocab_size(), 8192);
        for text in [
            "dim the living room lights to 30",
            "what's it like in Lagos right now?",
            "Invoice from Acme Corp, $1,200.00 — ref #A-42/B",
            "turn on the kitchen light and set brightness to 75%",
        ] {
            let ids = tok.encode(text);
            assert!(!ids.is_empty());
            let back = tok.decode(&ids);
            assert_eq!(back, text, "roundtrip failed for {text:?}");
        }
    }

    #[test]
    fn markers_encode_as_single_tokens() {
        let Some(tok) = load() else {
            eprintln!("skip: needle3.cact not downloaded");
            return;
        };
        let ids = tok.encode(&format!("{IM_START}system\nhello{IM_END}"));
        assert_eq!(ids[0], 4);
        assert!(ids.contains(&5));
        // marker text is not split by BPE
        assert_eq!(tok.piece(4), IM_START);
    }

    #[test]
    fn byte_fallback_covers_unicode() {
        let Some(tok) = load() else {
            eprintln!("skip: needle3.cact not downloaded");
            return;
        };
        let text = "café ☃";
        let ids = tok.encode(text);
        let back = tok.decode(&ids);
        assert_eq!(back, text);
    }
}
