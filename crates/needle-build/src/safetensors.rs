//! Minimal pure-Rust safetensors reader (f32/f16/bf16) with metadata.

use half::f16;
use std::collections::HashMap;

#[derive(Debug)]
pub struct Safetensors {
    pub tensors: HashMap<String, Vec<f32>>,
    pub shapes: HashMap<String, Vec<usize>>,
    pub metadata: HashMap<String, String>,
}

fn le_u64(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().unwrap())
}

impl Safetensors {
    pub fn read(path: &std::path::Path) -> Result<Safetensors, String> {
        let raw = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
        Self::from_bytes(&raw)
    }

    pub fn from_bytes(raw: &[u8]) -> Result<Safetensors, String> {
        if raw.len() < 8 {
            return Err("truncated safetensors".into());
        }
        let hlen = le_u64(raw, 0) as usize;
        if raw.len() < 8 + hlen {
            return Err("truncated header".into());
        }
        let header: serde_json::Value =
            serde_json::from_slice(&raw[8..8 + hlen]).map_err(|e| format!("header json: {e}"))?;
        let mut tensors = HashMap::new();
        let mut shapes = HashMap::new();
        let mut metadata = HashMap::new();
        if let Some(meta) = header.get("__metadata__").and_then(|m| m.as_object()) {
            for (k, v) in meta {
                if let Some(s) = v.as_str() {
                    metadata.insert(k.clone(), s.to_string());
                }
            }
        }
        for (name, info) in header.as_object().cloned().unwrap_or_default() {
            if name == "__metadata__" {
                continue;
            }
            let dtype = info.get("dtype").and_then(|d| d.as_str()).unwrap_or("");
            let shape: Vec<usize> = info
                .get("shape")
                .and_then(|s| s.as_array())
                .map(|a| a.iter().filter_map(|v| v.as_u64().map(|n| n as usize)).collect())
                .unwrap_or_default();
            let offsets = info
                .get("data_offsets")
                .and_then(|o| o.as_array())
                .and_then(|a| {
                    Some((a.first()?.as_u64()? as usize, a.get(1)?.as_u64()? as usize))
                })
                .ok_or("missing data_offsets")?;
            let start = 8 + hlen + offsets.0;
            let end = 8 + hlen + offsets.1;
            if end > raw.len() {
                return Err(format!("tensor {name} out of bounds"));
            }
            let blob = &raw[start..end];
            let count: usize = shape.iter().product();
            let data: Vec<f32> = match dtype {
                "F32" => blob
                    .chunks_exact(4)
                    .take(count)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect(),
                "F16" => blob
                    .chunks_exact(2)
                    .take(count)
                    .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
                    .collect(),
                "BF16" => blob
                    .chunks_exact(2)
                    .take(count)
                    .map(|c| {
                        let bits = u16::from_le_bytes([c[0], c[1]]);
                        f32::from_bits((bits as u32) << 16)
                    })
                    .collect(),
                other => return Err(format!("unsupported dtype {other} for {name}")),
            };
            if data.len() != count {
                return Err(format!("tensor {name} size mismatch"));
            }
            shapes.insert(name.clone(), shape);
            tensors.insert(name, data);
        }
        Ok(Safetensors { tensors, shapes, metadata })
    }

    pub fn get(&self, name: &str) -> Option<&[f32]> {
        self.tensors.get(name).map(|v| v.as_slice())
    }

    pub fn shape(&self, name: &str) -> Option<&[usize]> {
        self.shapes.get(name).map(|v| v.as_slice())
    }
}
