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
        // checked: a huge claimed header length must not wrap `8 + hlen`
        let header_end = 8usize.checked_add(hlen).ok_or("header length overflows")?;
        if raw.len() < header_end {
            return Err("truncated header".into());
        }
        let header: serde_json::Value =
            serde_json::from_slice(&raw[8..header_end]).map_err(|e| format!("header json: {e}"))?;
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
                .ok_or_else(|| format!("tensor {name}: missing data_offsets"))?;
            if offsets.0 > offsets.1 {
                return Err(format!("tensor {name}: data_offsets start {} > end {}", offsets.0, offsets.1));
            }
            let start = header_end.checked_add(offsets.0).ok_or("tensor range overflows")?;
            let end = header_end.checked_add(offsets.1).ok_or("tensor range overflows")?;
            if end > raw.len() {
                return Err(format!("tensor {name} out of bounds"));
            }
            let blob = &raw[start..end];
            let count: usize = shape.iter().product();
            let data: Vec<f32> = match dtype {
                "F32" => blob
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .take(count)
                    .map(|c| f32::from_le_bytes(*c))
                    .collect(),
                "F16" => blob
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .take(count)
                    .map(|c| f16::from_le_bytes(*c).to_f32())
                    .collect(),
                "BF16" => blob
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .take(count)
                    .map(|c| f32::from_bits((u16::from_le_bytes(*c) as u32) << 16))
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

#[cfg(test)]
mod tests {
    use super::*;

    fn file_with_header(header: &serde_json::Value, data: &[u8]) -> Vec<u8> {
        let h = serde_json::to_vec(header).unwrap();
        let mut raw = (h.len() as u64).to_le_bytes().to_vec();
        raw.extend_from_slice(&h);
        raw.extend_from_slice(data);
        raw
    }

    #[test]
    fn rejects_huge_claimed_header_without_overflow() {
        // a u64::MAX header length must error, not wrap 8 + hlen and panic
        let mut raw = u64::MAX.to_le_bytes().to_vec();
        raw.extend_from_slice(b"not a real header at all");
        let err = Safetensors::from_bytes(&raw).unwrap_err();
        assert!(err.contains("header"), "{err}");
    }

    #[test]
    fn rejects_inverted_data_offsets() {
        // start > end used to pass the bounds check and panic on slicing
        let header = serde_json::json!({"t": {"dtype": "F32", "shape": [1], "data_offsets": [8, 4]}});
        let raw = file_with_header(&header, &[0.0f32.to_le_bytes()].concat());
        let err = Safetensors::from_bytes(&raw).unwrap_err();
        assert!(err.contains("t"), "{err}");
    }

    #[test]
    fn rejects_offsets_past_eof_and_bad_json() {
        let header = serde_json::json!({"t": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]}});
        let raw = file_with_header(&header, &[0u8; 8]);
        assert!(Safetensors::from_bytes(&raw).unwrap_err().contains("out of bounds"));

        let mut raw = 100u64.to_le_bytes().to_vec();
        raw.extend_from_slice(b"{not json");
        raw.resize(8 + 100, 0);
        assert!(Safetensors::from_bytes(&raw).unwrap_err().contains("header json"));

        assert!(Safetensors::from_bytes(&[1, 2, 3]).unwrap_err().contains("truncated"));
    }

    #[test]
    fn parses_wellformed_file() {
        let header = serde_json::json!({
            "__metadata__": {"scale": "0.25"},
            "t": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
        });
        let data: Vec<u8> = [1.5f32.to_le_bytes(), (-2.5f32).to_le_bytes()].concat();
        let raw = file_with_header(&header, &data);
        let st = Safetensors::from_bytes(&raw).unwrap();
        assert_eq!(st.get("t").unwrap(), [1.5, -2.5]);
        assert_eq!(st.shape("t").unwrap(), [2]);
        assert_eq!(st.metadata["scale"], "0.25");
    }
}
