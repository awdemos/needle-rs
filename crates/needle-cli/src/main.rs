//! `needle-rs` — the Needle 3 CLI (Rust port).

mod sha256;

use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};

/// Pinned integrity of the published base archive
/// (<https://huggingface.co/Cactus-Compute/needle3/resolve/main/needle3.cact>),
/// verified by `download` before the archive is published under its final
/// name. Computed 2026-09-19 from the known-good copy in this repo
/// (`sha256sum models/needle3.cact`, 35,335,380 bytes).
const EXPECTED_SHA256: &str = "c9d915eca282ed42d1a09b143b592adb4cc6744ffe2d294adf5cfc5548170c38";
const EXPECTED_BYTES: u64 = 35_335_380;

#[derive(Parser)]
#[command(name = "needle-rs", about = "Needle 3 runtime (Rust port): on-device tool-calling model")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run a query against a .cact archive and print the JSON envelope.
    Run {
        /// Path to the model archive (needle3.cact or a tuned .cact).
        #[arg(long)]
        model: PathBuf,
        /// The user query.
        #[arg(long)]
        query: Option<String>,
        /// Tools JSON file (a JSON array of schemas).
        #[arg(long)]
        tools: Option<PathBuf>,
        /// System facts string.
        #[arg(long)]
        system: Option<String>,
        #[arg(long, default_value_t = 512)]
        max_tokens: usize,
    },
    /// Print the archive header geometry.
    Info {
        #[arg(long)]
        model: PathBuf,
    },
    /// Tokenize text and print the ids.
    Tokens {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        text: String,
    },
    /// Sentence embedding (requires an embedding head in the archive).
    Embed {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        text: String,
    },
    /// Download the base needle3.cact from Hugging Face.
    Download {
        /// Output directory.
        #[arg(long, default_value = ".")]
        out: PathBuf,
    },
    /// Export a checkpoint (+ optional LoRA adapter, optional ladder rung)
    /// to a .cact archive.
    Build {
        /// Base checkpoint (.safetensors).
        checkpoint: PathBuf,
        /// Optional LoRA adapter (.safetensors) to merge before export.
        #[arg(long)]
        lora: Option<PathBuf>,
        /// Export the N-layer rung (2..layers).
        #[arg(long)]
        layers: Option<usize>,
        /// Output .cact path.
        #[arg(long)]
        out: PathBuf,
        /// Archive to copy the embedded tokenizer from (default: needle3.cact
        /// next to the checkpoint, then NEEDLE_MODELS_DIR/needle3.cact).
        #[arg(long)]
        tokenizer_from: Option<PathBuf>,
        /// CQ width in bits (1..=4; 5 is reserved for ternary records).
        #[arg(long, default_value_t = 4, value_parser = clap::value_parser!(u32).range(1..=4))]
        bits: u32,
    },
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

/// The archive's own format tag (the reader validates it but does not
/// expose which generation was matched).
fn archive_tag(path: &Path) -> Result<u32, String> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut b = [0u8; 4];
    f.read_exact(&mut b)
        .map_err(|e| format!("read {}: not a complete .cact archive ({e})", path.display()))?;
    Ok(u32::from_le_bytes(b))
}

/// The tokenizer blob to embed: the explicit `--tokenizer-from` archive if
/// given, else needle3.cact next to the checkpoint, then
/// NEEDLE_MODELS_DIR/needle3.cact. A build without a tokenizer cannot be
/// loaded, so every miss is an error rather than a silent tokenizer-less
/// archive.
fn resolve_tokenizer(explicit: Option<&Path>, checkpoint: &Path) -> Result<Vec<u8>, String> {
    let blob = |p: &Path| -> Option<Vec<u8>> {
        needle_format::read_archive(p)
            .ok()
            .and_then(|ar| ar.tokenizer_blob().map(|b| b.to_vec()))
            .filter(|b| !b.is_empty())
    };
    if let Some(p) = explicit {
        return blob(p).ok_or_else(|| {
            format!(
                "--tokenizer-from {} is not a readable archive with an embedded tokenizer",
                p.display()
            )
        });
    }
    let mut probed: Vec<PathBuf> = vec![checkpoint.with_file_name("needle3.cact")];
    if let Ok(dir) = std::env::var("NEEDLE_MODELS_DIR") {
        probed.push(PathBuf::from(dir).join("needle3.cact"));
    }
    for p in &probed {
        if let Some(b) = blob(p) {
            println!("tokenizer:      embedded from {}", p.display());
            return Ok(b);
        }
    }
    Err(format!(
        "no tokenizer source: pass --tokenizer-from <archive.cact> (probed {})",
        probed.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(", ")
    ))
}

fn run() -> Result<(), String> {
    let cli = Cli::parse();
    match cli.command {
        Command::Info { model } => {
            let tag = archive_tag(&model)?;
            let ar = needle_format::read_archive(&model).map_err(|e| e.to_string())?;
            let c = &ar.config;
            println!("tag:            0x{tag:08x}");
            println!("tensors:        {}", ar.tensors.len());
            println!("vocab:          {} (out {})", c.vocab_size, c.out_vocab);
            println!("d_model:        {}", c.d_model);
            println!("heads:          {} (kv {})", c.num_heads, c.num_kv_heads);
            println!("head dims:      qk {} v {}", c.qk_head_dim, c.v_head_dim);
            println!("layers:         {}", c.num_layers);
            println!("mhc lanes:      {}", c.mhc_lanes);
            println!("max_seq_len:    {}", c.max_seq_len);
            println!("sliding_window: {}", c.sliding_window);
            println!("global layers:  {:?}", c.global_layers);
            println!("qkv taps:       {}", c.qkv_conv_taps);
            println!("engram:         orders {:?} tables {} slots {} sub {} dil {}",
                c.engram_orders, c.num_engram_tables, c.engram_slots, c.engram_sub_dim, c.engram_conv_dilation);
            println!("engram sites:   {:?}", c.engram_layers);
            println!("rope theta:     {}", c.rope_theta);
            println!("kv:             window {} bits {}", c.kv_window, c.kv_bits);
            println!("hada_n:         {}", c.hada_n);
            println!("tokenizer:      {}", if ar.tokenizer_blob().is_some() { "embedded" } else { "none" });
        }
        Command::Tokens { model, text } => {
            let ar = needle_format::read_archive(&model).map_err(|e| e.to_string())?;
            let tok = needle_tokenizer::Tokenizer::from_blob(ar.tokenizer_blob().ok_or("no tokenizer in archive")?)
                .map_err(|e| e.to_string())?;
            let ids = tok.encode(&text);
            println!("n = {}", ids.len());
            println!("ids = {ids:?}");
            println!("pieces = {:?}", ids.iter().map(|&i| tok.piece(i)).collect::<Vec<_>>());
            println!("roundtrip = {:?}", tok.decode(&ids));
        }
        Command::Run {
            model,
            query,
            tools,
            system,
            max_tokens,
        } => {
            let query = query.unwrap_or_else(|| {
                use std::io::Read;
                let mut s = String::new();
                std::io::stdin().read_to_string(&mut s).ok();
                s.trim().to_string()
            });
            let tool_values: Vec<serde_json::Value> = match tools {
                Some(p) => serde_json::from_str(
                    &std::fs::read_to_string(&p).map_err(|e| format!("read {}: {e}", p.display()))?,
                )
                .map_err(|e| format!("tools JSON: {e}"))?,
                None => Vec::new(),
            };
            let mut agent = needle_agent::Needle::new(&model, tool_values, system)?;
            let mut exec = |name: &str, args: &serde_json::Value| {
                Ok(serde_json::json!({"executed": name, "arguments": args, "note": "no handler bound in CLI"}))
            };
            let response = agent.run_limited(&query, 8, max_tokens, true, &mut exec)?;
            println!("{}", serde_json::to_string_pretty(&response.to_json()).unwrap());
        }
        Command::Embed { model, text } => {
            let ar = needle_format::read_archive(&model).map_err(|e| e.to_string())?;
            let m = needle_model::Model::from_archive(&ar).map_err(|e| e.to_string())?;
            let tok = needle_tokenizer::Tokenizer::from_blob(ar.tokenizer_blob().ok_or("no tokenizer in archive")?)
                .map_err(|e| e.to_string())?;
            let engine = needle_engine::Engine::new(&m, tok, Vec::new());
            let mut ids = vec![needle_tokenizer::BOS_ID];
            ids.extend(engine.tokenizer.encode(&text));
            match engine.embed(&ids) {
                Some(v) => println!("{}", serde_json::to_string(&v).unwrap()),
                None => return Err("this archive carries no embedding head".into()),
            }
        }
        Command::Build {
            checkpoint,
            lora,
            layers,
            out,
            tokenizer_from,
            bits,
        } => {
            let mut ckpt = needle_build::Checkpoint::load(&checkpoint)?;
            if let Some(adapter_path) = lora {
                let adapter = needle_build::Safetensors::read(&adapter_path)?;
                // Python requires `scale` in the adapter metadata; merging
                // with a made-up default silently mis-scales the update
                let scale: f32 = adapter
                    .metadata
                    .get("scale")
                    .ok_or_else(|| "adapter metadata lacks `scale`".to_string())
                    .and_then(|s| {
                        s.parse()
                            .map_err(|_| format!("adapter metadata `scale` is not a number: {s:?}"))
                    })?;
                let n = ckpt.merge_lora(&adapter, scale)?;
                println!("merged {n} LoRA groups (scale {scale})");
            }
            if let Some(depth) = layers {
                ckpt.ladder_slice(depth)?;
                println!("depth: {} layers", ckpt.config.num_layers);
            }
            let blob = resolve_tokenizer(tokenizer_from.as_deref(), &checkpoint)?;
            ckpt.write_cact(&out, Some(&blob), bits, 128)?;
            println!(
                "wrote {} ({} bytes, {} layers, W{bits}A8)",
                out.display(),
                std::fs::metadata(&out).map_err(|e| format!("stat {}: {e}", out.display()))?.len(),
                ckpt.config.num_layers
            );
        }
        Command::Download { out } => {
            let url = "https://huggingface.co/Cactus-Compute/needle3/resolve/main/needle3.cact";
            std::fs::create_dir_all(&out).map_err(|e| format!("mkdir {}: {e}", out.display()))?;
            let dest = out.join("needle3.cact");
            // stream to a .part sibling and only publish after size + digest
            // verify, so an interrupted or corrupted download never sits at
            // the final name looking like a valid archive
            let part = out.join("needle3.cact.part");
            println!("fetching {url}");
            let resp = ureq::get(url).call().map_err(|e| format!("download: {e}"))?;
            let mut file = std::fs::File::create(&part).map_err(|e| format!("create {}: {e}", part.display()))?;
            std::io::copy(&mut resp.into_reader(), &mut file).map_err(|e| format!("write {}: {e}", part.display()))?;
            drop(file);
            let bytes = std::fs::read(&part).map_err(|e| format!("read back {}: {e}", part.display()))?;
            let bad = |why: String| {
                let _ = std::fs::remove_file(&part);
                why
            };
            if bytes.len() as u64 != EXPECTED_BYTES {
                return Err(bad(format!(
                    "downloaded archive is {} bytes, expected {EXPECTED_BYTES} — refusing to publish a truncated archive",
                    bytes.len()
                )));
            }
            let digest = sha256::sha256_hex(&bytes);
            if digest != EXPECTED_SHA256 {
                return Err(bad(format!(
                    "downloaded archive sha256 {digest} does not match the pinned digest {EXPECTED_SHA256}"
                )));
            }
            std::fs::rename(&part, &dest)
                .map_err(|e| format!("rename {} -> {}: {e}", part.display(), dest.display()))?;
            println!("wrote {} ({} bytes, sha256 verified)", dest.display(), bytes.len());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// NEEDLE_MODELS_DIR is process-global; tests touching it must not race
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// a tiny well-formed archive with a 3-byte RAW "tokenizer" attachment
    fn fake_archive(path: &Path) {
        let mut raw = Vec::new();
        raw.extend_from_slice(&needle_format::TAG_V3.to_le_bytes());
        raw.extend_from_slice(&1u32.to_le_bytes()); // num_tensors
        raw.extend_from_slice(&0u32.to_le_bytes()); // codebook len
        raw.extend_from_slice(&[0u8; 196 - 12 - 4]); // remaining geometry + rope slot
        raw.extend_from_slice(&0.0f32.to_le_bytes());
        // one RAW record: ndim 0, offset 240, 3 bytes
        raw.push(needle_format::DTYPE_RAW);
        raw.push(0);
        raw.extend_from_slice(&[0u8; 2]);
        raw.extend_from_slice(&[0u8; 16]);
        raw.extend_from_slice(&240u64.to_le_bytes());
        raw.extend_from_slice(&3u64.to_le_bytes());
        raw.extend_from_slice(&0u32.to_le_bytes());
        raw.extend_from_slice(&0u32.to_le_bytes());
        assert_eq!(raw.len(), 240);
        raw.extend_from_slice(b"tok");
        std::fs::write(path, raw).unwrap();
    }

    fn temp_dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("needle-cli-test-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn archive_tag_reads_the_file_header() {
        let d = temp_dir("tag");
        let p = d.join("x.cact");
        fake_archive(&p);
        assert_eq!(archive_tag(&p).unwrap(), needle_format::TAG_V3);
        assert!(archive_tag(&d.join("missing.cact")).is_err());
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn tokenizer_resolution_probes_the_documented_fallbacks() {
        let _g = ENV_LOCK.lock().unwrap();
        let d = temp_dir("tok");
        let base = d.join("needle3.cact");
        fake_archive(&base);
        let checkpoint = d.join("needle3.safetensors");

        // explicit source wins, even though the fallback also exists
        let other = d.join("other.cact");
        fake_archive(&other);
        assert_eq!(resolve_tokenizer(Some(&other), &checkpoint).unwrap(), b"tok");
        // fallback: needle3.cact next to the checkpoint
        assert_eq!(resolve_tokenizer(None, &checkpoint).unwrap(), b"tok");
        // fallback: NEEDLE_MODELS_DIR
        std::fs::remove_file(&base).unwrap();
        let models = d.join("models-env");
        std::fs::create_dir_all(&models).unwrap();
        fake_archive(&models.join("needle3.cact"));
        std::env::set_var("NEEDLE_MODELS_DIR", &models);
        assert_eq!(resolve_tokenizer(None, &checkpoint).unwrap(), b"tok");
        // nothing left to probe -> hard error
        std::fs::remove_file(models.join("needle3.cact")).unwrap();
        assert!(resolve_tokenizer(None, &checkpoint).is_err());
        std::env::remove_var("NEEDLE_MODELS_DIR");
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn pinned_digest_matches_the_repo_copy() {
        let _g = ENV_LOCK.lock().unwrap();
        // the pinned EXPECTED_SHA256 must describe models/needle3.cact when present
        let models = std::env::var("NEEDLE_MODELS_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models"));
        let path = models.join("needle3.cact");
        if !path.exists() {
            eprintln!("skip: {} not found", path.display());
            return;
        }
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(bytes.len() as u64, EXPECTED_BYTES, "size pin");
        assert_eq!(sha256::sha256_hex(&bytes), EXPECTED_SHA256, "digest pin");
    }
}
