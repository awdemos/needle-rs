//! `needle-rs` — the Needle 3 CLI (Rust port).

use clap::{Parser, Subcommand};
use std::path::PathBuf;

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
        /// Archive to copy the embedded tokenizer from (default: the base
        /// needle3.cact next to the checkpoint, else none).
        #[arg(long)]
        tokenizer_from: Option<PathBuf>,
        /// CQ width in bits (default 4).
        #[arg(long, default_value_t = 4)]
        bits: u32,
    },
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Command::Info { model } => {
            let ar = needle_format::read_archive(&model).expect("read archive");
            let c = &ar.config;
            println!("tag:            0x{:08x}", needle_format::TAG_V3);
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
            let ar = needle_format::read_archive(&model).expect("read archive");
            let tok = needle_tokenizer::Tokenizer::from_blob(ar.tokenizer_blob().expect("tokenizer"))
                .expect("tokenizer");
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
                    &std::fs::read_to_string(&p).expect("read tools file"),
                )
                .expect("tools JSON"),
                None => Vec::new(),
            };
            let mut agent = needle_agent::Needle::new(&model, tool_values, system).expect("agent");
            let mut exec = |name: &str, args: &serde_json::Value| {
                Ok(serde_json::json!({"executed": name, "arguments": args, "note": "no handler bound in CLI"}))
            };
            let response = agent.run_limited(&query, 8, max_tokens, true, &mut exec).expect("run");
            println!("{}", serde_json::to_string_pretty(&response.to_json()).unwrap());
        }
        Command::Embed { model, text } => {
            let ar = needle_format::read_archive(&model).expect("read archive");
            let m = needle_model::Model::from_archive(&ar).expect("model");
            let tok = needle_tokenizer::Tokenizer::from_blob(ar.tokenizer_blob().unwrap()).unwrap();
            let engine = needle_engine::Engine::new(&m, tok, Vec::new());
            let mut ids = vec![needle_tokenizer::BOS_ID];
            ids.extend(engine.tokenizer.encode(&text));
            match engine.embed(&ids) {
                Some(v) => println!("{}", serde_json::to_string(&v).unwrap()),
                None => {
                    eprintln!("this archive carries no embedding head");
                    std::process::exit(1);
                }
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
            let mut ckpt = needle_build::Checkpoint::load(&checkpoint).expect("checkpoint");
            if let Some(adapter_path) = lora {
                let adapter = needle_build::Safetensors::read(&adapter_path).expect("adapter");
                let scale: f32 = adapter
                    .metadata
                    .get("scale")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(1.0);
                let n = ckpt.merge_lora(&adapter, scale).expect("merge lora");
                println!("merged {n} LoRA groups (scale {scale})");
            }
            if let Some(depth) = layers {
                ckpt.ladder_slice(depth).expect("ladder slice");
                println!("depth: {} layers", ckpt.config.num_layers);
            }
            let blob = tokenizer_from
                .as_ref()
                .and_then(|p| needle_format::read_archive(p).ok())
                .and_then(|ar| ar.tokenizer_blob().map(|b| b.to_vec()));
            ckpt.write_cact(&out, blob.as_deref(), bits, 128).expect("export");
            println!(
                "wrote {} ({} bytes, {} layers, W{bits}A8)",
                out.display(),
                std::fs::metadata(&out).unwrap().len(),
                ckpt.config.num_layers
            );
        }
        Command::Download { out } => {
            let url = "https://huggingface.co/Cactus-Compute/needle3/resolve/main/needle3.cact";
            std::fs::create_dir_all(&out).expect("mkdir");
            let dest = out.join("needle3.cact");
            println!("fetching {url}");
            let resp = ureq::get(url).call().expect("download");
            let mut file = std::fs::File::create(&dest).expect("create file");
            std::io::copy(&mut resp.into_reader(), &mut file).expect("write");
            println!("wrote {} ({} bytes)", dest.display(), file.metadata().unwrap().len());
        }
    }
}
