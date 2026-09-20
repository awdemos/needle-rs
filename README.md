# needle-rs

A Rust port of [Needle 3](https://github.com/awdemos/needle) (Cactus Compute) — the
on-device foundation model for tool calling, device control, and structured
extraction. This workspace is a from-scratch, pure-Rust runtime: it loads the same
`.cact` weight archives, runs the same SAN forward pass, compiles your tool
schemas into a byte-level decode grammar, and returns the same response envelope
as the Python package — no native engine, no JAX, no SentencePiece.

> **Disclaimer:** needle-rs is a community project. It is not affiliated with,
> endorsed by, or supported by Cactus Compute (the developers of Needle), and
> it is not an official Rust implementation of Needle. Trademarks and model
> weights belong to their respective owners; the weights are downloaded from
> the upstream project's published channels.

```
crates/
  needle-format    .cact archive reader (header, CQ 1/2/3/4-bit + ternary, tokenizer blob)
  needle-tokenizer the embedded SentencePiece BPE reference tokenizer (RefTokenizer port)
  needle-model     the SAN: engram memory, MHC lanes, GQA + conv taps, Hadamard MLP,
                   probe heads; incremental KV-cached forward pass
  needle-engine    prompt rendering, grammar-constrained tool-call decode, confidence
                   gating, response envelope
  needle-agent     the `Needle` agent API: complete / run / extract / embed, grounding
  needle-build     checkpoint -> .cact export: safetensors reader, LoRA merge,
                   ladder rung slicing, CQ quantization
  needle-cli       the `needle-rs` binary
examples/
  smart_home       the smart_home acceptance suite (port of needle/environments)
```

## Quick start

```sh
cargo build --release
./target/release/needle-rs download --out models          # fetch needle3.cact (~35 MB)
./target/release/needle-rs run --model models/needle3.cact \
    --query "turn on the living room lights" --tools tools.json
```

with `tools.json` = `[{"name":"set_lights","description":"Turn a room's lights on/off ...","parameters":{...}}]`
(the same JSON Schema shape the Python package consumes).

Library use:

```rust
let mut agent = needle_agent::Needle::new(
    std::path::Path::new("models/needle3.cact"),
    tools,               // Vec<serde_json::Value> raw JSON Schemas
    None,                // optional system facts; a `date:` fact is auto-prefixed
)?;
let response = agent.complete("turn on the living room lights")?;
println!("{}", serde_json::to_string_pretty(&response.to_json())?);
```

Every turn returns one envelope:

```json
{
  "type": "call", "success": true, "error": null, "error_code": null,
  "function_calls": [{"name": "set_lights", "arguments": {"room": "living room", "on": true}}],
  "suppressed_calls": [], "reasoning": "...", "confidence": 0.72,
  "prefill_tps": 48.0, "decode_tps": 12.0
}
```

Off-topic input returns an empty `function_calls` (a refusal); calls the engine
is unsure about (confidence below 0.1) move to `suppressed_calls` — show them to
the user to confirm.

## CLI

```
needle-rs run --model m.cact --query "..." --tools tools.json [--system "..."] [--max-tokens N]
needle-rs info --model m.cact          # header geometry
needle-rs tokens --model m.cact --text "..."   # tokenizer debug
needle-rs embed --model m.cact --text "..."    # sentence embedding (needs an embedding head)
needle-rs download --out models        # fetch the base needle3.cact from Hugging Face
needle-rs build ckpt.safetensors [--lora adapter.safetensors] [--layers N] \
    --out tuned.cact --tokenizer-from models/needle3.cact
```

## Build / export your own archive

```sh
# from a checkpoint (LoRA-merged, any ladder rung 2..20), tokenizer copied from the base:
needle-rs build checkpoints/needle3.safetensors --layers 8 --out needle3-8l.cact \
    --tokenizer-from models/needle3.cact
```

The exporter quantizes weights with Cactus Quants (CQ, W4A8 by default) at group
128; the Hadamard perms reproduce `np.random.RandomState(11/13).permutation` exactly.

## Verification status (what's exact vs approximated)

Exact (differential-tested against the Python JAX reference on the real weights
and on random models of five geometries):

- `.cact` parsing and CQ dequantization (2/3/4-bit, ternary) — matches
  `needle.model.export.read_export` to 5e-5 on every tensor of the published archive.
- The SAN forward pass — logits and per-layer cells match JAX to 0.000000 on
  archive-reconstructed weights; greedy decode of a tool-call prompt is
  byte-identical to the JAX reference.
- The embedded tokenizer — ids identical to `RefTokenizer` on the real archive.
- Tool-call selection and argument grounding — same calls as the native engine in
  end-to-end tests (`crates/needle-engine/tests/e2e.rs`).

Approximated (the closed-source engine's internals, documented choices):

- **Confidence**: `min(sigmoid(confidence_head), mean token probability of the
  tool-call region)`; the native engine reports higher scores (e.g. 1.0 for clean
  calls). The suppression floor (0.1) and the gating semantics match.
- **Grammar**: guarantees well-formed, schema-shaped calls (types, enums,
  required keys). Numeric ranges/string patterns are not enforced at decode time.
- **Multi-turn result feedback**: the engine's exact wire format is not public;
  results are fed back in a `<tool_result>` user turn.
- The published base archive ships only the confidence probe head; `embed()`
  returns `None` unless a tuned archive carries the embedding head.

Out of scope for v1: LoRA fine-tuning / data generation (the JAX training stack).
Weights are consumed and exported, not trained.

## Tests

`cargo test --workspace --release` — includes real-weight end-to-end tests that
skip gracefully if `models/needle3.cact` is absent (fetch it with
`needle-rs download --out models`). The JAX differential fixtures under `models/`
are generated by `scripts/make_oracle.py`.

## License

Apache-2.0 (same as the upstream Needle project).
