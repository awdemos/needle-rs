"""Build a tiny random Needle 3 model, export it to .cact, and dump reference
logits + hidden cells from the JAX implementation for the Rust port to match."""
import json
import sys

import numpy as np

sys.path.insert(0, "/var/home/a/code/needle")

import jax
import jax.numpy as jnp
from needle.model.architecture import SimpleAttentionNetwork, TransformerConfig
from needle.model.export import write_export, read_export

import os

DEFAULT_CFG = dict(
    vocab_size=256,
    d_model=64,
    num_heads=4,
    num_kv_heads=2,
    num_layers=4,
    qk_head_dim=16,
    v_head_dim=16,
    max_seq_len=64,
    sliding_window=8,
    global_layers=(3,),
    qkv_conv_taps=3,
    engram_orders=(2, 3),
    engram_slots=512,
    engram_layers=(1, 3),
    mhc_lanes=4,
    rope_theta=10000.0,
    dtype="float32",
    flash=False,
    remat=False,
)
CFG = json.loads(os.environ["NEEDLE_TINY_CFG"]) if "NEEDLE_TINY_CFG" in os.environ else DEFAULT_CFG
PREFIX = os.environ.get("NEEDLE_TINY_PREFIX", "tiny")


def main():
    cfg = TransformerConfig(**CFG)
    model = SimpleAttentionNetwork(cfg)
    key = jax.random.PRNGKey(0)
    tokens = np.array([2, 10, 11, 12, 13, 5, 100, 7, 8, 42], dtype=np.int32)[None, :]
    params = model.init(key, jnp.asarray(tokens))["params"]

    logits = np.asarray(model.apply({"params": params}, jnp.asarray(tokens)))[0]
    cells = np.asarray(
        model.apply({"params": params}, jnp.asarray(tokens), method=model.hidden_cells)
    )[0]  # (T, L+1, d)

    # Reference B: simulate the archive exactly — quantize then dequantize every
    # CQ-eligible leaf (kernel/embedding/mhc_phi, ndim>=2) at bits=4 group=8, like
    # write_export does; FP16 rounding of small tensors is ~5e-4, within tolerance.
    from needle.model.quantize import cq_quantize

    def qleaf(path, leaf):
        key = path[-1]
        k = key.key if hasattr(key, "key") else str(key)
        if (k in ("kernel", "embedding") or k.startswith("mhc_phi")) and getattr(leaf, "ndim", 0) >= 2:
            return np.asarray(cq_quantize(jnp.asarray(np.asarray(leaf, np.float32)), 4, 8))
        return np.asarray(leaf)

    import jax.tree_util as jtu

    flat = jtu.tree_map_with_path(qleaf, params)
    logits_q = np.asarray(model.apply({"params": flat}, jnp.asarray(tokens)))[0]
    cells_q = np.asarray(
        model.apply({"params": flat}, jnp.asarray(tokens), method=model.hidden_cells)
    )[0]

    info = write_export(params, cfg, f"/var/home/a/code/needle-rs/models/{PREFIX}.cact", bits=4, group=8)
    print("wrote", info)

    # Reference C: reconstruct the param tree from the archive exactly (f16 norms,
    # FP16 smalls) — this is what the Rust runtime sees. The tight tolerance test.
    geo, ts = read_export(f"/var/home/a/code/needle-rs/models/{PREFIX}.cact")
    L, n, d = geo["num_layers"], geo["mhc_lanes"], geo["d_model"]
    pos = 0
    emb = ts[pos]; pos += 1
    names = ["norm_in", "q_proj", "k_proj", "v_proj", "q_taps", "k_taps", "v_taps",
             "q_norm", "k_norm", "gate_proj", "out_proj", "post_norm", "attn_gate",
             "pre_hada", "d1", "d2", "b2", "d3", "d4", "w1a", "w1b", "w2a", "w2b",
             "w3a", "w3b", "cond_v", "cond_u"]
    layers = {nm: [] for nm in names}
    for _ in range(L):
        for nm in names:
            layers[nm].append(ts[pos]); pos += 1
    mhc_names = ("mhc_a_pre", "mhc_a_post", "mhc_a_res", "mhc_b_pre", "mhc_b_post", "mhc_b_res",
                 "mhc_phi_pre", "mhc_phi_post", "mhc_phi_res")
    mhc = {}
    for nm in mhc_names:
        mhc[nm] = ts[pos]; pos += 1
    pos += 2  # hada perms are not params
    sites = []
    for _ in geo["engram_layers"]:
        tables = ts[pos]; kp = ts[pos + 1]; vp = ts[pos + 2]; taps = ts[pos + 3]
        pos += 4
        sites.append((tables, kp, vp, taps))
    final_norm = ts[pos]; pos += 1
    assert pos == len(ts) or (pos == len(ts) - 1 and isinstance(ts[-1], (bytes, bytearray))), (pos, len(ts))

    def stack0(nm):
        return np.stack(layers[nm], axis=0)

    phi_pre = mhc["mhc_phi_pre"].reshape(L, n, n * d).transpose(0, 2, 1)
    phi_post = mhc["mhc_phi_post"].reshape(L, n, n * d).transpose(0, 2, 1)
    phi_res = mhc["mhc_phi_res"].reshape(L, n * n, n * d).transpose(0, 2, 1)
    tree = {
        "embedding": {"embedding": jnp.asarray(emb)},
        "stack": {
            "layers": {"block": {
                "ZCRMSNorm_0": {"scale": jnp.asarray(stack0("norm_in"))},
                "self_attn": {
                    "q_proj": {"kernel": jnp.asarray(stack0("q_proj").transpose(0, 2, 1))},
                    "k_proj": {"kernel": jnp.asarray(stack0("k_proj").transpose(0, 2, 1))},
                    "v_proj": {"kernel": jnp.asarray(stack0("v_proj").transpose(0, 2, 1))},
                    "q_taps": jnp.asarray(stack0("q_taps")),
                    "k_taps": jnp.asarray(stack0("k_taps")),
                    "v_taps": jnp.asarray(stack0("v_taps")),
                    "q_norm": {"scale": jnp.asarray(stack0("q_norm"))},
                    "k_norm": {"scale": jnp.asarray(stack0("k_norm"))},
                    "gate_proj": {"kernel": jnp.asarray(stack0("gate_proj").transpose(0, 2, 1))},
                    "out_proj": {"kernel": jnp.asarray(stack0("out_proj").transpose(0, 2, 1))},
                },
                "post_attn_norm": {"scale": jnp.asarray(stack0("post_norm"))},
                "attn_gate": jnp.asarray(stack0("attn_gate").reshape(L)),
                "pre_hada_norm": {"scale": jnp.asarray(stack0("pre_hada"))},
                "hadamard_mlp": {nm: jnp.asarray(stack0(nm)) for nm in
                                 ("d1", "d2", "b2", "d3", "d4", "w1a", "w1b", "w2a", "w2b",
                                  "w3a", "w3b", "cond_v", "cond_u")},
            }},
            "mhc_phi_pre": jnp.asarray(phi_pre),
            "mhc_phi_post": jnp.asarray(phi_post),
            "mhc_phi_res": jnp.asarray(phi_res),
            "mhc_b_pre": jnp.asarray(mhc["mhc_b_pre"]),
            "mhc_b_post": jnp.asarray(mhc["mhc_b_post"]),
            "mhc_b_res": jnp.asarray(mhc["mhc_b_res"]),
            "mhc_a_pre": jnp.asarray(mhc["mhc_a_pre"]),
            "mhc_a_post": jnp.asarray(mhc["mhc_a_post"]),
            "mhc_a_res": jnp.asarray(mhc["mhc_a_res"]),
            "final_norm": {"scale": jnp.asarray(final_norm)},
        },
    }
    orders = geo["engram_orders"]
    heads_ = max(1, d // (len(orders) * 128))
    sub = d // (len(orders) * heads_)
    for si, (tables, kp, vp, taps) in enumerate(sites):
        tree[f"engrams_{si}"] = {
            "embedding": jnp.asarray(tables.reshape(len(orders) * heads_, geo["engram_slots"], sub)),
            "key_proj": {"kernel": jnp.asarray(kp.T)},
            "value_proj": {"kernel": jnp.asarray(vp.T)},
            "taps": jnp.asarray(taps),
        }
    logits_a = np.asarray(model.apply({"params": tree}, jnp.asarray(tokens)))[0]
    cells_a = np.asarray(
        model.apply({"params": tree}, jnp.asarray(tokens), method=model.hidden_cells)
    )[0]
    print("archive-vs-tree max |cells diff|:", float(np.abs(cells_a - cells_q).max()))
    print("archive-vs-tree max |logits diff|:", float(np.abs(logits_a - logits_q).max()))

    np.savez(
        f"/var/home/a/code/needle-rs/models/{PREFIX}_oracle.npz",
        tokens=tokens[0],
        logits=logits,
        cells=cells,
        logits_q=logits_q,
        cells_q=cells_q,
        allow_pickle=False,
    )
    print("tokens:", tokens[0].tolist())
    print("logits shape:", logits.shape, "cells:", cells.shape)
    print("logits[0,:6]:", logits[0, :6])
    print("max |logits - logits_q|:", float(np.abs(logits - logits_q).max()))
    # JSON dump for the Rust test
    with open(f"/var/home/a/code/needle-rs/models/{PREFIX}_oracle.json", "w") as fh:
        json.dump(
            {
                "tokens": tokens[0].tolist(),
                "logits_q": np.asarray(logits_q, dtype=np.float64).tolist(),
                "cells_q": np.asarray(cells_q, dtype=np.float64).tolist(),
                "logits_a": np.asarray(logits_a, dtype=np.float64).tolist(),
                "cells_a": np.asarray(cells_a, dtype=np.float64).tolist(),
            },
            fh,
        )
    print("json written")


if __name__ == "__main__":
    main()
