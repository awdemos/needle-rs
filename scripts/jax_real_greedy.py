"""Greedy-decode a tool-call prompt in JAX from the REAL needle3.cact archive
(reconstructed tree) — the reference the Rust port must match."""
import sys

import numpy as np

sys.path.insert(0, "/var/home/a/code/needle")

import jax
import jax.numpy as jnp
from needle.model.architecture import SimpleAttentionNetwork, TransformerConfig
from needle.model.export import read_export, RefTokenizer


def tree_from_archive(path):
    geo, ts = read_export(path)
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
    mhc = {}
    for nm in ("mhc_a_pre", "mhc_a_post", "mhc_a_res", "mhc_b_pre", "mhc_b_post",
               "mhc_b_res", "mhc_phi_pre", "mhc_phi_post", "mhc_phi_res"):
        mhc[nm] = ts[pos]; pos += 1
    pos += 2
    sites = []
    for _ in geo["engram_layers"]:
        sites.append((ts[pos], ts[pos + 1], ts[pos + 2], ts[pos + 3])); pos += 4
    final_norm = ts[pos]; pos += 1
    heads = []
    head_entries = []
    if (pos < len(ts) and isinstance(ts[pos], np.ndarray) and ts[pos].ndim == 1
            and 1 <= ts[pos].size <= 3 and set(np.asarray(ts[pos]).tolist()) <= {1.0, 2.0, 3.0}):
        manifest = ts[pos]; pos += 1
        L1 = geo["num_layers"] + 1
        d = geo["d_model"]
        for code in manifest:
            probes = ts[pos]; gain = ts[pos + 1]; query = ts[pos + 2]
            row_bias = ts[pos + 3]; proj = ts[pos + 4]; bias = ts[pos + 5]
            pos += 6
            cal = None
            if int(code) == 3:
                cal = ts[pos]; pos += 1
            heads.append((int(code), probes, gain, query, row_bias, proj, bias, cal))
            k = probes.shape[0] // L1
            q = query.shape[0]
            out_dim = proj.shape[0]
            name = {1: "embedding_head", 2: "confidence_head", 3: "router_head"}[int(code)]
            head_entries.append((name, {
                "probes": jnp.asarray(probes.reshape(L1, k, d)),
                "gain": jnp.asarray(gain.reshape(L1, k)),
                "query": jnp.asarray(query),
                "row_bias": jnp.asarray(row_bias.reshape(q, L1, k)),
                "proj": {"kernel": jnp.asarray(proj.T), "bias": jnp.asarray(bias)},
            }))
            if cal is not None:
                head_entries[-1][1]["calibration"] = jnp.asarray(cal)
    assert pos == len(ts) or pos == len(ts) - 1, (pos, len(ts))

    def stack0(nm):
        return np.stack(layers[nm], axis=0)

    nC = n * d
    phi_pre = mhc["mhc_phi_pre"].reshape(L, n, nC).transpose(0, 2, 1)
    phi_post = mhc["mhc_phi_post"].reshape(L, n, nC).transpose(0, 2, 1)
    phi_res = mhc["mhc_phi_res"].reshape(L, n * n, nC).transpose(0, 2, 1)
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
    heads_n = max(1, d // (len(orders) * 128))
    sub = d // (len(orders) * heads_n)
    for si, (tables, kp, vp, taps) in enumerate(sites):
        tree[f"engrams_{si}"] = {
            "embedding": jnp.asarray(tables.reshape(len(orders) * heads_n, geo["engram_slots"], sub)),
            "key_proj": {"kernel": jnp.asarray(kp.T)},
            "value_proj": {"kernel": jnp.asarray(vp.T)},
            "taps": jnp.asarray(taps),
        }
    for name, entry in head_entries:
        tree[name] = entry
    return geo, tree


def main():
    geo, tree = tree_from_archive("/var/home/a/code/needle-rs/models/needle3.cact")
    cfg = TransformerConfig(**{k: (tuple(v) if isinstance(v, list) else v)
                               for k, v in geo.items() if k != "num_tensors"})
    cfg.flash = False
    cfg.remat = False
    cfg.dtype = "float32"
    model = SimpleAttentionNetwork(cfg)

    tok = RefTokenizer.from_cact("/var/home/a/code/needle-rs/models/needle3.cact")
    tools = '[{"name":"set_lights","description":"Turn a room\'s lights on/off and set brightness","parameters":{"type":"object","properties":{"room":{"type":"string"},"on":{"type":"boolean"},"brightness":{"type":"integer","minimum":0,"maximum":100}},"required":["room","on"]}}]'
    prompt = f"<|im_start|>user\n<tools>{tools}</tools>\ndim the living room lights to 30<|im_end|>\n<|im_start|>assistant\n"
    ids = [2] + tok.encode(prompt)
    buf = np.asarray([ids], dtype=np.int32)

    @jax.jit
    def step(params, tokens):
        return model.apply({"params": params}, tokens)

    generated = []
    # decode one position at a time, refeeding the growing buffer
    for pos in range(len(ids) - 1, len(ids) + 60):
        logits = step(tree, jnp.asarray(buf))[0, pos]
        nxt = int(np.argmax(np.asarray(logits)))
        if nxt == 1:
            break
        generated.append(nxt)
        buf = np.concatenate([buf, np.asarray([[nxt]], dtype=np.int32)], axis=1)
    print("generated ids:", generated)
    print("decoded:", repr(tok.decode(generated)))




def dump_logits():
    import json
    geo, tree = tree_from_archive("/var/home/a/code/needle-rs/models/needle3.cact")
    cfg = TransformerConfig(**{k: (tuple(v) if isinstance(v, list) else v)
                               for k, v in geo.items() if k != "num_tensors"})
    cfg.flash = False
    cfg.remat = False
    cfg.dtype = "float32"
    model = SimpleAttentionNetwork(cfg)
    tok = RefTokenizer.from_cact("/var/home/a/code/needle-rs/models/needle3.cact")
    tools = '[{"name":"set_lights","description":"Turn a room\'s lights on/off and set brightness","parameters":{"type":"object","properties":{"room":{"type":"string"},"on":{"type":"boolean"},"brightness":{"type":"integer","minimum":0,"maximum":100}},"required":["room","on"]}}]'
    prompt = f"<|im_start|>user\n<tools>{tools}</tools>\ndim the living room lights to 30<|im_end|>\n<|im_start|>assistant\n"
    ids = [2] + tok.encode(prompt)
    buf = np.asarray([ids], dtype=np.int32)

    @jax.jit
    def step(params, tokens):
        return model.apply({"params": params}, tokens)

    outs = {}
    for pos in [0, 1, 2, 10, len(ids) - 1]:
        logits = np.asarray(step(tree, jnp.asarray(buf))[0, pos], np.float64)
        outs[str(pos)] = logits.round(5).tolist()
    cells = np.asarray(
        model.apply({"params": tree}, jnp.asarray(buf), method=model.hidden_cells)
    )[0]  # (T, L+1, d)
    outs["cells0"] = np.asarray(cells[0], np.float64).round(4).tolist()  # (L+1, d) at t0
    outs["cells1"] = np.asarray(cells[1], np.float64).round(4).tolist()
    with open("/var/home/a/code/needle-rs/models/real_logits.json", "w") as fh:
        json.dump(outs, fh)
    print("dumped logits + cells at", [k for k in outs])

if __name__ == "__main__":
    import sys as _s
    if len(_s.argv) > 1 and _s.argv[1] == "dump":
        dump_logits()
    else:
        main()
