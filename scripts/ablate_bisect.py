"""Bisect the layer-0 deviation: 1-layer tiny model, JAX vs an independent numpy
forward (identical to the Rust logic), under param ablations."""
import json
import sys

import numpy as np

sys.path.insert(0, "/var/home/a/code/needle")

import jax
import jax.numpy as jnp
from needle.model.architecture import SimpleAttentionNetwork, TransformerConfig
from needle.model.quantize import cq_quantize
import jax.tree_util as jtu

BASE = dict(
    vocab_size=256, d_model=64, num_heads=4, num_kv_heads=2, num_layers=1,
    qk_head_dim=16, v_head_dim=16, max_seq_len=64, sliding_window=8,
    global_layers=(0,), qkv_conv_taps=3, engram_orders=(2, 3), engram_slots=512,
    engram_layers=(), mhc_lanes=4, rope_theta=10000.0, dtype="float32",
    flash=False, remat=False,
)
EPS = 1e-6


def zcrms(x, scale):
    return (1 + scale) * x / np.sqrt(np.mean(x.astype(np.float32) ** 2) + EPS)


def rms_unit(x):
    xf = x.astype(np.float32)
    return xf * (np.mean(xf ** 2) + EPS) ** -0.5


def logsumexp(x, axis, keepdims):
    m = np.max(x, axis=axis, keepdims=True)
    return m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=keepdims))


def sinkhorn(logits, iters=20):
    k = logits
    for _ in range(iters):
        k = k - logsumexp(k, axis=-1, keepdims=True)
        k = k - logsumexp(k, axis=-2, keepdims=True)
    return np.exp(k)


def numpy_row1(flat, tokens):
    d = 64
    n = 4
    h, kvh, qh, vh = 4, 2, 16, 16
    li = 0
    emb = np.asarray(flat["embedding"]["embedding"], np.float32)
    x0 = emb[tokens[0]] * np.sqrt(d)
    stream = np.tile(x0, (n, 1))
    nx = rms_unit(stream.reshape(-1))
    nC = n * d
    phi_pre = np.asarray(flat["stack"]["mhc_phi_pre"])[li].reshape(n, nC).T  # (nC, n)
    phi_post = np.asarray(flat["stack"]["mhc_phi_post"])[li].reshape(n, nC).T
    phi_res = np.asarray(flat["stack"]["mhc_phi_res"])[li].reshape(n * n, nC).T  # (nC, n*n)
    lane = li % n
    pre_off = 8 * np.eye(n)[lane] - 4
    post_off = -4 * (1 - np.eye(n)[lane])
    a_pre = float(np.asarray(flat["stack"]["mhc_a_pre"])[li])
    a_post = float(np.asarray(flat["stack"]["mhc_a_post"])[li])
    a_res = float(np.asarray(flat["stack"]["mhc_a_res"])[li])
    hpre = 1 / (1 + np.exp(-(a_pre * (nx @ phi_pre) + np.asarray(flat["stack"]["mhc_b_pre"])[li] + pre_off)))
    u = np.einsum("n,nc->c", hpre, stream)

    blk = flat["stack"]["layers"]["block"]
    x = u.copy()
    attn_in = zcrms(x, np.asarray(blk["ZCRMSNorm_0"]["scale"][li], np.float32))
    q = np.asarray(blk["self_attn"]["q_proj"]["kernel"], np.float32)[li].T @ attn_in
    k = np.asarray(blk["self_attn"]["k_proj"]["kernel"], np.float32)[li].T @ attn_in
    v = np.asarray(blk["self_attn"]["v_proj"]["kernel"], np.float32)[li].T @ attn_in
    # conv taps: init [1,0,0]; trained zeros -> identity; handle trained case generally at t0 (identity anyway)
    q = q.reshape(h, qh)
    k = k.reshape(kvh, qh)
    v = v.reshape(kvh, vh)
    for hh in range(h):
        q[hh] = zcrms(q[hh], np.asarray(blk["self_attn"]["q_norm"]["scale"], np.float32)[li])
    for kk in range(kvh):
        k[kk] = zcrms(k[kk], np.asarray(blk["self_attn"]["k_norm"]["scale"], np.float32)[li])
    # rope at t0 = identity
    out = np.zeros((h, vh))
    for hh in range(h):
        kk = hh // (h // kvh)
        score = q[hh] @ k[kk] / np.sqrt(qh)
        w = np.exp(score) / np.exp(score).sum()
        out[hh] = w * v[kk]
    gate = 1 / (1 + np.exp(-(np.asarray(blk["self_attn"]["gate_proj"]["kernel"], np.float32)[li].T @ attn_in)))
    attn_out = out.reshape(-1) * gate
    hattn = np.asarray(blk["self_attn"]["out_proj"]["kernel"], np.float32)[li].T @ attn_out
    x = x + (1 / (1 + np.exp(-float(np.asarray(blk["attn_gate"])[li])))) * zcrms(
        hattn, np.asarray(blk["post_attn_norm"]["scale"], np.float32)[li])

    xin = zcrms(x, np.asarray(blk["pre_hada_norm"]["scale"], np.float32)[li])
    hm = blk["hadamard_mlp"]
    hada_n = 64
    w1a, w1b = np.asarray(hm["w1a"], np.float32)[li], np.asarray(hm["w1b"], np.float32)[li]
    w2a, w2b = np.asarray(hm["w2a"], np.float32)[li], np.asarray(hm["w2b"], np.float32)[li]
    w3a, w3b = np.asarray(hm["w3a"], np.float32)[li], np.asarray(hm["w3b"], np.float32)[li]
    from needle.model.architecture import _hada_perms
    p1, p2 = [np.asarray(p).astype(int) for p in _hada_perms(64, False)]

    def kron(z, a, b):
        m = a.shape[0]
        z = z.reshape(m, m)
        return np.einsum("ij,ik,jl->kl", z, a, b).reshape(-1)

    cv = xin @ np.asarray(hm["cond_v"], np.float32)[li]
    sm = np.exp(cv - cv.max())
    sm /= sm.sum()
    cond = 1 + sm @ np.asarray(hm["cond_u"], np.float32)[li]
    z = xin
    z = kron(np.asarray(hm["d1"], np.float32)[li] * z, w1a, w1b)[p1]
    pre2 = np.asarray(hm["d2"], np.float32)[li] * cond * z + np.asarray(hm["b2"], np.float32)[li]
    z = kron(pre2 / (1 + np.exp(-pre2)), w2a, w2b)[p2]
    z = kron(np.asarray(hm["d3"], np.float32)[li] * z, w3a, w3b)
    hada = (np.asarray(hm["d4"], np.float32)[li] * z)[:d]
    block_out = x + hada
    y = block_out - u

    hpost = 2 / (1 + np.exp(-(a_post * (nx @ phi_post) + np.asarray(flat["stack"]["mhc_b_post"])[li] + post_off)))
    res = (nx @ phi_res).reshape(n, n)
    hres = sinkhorn(a_res * res + np.asarray(flat["stack"]["mhc_b_res"])[li])
    new_stream = hres @ stream + hpost[:, None] * y
    return new_stream.mean(axis=0)


def run(ablate):
    cfg = TransformerConfig(**BASE)
    model = SimpleAttentionNetwork(cfg)
    tokens = np.array([2, 10, 11], dtype=np.int32)
    params = model.init(jax.random.PRNGKey(0), jnp.asarray(tokens[None]))["params"]

    def qleaf(path, leaf):
        key = path[-1]
        k = key.key if hasattr(key, "key") else str(key)
        if (k in ("kernel", "embedding") or k.startswith("mhc_phi")) and getattr(leaf, "ndim", 0) >= 2:
            return np.asarray(cq_quantize(jnp.asarray(np.asarray(leaf, np.float32)), 4, 8))
        return np.asarray(leaf)

    flat = jtu.tree_map_with_path(qleaf, params)
    for path in ablate:
        node = flat
        for p in path[:-1]:
            node = node[p]
        node[path[-1]] = np.zeros_like(np.asarray(node[path[-1]]))
    row1_jax = np.asarray(
        model.apply({"params": flat}, jnp.asarray(tokens[None]), method=model.hidden_cells)
    )[0, 0, 1]
    row1_np = numpy_row1(flat, tokens)
    return row1_jax, row1_np


cases = {
    "orig": [],
    "no_attn": [("stack", "layers", "block", "self_attn", "out_proj", "kernel")],
    "no_hada": [("stack", "layers", "block", "hadamard_mlp", "d4")],
    "no_gate": [("stack", "layers", "block", "self_attn", "gate_proj", "kernel")],
    "no_postnorm": [("stack", "layers", "block", "post_attn_norm", "scale")],
}
for name, ab in cases.items():
    rj, rn = run(ab)
    diff = np.abs(rj - rn)
    print(f"{name:12s} worst diff {diff.max():.6f} at {diff.argmax()}  jax={rj[diff.argmax()]:.5f} np={rn[diff.argmax()]:.5f}")
