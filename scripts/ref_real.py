"""Independent numpy reference for tiny.cact layer 0 at position 0.
Follows needle/model/architecture.py exactly; compares against Rust DBG dumps."""
import re
import sys

import numpy as np

sys.path.insert(0, "/var/home/a/code/needle")
from needle.model.export import read_export

geo, ts = read_export("/var/home/a/code/needle-rs/models/needle3.cact")
L, n, d = geo["num_layers"], geo["mhc_lanes"], geo["d_model"]
h, kvh, qh, vh = geo["num_heads"], geo["num_kv_heads"], geo["qk_head_dim"], geo["v_head_dim"]
print("geometry:", dict(L=L,n=n,d=d,h=h,kvh=kvh,qh=qh,vh=vh))

# ---- unpack canon order ----
pos = 0
embedding = ts[pos]; pos += 1
layers = []
for _ in range(L):
    names = ["norm_in", "q_proj", "k_proj", "v_proj", "q_taps", "k_taps", "v_taps",
             "q_norm", "k_norm", "gate_proj", "out_proj", "post_norm", "attn_gate",
             "pre_hada", "d1", "d2", "b2", "d3", "d4", "w1a", "w1b", "w2a", "w2b",
             "w3a", "w3b", "cond_v", "cond_u"]
    layer = {}
    for nm in names:
        layer[nm] = ts[pos]; pos += 1
    layers.append(layer)
mhc = {}
for nm in ("mhc_a_pre", "mhc_a_post", "mhc_a_res", "mhc_b_pre", "mhc_b_post", "mhc_b_res",
           "mhc_phi_pre", "mhc_phi_post", "mhc_phi_res"):
    mhc[nm] = ts[pos]; pos += 1
p1, p2 = ts[pos], ts[pos + 1]; pos += 2
final_norm = ts[pos]; pos += 1

EPS = 1e-6

def zcrms(x, scale):
    return (1 + scale) * x / np.sqrt(np.mean(x.astype(np.float32) ** 2) + EPS)

def rms_unit(x):
    xf = x.astype(np.float32)
    return xf * (np.mean(xf ** 2) + EPS) ** -0.5

def sinkhorn(logits, iters=20):
    k = logits
    for _ in range(iters):
        k = k - logsumexp(k, axis=-1, keepdims=True)
        k = k - logsumexp(k, axis=-2, keepdims=True)
    return np.exp(k)

def logsumexp(x, axis, keepdims):
    m = np.max(x, axis=axis, keepdims=True)
    return m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=keepdims))

token = 2
x0 = embedding[token] * np.sqrt(d)
stream = np.tile(x0, (n, 1))  # (n, d)

nx = rms_unit(stream.reshape(-1))
li = 0
lane = li % n
pre_off = 8 * np.eye(n)[lane] - 4
post_off = -4 * (1 - np.eye(n)[lane])
# phi tensors are exported (L*X, nC); restore (L, nC, X)
nC = n * d
phi_pre = mhc["mhc_phi_pre"].reshape(L, n, nC).transpose(0, 2, 1)   # (L, nC, n)
phi_post = mhc["mhc_phi_post"].reshape(L, n, nC).transpose(0, 2, 1)
phi_res = mhc["mhc_phi_res"].reshape(L, n * n, nC).transpose(0, 2, 1)  # (L, nC, n*n)
hpre = 1 / (1 + np.exp(-(mhc["mhc_a_pre"][li] * (nx @ phi_pre[li]) + mhc["mhc_b_pre"][li] + pre_off)))
u = np.einsum("n,nc->c", hpre, stream)
# print("REF u      =", u.round(6).tolist())

lw = layers[li]
x = u.copy()  # no engram at layer 0
attn_in = zcrms(x, lw["norm_in"])
# print("REF attn_in=", attn_in.round(6).tolist())

q = lw["q_proj"] @ attn_in
k = lw["k_proj"] @ attn_in
v = lw["v_proj"] @ attn_in
# conv taps at t0: only j=0
q = q.reshape(h, qh); k = k.reshape(kvh, qh); v = v.reshape(kvh, vh)
for hh in range(h):
    q[hh] = zcrms(q[hh], lw["q_norm"])
for kk in range(kvh):
    k[kk] = zcrms(k[kk], lw["k_norm"])
# rope at t0 = identity
out = np.zeros((h, vh))
for hh in range(h):
    kk = hh // (h // kvh)
    score = q[hh] @ k[kk] / np.sqrt(qh)
    w = np.exp(score) / np.exp(score).sum()
    out[hh] = w * v[kk]
gate = 1 / (1 + np.exp(-(lw["gate_proj"] @ attn_in)))
attn_out = (out.reshape(-1) * gate)
hattn = lw["out_proj"] @ attn_out
x = x + (1 / (1 + np.exp(-lw["attn_gate"][0]))) * zcrms(hattn, lw["post_norm"])
# print("REF x_post_attn =", x.round(6).tolist())

xin = zcrms(x, lw["pre_hada"])
# hadamard
hada_n = geo["hada_n"]
ba = 1 << (((hada_n - 1).bit_length()) // 2)
bb = hada_n // ba

def kron_apply(z, a, b):
    z2 = z.reshape(a.shape[0], b.shape[0])
    z2 = np.einsum("ij,ik,jl->kl", z2, a, b)
    return z2.reshape(-1)

cv = xin @ lw["cond_v"]  # (8,)
import json as _json
print("REFX xin =", _json.dumps(np.asarray(xin, np.float64).round(5).tolist()))
print("REFX cv =", _json.dumps(np.asarray(cv, np.float64).round(5).tolist()))
sm = np.exp(cv - cv.max()); sm /= sm.sum()
cond = 1 + sm @ lw["cond_u"]  # (hada_n,)
z = np.zeros(hada_n)
z[:d] = xin
dump = {}
dump["k1"] = kron_apply(lw["d1"] * z, lw["w1a"], lw["w1b"])
dump["p1"] = dump["k1"][p1.astype(int)]
dump["cond"] = cond
z = dump["p1"]
pre2 = lw["d2"] * cond * z + lw["b2"]
dump["k2"] = kron_apply(pre2 / (1 + np.exp(-pre2)), lw["w2a"], lw["w2b"])
dump["p2"] = dump["k2"][p2.astype(int)]
z = dump["p2"]
dump["k3"] = kron_apply(lw["d3"] * z, lw["w3a"], lw["w3b"])
hada = (lw["d4"] * dump["k3"])[:d]
import json as _json
for kk, vv in dump.items():
    print(f"REFX {kk} =", _json.dumps(np.asarray(vv, np.float64).round(4).tolist()))
# print("REF hada_out =", hada.round(6).tolist())
block_out = x + hada
# print("REF block_out=", block_out.round(6).tolist())
y = block_out - u
# print("REF y =", y.round(6).tolist())

hpost = 2 / (1 + np.exp(-(mhc["mhc_a_post"][li] * (nx @ phi_post[li]) + mhc["mhc_b_post"][li] + post_off)))
# print("REF hpre =", hpre.round(6).tolist())
# print("REF hpost=", hpost.round(6).tolist())
res = (nx @ phi_res[li]).reshape(n, n)
import json as _json
print("REFX sink_in =", _json.dumps(np.asarray(mhc["mhc_a_res"][li] * res + mhc["mhc_b_res"][li], np.float64).round(6).tolist()))
hres = sinkhorn(mhc["mhc_a_res"][li] * res + mhc["mhc_b_res"][li])
print("REFX sink_out =", _json.dumps(np.asarray(hres, np.float64).reshape(-1).round(6).tolist()))
# print("REF hres =", list(np.round(hres.reshape(-1), 6)))
new_stream = hres @ stream + hpost[:, None] * y
row1 = new_stream.mean(axis=0)
# print("REF row1 =", row1.round(6).tolist())

import json
dump = {
  "u": np.asarray(u, np.float64).round(6).tolist(),
  "attn_in": np.asarray(attn_in, np.float64).round(6).tolist(),
  "x_post_attn": np.asarray(x, np.float64).round(6).tolist(),
  "hada_out": np.asarray(hada, np.float64).round(6).tolist(),
  "block_out": np.asarray(block_out, np.float64).round(6).tolist(),
  "y": np.asarray(y, np.float64).round(6).tolist(),
  "hpre": np.asarray(hpre, np.float64).round(6).tolist(),
  "hpost": np.asarray(hpost, np.float64).round(6).tolist(),
  "hres": np.asarray(hres, np.float64).reshape(-1).round(6).tolist(),
  "row1": np.asarray(row1, np.float64).round(6).tolist(),
}
with open("/tmp/ref_dbg.json", "w") as fh:
    json.dump(dump, fh)
print("json dumped")
