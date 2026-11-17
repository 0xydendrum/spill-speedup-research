"""Pre-trained Engram from a snapshot (0xydendrum's idea, 2026-11-15): before carving, run the dense model over text and
store, per recent-token pattern (n-gram), what each dense feed-forward layer produced that the carved layer misses.
At inference the carved model looks the correction up (O(1), could live in RAM/SSD) and adds it back. No training.

  python engram_snapshot.py --model path/to/Qwen3.5-2B --config 0.25:4 --rank 32

Per layer l:  gap_t = dense_ffn(x_t) - carved_ffn(x_t)
  basis   : mean mu_l and top-r principal directions U_l of the gap (gap is stored as an r-number code)
  tables  : average code per key: 1-gram (current token, exact), 2-gram and 3-gram (hashed buckets)
  lookup  : back-off: 3-gram if seen >= min_count, else 2-gram, else 1-gram, else nothing (bias only)
  output  : carved_ffn(x) + g * (mu_l + U_l @ code)
Scored by WikiText-2 test perplexity (held out from the snapshot text).
"""
import argparse, json, time
from pathlib import Path

import torch
import torch.nn.functional as F

import carve_proto as C

K1, K2 = 1_000_003, 998_244_353


def keys(ids, V, B2, B3):
    """ids (B,T) -> flattened 1-gram id, hashed 2-gram, hashed 3-gram keys ending at each position."""
    p1 = torch.cat([torch.full_like(ids[:, :1], V), ids[:, :-1]], 1)
    p2 = torch.cat([torch.full_like(ids[:, :1], V), p1[:, :-1]], 1)
    h2 = (p1 * K1 + ids) % B2
    h3 = ((p2 * K1 + p1) * K2 + ids) % B3
    return ids.reshape(-1), h2.reshape(-1), h3.reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--experts", type=int, default=32)
    ap.add_argument("--config", default="0.25:4")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--b2", type=int, default=1 << 19)
    ap.add_argument("--b3", type=int, default=1 << 20)
    ap.add_argument("--snap-tokens", type=int, default=2_500_000)
    ap.add_argument("--basis-tokens", type=int, default=65536)
    ap.add_argument("--eval-tokens", type=int, default=16384)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--min-count", type=int, default=3)
    ap.add_argument("--out", default="engram_results.jsonl")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.snap_tokens, a.basis_tokens, a.eval_tokens, a.seq, a.b2, a.b3 = 16384, 8192, 2048, 512, 1 << 12, 1 << 13
    torch.backends.cuda.matmul.allow_tf32 = True
    t0 = time.time()
    model, tok = C.load_model(a.model, "cuda")
    layers = C.mlps(model)
    L, H = len(layers), layers[0].gate_proj.in_features
    V = model.get_input_embeddings().weight.shape[0]
    sf, k = float(a.config.split(":")[0]), int(a.config.split(":")[1])
    r = a.rank
    snap = C.text_tokens(tok, "train", a.snap_tokens)
    ev = C.text_tokens(tok, "test", a.eval_tokens)
    C.log(f"{L} layers, H={H}, vocab={V}, snapshot text {len(snap)} tokens")

    # ---- carve (same recipe as the other prototypes) ----
    freq, xs, acts = C.collect(model, layers, snap[: min(len(snap), 65536)], a.seq, "cuda", 8192 if not a.smoke else 1024)
    carves = [C.carve_layer(m, freq[i], xs[i], acts[i], a.experts, sf) for i, m in enumerate(layers)]
    del xs, acts
    keep_info = []
    for i, m in enumerate(layers):
        c = carves[i]
        n_ff = m.gate_proj.out_features
        owner = torch.full((n_ff,), -1, dtype=torch.long, device="cuda")
        E = c["experts"].cuda()
        owner[E.flatten()] = torch.arange(E.shape[0], device="cuda").repeat_interleave(E.shape[1])
        onehot = F.one_hot(owner.clamp(min=0), E.shape[0]).to(torch.bfloat16) * (owner >= 0).unsqueeze(1).to(torch.bfloat16)
        keep_info.append((onehot, (owner < 0).to(torch.bfloat16), c["router_w"].cuda(), c["router_b"].cuda()))

    def carved_out(i, x2):
        m = layers[i]
        onehot, shared, rw, rb = keep_info[i]
        a_ = m.act_fn(F.linear(x2, m.gate_proj.weight)) * F.linear(x2, m.up_proj.weight)
        idx = (x2.float() @ rw.T + rb).topk(k, 1).indices
        keep_e = torch.zeros(x2.shape[0], onehot.shape[1], device=x2.device, dtype=x2.dtype).scatter_(1, idx, 1.0)
        return F.linear(a_ * (keep_e @ onehot.T + shared), m.down_proj.weight)

    holder = {}
    chunks = lambda ids, n: [ids[i:i + a.seq] for i in range(0, min(len(ids), n) - a.seq + 1, a.seq)]

    # ---- pass 1: mean and principal directions of the gap ----
    cov = torch.zeros(L, H, H, device="cuda"); s1 = torch.zeros(L, H, device="cuda"); cnt = [0]
    def hook_basis(i):
        def fh(mod, inp, out):
            x2 = inp[0].reshape(-1, H); gap = (out.reshape(-1, H) - carved_out(i, x2)).float()
            cov[i] += gap.T @ gap; s1[i] += gap.sum(0)
            if i == 0: cnt[0] += gap.shape[0]
        return fh
    hs = [m.register_forward_hook(hook_basis(i)) for i, m in enumerate(layers)]
    with torch.no_grad():
        ch = chunks(snap, a.basis_tokens)
        for j in range(0, len(ch), 4):
            model(input_ids=torch.stack(ch[j:j + 4]).cuda())
    for h in hs: h.remove()
    mu = s1 / cnt[0]
    U = torch.zeros(L, H, r, device="cuda"); pca_frac = []
    for i in range(L):
        C_ = cov[i] / cnt[0] - torch.outer(mu[i], mu[i])
        ev_, evec = torch.linalg.eigh(C_)
        U[i] = evec[:, -r:]
        pca_frac.append((ev_[-r:].sum() / ev_.sum().clamp(min=1e-9)).item())
    del cov
    C.log(f"gap basis: rank-{r} PCA keeps {sum(pca_frac)/L:.1%} of the gap's variance on average")

    # ---- pass 2: snapshot tables (average code per key) on CPU ----
    T1 = torch.zeros(L, V, r); T2 = torch.zeros(L, a.b2, r); T3 = torch.zeros(L, a.b3, r)
    N1 = torch.zeros(V); N2 = torch.zeros(a.b2); N3 = torch.zeros(a.b3)
    def hook_snap(i):
        def fh(mod, inp, out):
            x2 = inp[0].reshape(-1, H)
            code = ((out.reshape(-1, H) - carved_out(i, x2)).float() - mu[i]) @ U[i]     # (N, r)
            code = code.cpu(); k1, k2, k3 = holder["keys"]
            T1[i].index_add_(0, k1, code); T2[i].index_add_(0, k2, code); T3[i].index_add_(0, k3, code)
        return fh
    hs = [m.register_forward_hook(hook_snap(i)) for i, m in enumerate(layers)]
    with torch.no_grad():
        ch = chunks(snap, a.snap_tokens)
        for j in range(0, len(ch), 4):
            ids = torch.stack(ch[j:j + 4])
            k1, k2, k3 = keys(ids, V, a.b2, a.b3)
            holder["keys"] = (k1, k2, k3)
            N1.index_add_(0, k1, torch.ones(k1.shape[0])); N2.index_add_(0, k2, torch.ones(k2.shape[0])); N3.index_add_(0, k3, torch.ones(k3.shape[0]))
            model(input_ids=ids.cuda())
            if (j // 4) % 100 == 0:
                C.log(f"snapshot {j * a.seq}/{len(ch) * a.seq} tokens")
    for h in hs: h.remove()
    for T, N in ((T1, N1), (T2, N2), (T3, N3)):
        T.div_(N.clamp(min=1).unsqueeze(0).unsqueeze(2))
    C.log(f"snapshot done: {int((N1 > 0).sum())} unigrams, {int((N2 > 0).sum())} 2-gram buckets, {int((N3 > 0).sum())} 3-gram buckets used")

    # ---- evaluation: carved model + looked-up correction ----
    state = {"mode": None, "g": 1.0}
    energy = {}
    def lookup(i, k1, k2, k3, mode):
        if mode == "bias":
            return torch.zeros(k1.shape[0], r)
        c1 = T1[i][k1]
        if mode == "uni":
            return c1 * (N1[k1] >= a.min_count).unsqueeze(1)
        c2, c3 = T2[i][k2], T3[i][k3]
        use3 = (N3[k3] >= a.min_count).unsqueeze(1); use2 = (N2[k2] >= a.min_count).unsqueeze(1); use1 = (N1[k1] >= a.min_count).unsqueeze(1)
        return torch.where(use3, c3, torch.where(use2, c2, torch.where(use1, c1, torch.zeros_like(c1))))

    def install(i, m):
        def fwd(x):
            mode = state["mode"]
            if mode == "dense":
                return m._orig_forward(x)
            shp = x.shape; x2 = x.reshape(-1, H)
            out = carved_out(i, x2)
            if mode != "carve":
                k1, k2, k3 = holder["keys"]
                corr = mu[i] + (lookup(i, k1, k2, k3, mode).cuda() @ U[i].T)
                if state.get("measure"):
                    gap = (m._orig_forward(x).reshape(-1, H) - out).float()
                    e = energy.setdefault(mode, [0.0, 0.0])
                    e[0] += (gap ** 2).sum().item(); e[1] += ((gap - corr) ** 2).sum().item()
                out = out + (state["g"] * corr).to(out.dtype)
            return out.reshape(shp)
        m._orig_forward = m.forward
        m.forward = fwd
    for i, m in enumerate(layers):
        install(i, m)

    @torch.no_grad()
    def ppl(mode, g=1.0, measure=False):
        state.update(mode=mode, g=g, measure=measure)
        nll, n = 0.0, 0
        ch = chunks(ev, len(ev))
        for j in range(0, len(ch), 4):
            ids = torch.stack(ch[j:j + 4])
            holder["keys"] = keys(ids, V, a.b2, a.b3)
            o = model(input_ids=ids.cuda(), labels=ids.cuda())
            kk = ids.numel() - ids.shape[0]
            nll += o.loss.float().item() * kk; n += kk
        return float(torch.tensor(nll / n).exp())

    base = ppl("dense")
    res = {"model": Path(a.model).name, "config": a.config, "rank": r, "snap_tokens": len(snap), "dense_ppl": round(base, 4),
           "pca_keep": round(sum(pca_frac) / L, 4), "smoke": a.smoke, "configs": []}
    C.log(f"dense perplexity {base:.3f}")
    for mode, g in [("carve", 1.0), ("bias", 1.0), ("uni", 1.0), ("backoff", 1.0), ("backoff", 0.5), ("backoff", 0.75)]:
        p = ppl(mode, g, measure=(g == 1.0 and mode in ("bias", "uni", "backoff")))
        row = {"mode": mode, "g": g, "ppl": round(p, 4), "ratio": round(p / base, 4)}
        if mode in energy and g == 1.0:
            e = energy[mode]; row["gap_energy_removed"] = round(1 - e[1] / max(e[0], 1e-9), 4)
        res["configs"].append(row)
        C.log(json.dumps(row))
    res["total_s"] = round(time.time() - t0, 1)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(res) + "\n")
    if not a.smoke:
        torch.save({"mu": mu.cpu(), "U": U.cpu(), "T1": T1.half(), "T2": T2.half(), "T3": T3.half(), "N1": N1, "N2": N2, "N3": N3,
                    "config": a.config}, Path(a.out).with_name(f"engram-snapshot-{Path(a.model).name}.pt"))
    C.log("done")


if __name__ == "__main__":
    main()
