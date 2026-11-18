"""Engram v2 step A0: how much of the healed carve's residual drift can n-gram keys predict? (finding F7)

Forward passes only. One model, two passes per batch: dense MLPs, then the healed carve's MLPs (each pass stops after
the last probe layer, so the 248K-vocab lm_head never runs during the fill).

  python engram_diag.py --model <Qwen3.5-2B dir> --healed cache/healed-Qwen3.5-2B-0.25k4.pt [--smoke]

Per probe layer L:  d_L = h_dense(L) - h_healed(L)   (healed model on its OWN upstream states), full width.
  basis   rank-r PCA (mu_L, U_L) of d_L on the FineWeb basis slice; code c = U^T (d - mu)
  tables  order 1 exact (token id) + orders 2, 3 hashed with K heads each (2^b rows); per row: sum of codes, count,
          sum ||c||^2. The back-off column uses head 0 of orders 2/3, so it shares those accumulators with the
          A-readout column (same hash, same sums: identical numbers, 2 fewer tables per layer).
          router-key column: bucket = healed router's top-2 expert ids at L.
  predict back-off (3 -> 2 -> 1, shrinkage lambda toward the parent, order 1 toward 0), A-readout (mean over the 2K
          head bucket means, no back-off), router-key (shrunk bucket mean).
  R^2     1 - [sum ||r||^2 + sum ||c - c_hat||^2] / sum ||d - mean(d)||^2, full width, pooled (r = part of d - mu
          outside U, so the rank-r loss counts against R^2).
Also: true coverage and collision rate from an exact key log, a collision-free exact-key back-off at --exact-tokens,
held-out injected gap closed G and an oracle ceiling (eval text filled into the table) at the best layer, dense /
healed baselines on the same slices, the PopQA SD pilot, and the decision table outcome.
Results: one JSON line in --out; progress in the log.
"""
import argparse, json, math, os, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import carve_proto as C
import heal_proto as H

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "engram"
MULTS = [0x9E3779B97F4A7C15, 0xC2B2AE3D27D4EB4F, 0x165667B19E3779F9, 0xD6E8FEB86659FD93, 0xFF51AFD7ED558CCD,
         0xC4CEB9FE1A85EC53]
MULTS = [((m + 2 ** 63) % 2 ** 64) - 2 ** 63 for m in MULTS]   # as signed int64 (multiplication wraps)


class StopForward(Exception):
    pass


def bucket_t(key, mult, bits):
    return ((key * mult) >> (63 - bits)) & ((1 << bits) - 1)


def bucket_np(key, mult, bits):
    with np.errstate(over="ignore"):
        return ((key.astype(np.int64) * np.int64(mult)) >> np.int64(63 - bits)) & np.int64((1 << bits) - 1)


def ngram_keys(tok, V):
    """tok (B, T) int64 -> flattened k1, k2, k3 (exact int64 keys) and validity masks for orders 2, 3."""
    B, T = tok.shape
    p1 = torch.zeros_like(tok); p1[:, 1:] = tok[:, :-1]
    p2 = torch.zeros_like(tok); p2[:, 2:] = tok[:, :-2]
    pos = torch.arange(T, device=tok.device).expand(B, T)
    k1 = tok
    k2 = p1 * V + tok
    k3 = (p2 * V + p1) * V + tok
    return k1.reshape(-1), k2.reshape(-1), k3.reshape(-1), (pos >= 1).reshape(-1), (pos >= 2).reshape(-1)


class Tables:
    """Per-layer accumulators on one device. Index 0 = order 1 (exact), then orders 2..3 x heads (hashed)."""

    def __init__(self, n_layers, V, r, bits, heads, dev):
        self.V, self.r, self.bits, self.heads, self.dev = V, r, bits, heads, dev
        nb = 1 << bits
        self.sizes = [V] + [nb] * (2 * heads)
        self.sum = [[torch.zeros(s, r, device=dev) for s in self.sizes] for _ in range(n_layers)]
        self.cnt = [[torch.zeros(s, device=dev) for s in self.sizes] for _ in range(n_layers)]
        self.sq = [[torch.zeros(s, device=dev) for s in self.sizes] for _ in range(n_layers)]
        self.rsum = [torch.zeros(1024, r, device=dev) for _ in range(n_layers)]
        self.rcnt = [torch.zeros(1024, device=dev) for _ in range(n_layers)]

    def gb(self):
        per = sum(s * (self.r + 2) for s in self.sizes) + 1024 * (self.r + 1)
        return per * 4 * len(self.sum) / 1e9

    def rows(self, k1, k2, k3, v2, v3):
        """Row index per table (-1 = no row) for flattened keys."""
        out = [k1]
        for order, (k, v) in enumerate(((k2, v2), (k3, v3))):
            for h in range(self.heads):
                b = bucket_t(k, MULTS[order * self.heads + h], self.bits)
                out.append(torch.where(v, b, torch.full_like(b, -1)))
        return out

    def add(self, li, rows, c, rkey=None):
        sq = (c * c).sum(1)
        ones = torch.ones(c.shape[0], device=c.device)
        for t, idx in enumerate(rows):
            m = idx >= 0
            ii = idx[m]
            self.sum[li][t].index_add_(0, ii, c[m]); self.cnt[li][t].index_add_(0, ii, ones[m]); self.sq[li][t].index_add_(0, ii, sq[m])
        if rkey is not None:
            self.rsum[li].index_add_(0, rkey, c); self.rcnt[li].index_add_(0, rkey, ones)

    def mean(self, li, t, idx):
        """bucket mean code and count at row idx (-1 -> zero, count 0)."""
        safe = idx.clamp(min=0)
        n = self.cnt[li][t][safe] * (idx >= 0)
        s = self.sum[li][t][safe] * (idx >= 0).unsqueeze(1)
        return s / n.clamp(min=1).unsqueeze(1), n

    def predict(self, li, rows, lam, rkey=None):
        m1, n1 = self.mean(li, 0, rows[0])
        p = (n1 / (n1 + lam)).unsqueeze(1) * m1
        heads = []
        for order in range(2):
            t = 1 + order * self.heads
            m, n = self.mean(li, t, rows[t])
            w = (n / (n + lam)).unsqueeze(1)
            p = w * m + (1 - w) * p                                          # back-off via head 0
            for h in range(self.heads):
                heads.append(self.mean(li, 1 + order * self.heads + h, rows[1 + order * self.heads + h])[0])
        aread = torch.stack(heads).mean(0)
        out = {"backoff": p, "aread": aread}
        if rkey is not None:
            n = self.rcnt[li][rkey]
            out["router"] = (n / (n + lam)).unsqueeze(1) * (self.rsum[li][rkey] / n.clamp(min=1).unsqueeze(1))
        return out


class Diag:
    def __init__(self, a):
        self.a = a
        torch.backends.cuda.matmul.allow_tf32 = True
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.95)
        self.model, self.tok = C.load_model(a.model, "cuda")
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.mlps = C.mlps(self.model)
        self.dec = [m for n, m in self.model.named_modules() if type(m).__name__.endswith("DecoderLayer")
                    and "visual" not in n and "mtp" not in n.lower()]
        assert len(self.dec) == len(self.mlps), (len(self.dec), len(self.mlps))
        self.V = self.model.get_input_embeddings().weight.shape[0]
        self.Hd = self.mlps[0].gate_proj.in_features
        self.layers = [int(x) for x in a.layers.split(",")]
        k = int(a.config.split(":")[1])
        ck = torch.load(a.healed, weights_only=False)
        self.mods = []
        for i, m in enumerate(self.mlps):
            mod = H.CarvedFFN(m, ck["carves"][i], k).to(torch.bfloat16)
            mod.load_state_dict(ck["state"][i])
            for p in mod.parameters():
                p.requires_grad_(False)
            self.mods.append(mod)
        del ck
        self.cap, self.rtop = {}, {}
        for li in self.layers:
            self.dec[li].register_forward_hook(self._cap_hook(li))
            self.mods[li].register_forward_pre_hook(self._router_hook(li))
        self.stop_at = max(self.layers)
        self.inject = None                                                   # (layer, fn(ids) -> add) for ppl evals
        self.dec[self.stop_at].register_forward_hook(self._stop_hook)
        self.stopping = True

    # ---- hooks / mode switches ----
    def _cap_hook(self, li):
        def fh(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if self.inject is not None and self.inject[0] == li:
                h = h + self.inject[1]().to(h.dtype)
                out = (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
            self.cap[li] = h
            return out
        return fh

    def _router_hook(self, li):
        def pre(mod, inp):
            h = inp[0]
            self.rtop[li] = (h.float() @ mod.rw.float().T + mod.rb.float()).topk(2, dim=1).indices
        return pre

    def _stop_hook(self, mod, inp, out):
        if self.stopping:
            raise StopForward

    def dense(self):
        for m in self.mlps:
            C.uninstall(m)

    def healed(self):
        for m, mod in zip(self.mlps, self.mods):
            H.install_healed(m, mod)

    @torch.no_grad()
    def run(self, ids):
        self.cap = {}
        try:
            self.model(input_ids=ids, use_cache=False)
        except StopForward:
            pass
        return {li: self.cap[li] for li in self.layers}

    @torch.no_grad()
    def drift(self, ids):
        """ids (B, T) on cuda -> {layer: d (B*T, Hd) float32}, {layer: router top-2 key (B*T,)}."""
        self.dense(); hd = self.run(ids)
        self.healed(); hh = self.run(ids)
        d = {li: (hd[li] - hh[li]).float().reshape(-1, self.Hd) for li in self.layers}
        rk = {li: (self.rtop[li][:, 0] * 32 + self.rtop[li][:, 1]).long() for li in self.layers}
        return d, rk

    def batches(self, arr, n_tokens=None):
        seq, bs = self.a.seq, self.a.bs
        n = len(arr) if n_tokens is None else min(len(arr), n_tokens)
        n_seq = n // seq
        for i in range(0, n_seq, bs):
            j = min(n_seq, i + bs)
            yield torch.from_numpy(arr[i * seq: j * seq].astype(np.int64)).view(j - i, seq).cuda()

    # ---- phases ----
    def basis(self, arr):
        r = self.a.rank
        s1 = {li: torch.zeros(self.Hd, device="cuda", dtype=torch.float64) for li in self.layers}
        s2 = {li: torch.zeros(self.Hd, self.Hd, device="cuda", dtype=torch.float64) for li in self.layers}
        n = 0
        for ids in self.batches(arr):
            d, _ = self.drift(ids)
            for li in self.layers:
                x = d[li].double(); s1[li] += x.sum(0); s2[li] += x.T @ x
            n += ids.numel()
        self.mu, self.U, self.keep = {}, {}, {}
        for li in self.layers:
            mu = s1[li] / n
            cov = s2[li] / n - torch.outer(mu, mu)
            ev, evec = torch.linalg.eigh(cov)
            self.mu[li] = mu.float()
            self.U[li] = evec[:, -r:].flip(1).float().contiguous()
            self.keep[li] = float(ev[-r:].sum() / ev.clamp(min=0).sum())
        C.log("basis: variance kept by rank", r, {li: round(v, 4) for li, v in self.keep.items()})

    def codes(self, d, li):
        return (d - self.mu[li]) @ self.U[li]

    def prep_eval(self, name, arr):
        """Cache per eval position: codes, residual energy, keys, router key; and SST per layer."""
        ev = {"codes": {li: [] for li in self.layers}, "r2": {li: 0.0 for li in self.layers},
              "s1": {li: torch.zeros(self.Hd, device="cuda", dtype=torch.float64) for li in self.layers},
              "ss": {li: 0.0 for li in self.layers}, "rk": {li: [] for li in self.layers}, "tok": [], "n": 0}
        for ids in self.batches(arr):
            d, rk = self.drift(ids)
            for li in self.layers:
                c = self.codes(d[li], li)
                x = d[li] - self.mu[li]
                ev["codes"][li].append(c)
                ev["r2"][li] += float((x * x).sum() - (c * c).sum())
                ev["s1"][li] += d[li].double().sum(0); ev["ss"][li] += float((d[li].double() ** 2).sum())
                ev["rk"][li].append(rk[li])
            ev["tok"].append(ids)
            ev["n"] += ids.numel()
        tok = torch.cat(ev["tok"])
        ev["keys"] = ngram_keys(tok, self.V)
        ev["tok"] = tok
        for li in self.layers:
            ev["codes"][li] = torch.cat(ev["codes"][li]); ev["rk"][li] = torch.cat(ev["rk"][li])
            mean = ev["s1"][li] / ev["n"]
            ev["sst"] = ev.get("sst", {}); ev["sst"][li] = ev["ss"][li] - ev["n"] * float((mean * mean).sum())
            del ev["s1"][li]
        C.log(f"eval {name}: {ev['n']} tokens cached")
        return ev

    def r2(self, tables, ev):
        res = {}
        rows = tables.rows(*ev["keys"])
        for i, li in enumerate(self.layers):
            pred = tables.predict(i, rows, self.a.lam, ev["rk"][li])
            c = ev["codes"][li]
            res[li] = {k: 1 - (ev["r2"][li] + float(((c - p) ** 2).sum())) / ev["sst"][li] for k, p in pred.items()}
        return res

    @torch.no_grad()
    def ppl(self, arr, li=None, add_fn=None):
        """Perplexity with bs=1 over the slice (248K vocab), optionally adding a per-position correction at layer li."""
        self.stopping = False
        nll, n = 0.0, 0
        seq = self.a.seq
        for s in range(len(arr) // seq):
            ids = torch.from_numpy(arr[s * seq:(s + 1) * seq].astype(np.int64)).view(1, seq).cuda()
            if add_fn is not None:
                self.inject = (li, lambda ids=ids: add_fn(ids))
            out = self.model(input_ids=ids, labels=ids, use_cache=False)
            nll += out.loss.float().item() * (seq - 1); n += seq - 1
        self.inject = None
        self.stopping = True
        return math.exp(nll / n)

    def add_fn(self, tables, t_idx, li, col):
        """Correction mu + U c_hat for layer li, read from table index t_idx (n-gram columns only)."""
        def f(ids):
            rows = tables.rows(*ngram_keys(ids, self.V))
            pred = tables.predict(t_idx, rows, self.a.lam, None)[col]
            return (self.mu[li] + pred @ self.U[li].T).view(ids.shape[0], ids.shape[1], -1)
        return f

    def popqa_sd(self, path):
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        lp = {"dense": [], "healed": []}
        self.stopping = False
        for mode in ("dense", "healed"):
            getattr(self, mode)()
            for r in rows:
                prompt = f"Q: {r['question']}\nA:"
                best = -1e9
                for ans in r["answers"][:3]:
                    p = self.tok(prompt, return_tensors="pt").input_ids[0]
                    t = self.tok(" " + ans, add_special_tokens=False, return_tensors="pt").input_ids[0]
                    ids = torch.cat([p, t]).view(1, -1).cuda()
                    with torch.no_grad():
                        logits = self.model(input_ids=ids, use_cache=False).logits[0, len(p) - 1:-1].float()
                    best = max(best, float(F.log_softmax(logits, -1).gather(1, t.cuda().view(-1, 1)).sum()))
                lp[mode].append(best)
        self.stopping = True
        dlt = np.array(lp["healed"]) - np.array(lp["dense"])
        return {"items": len(rows), "mean_dense": float(np.mean(lp["dense"])), "mean_healed": float(np.mean(lp["healed"])),
                "delta_mean": float(dlt.mean()), "delta_sd": float(dlt.std(ddof=1))}


def coverage(keylog2, keylog3, ev, bits, mult2, mult3):
    """True coverage (exact n-gram seen >= 3 times in the fill) and hash-collision stats for head 0."""
    out = {}
    for name, log_, k, v, mult in (("order2", keylog2, ev["keys"][1], ev["keys"][3], mult2),
                                   ("order3", keylog3, ev["keys"][2], ev["keys"][4], mult3)):
        uk, cnt = np.unique(log_, return_counts=True)
        ek = k[v].cpu().numpy()
        pos = np.searchsorted(uk, ek).clip(0, len(uk) - 1)
        seen = np.where(uk[pos] == ek, cnt[pos], 0)
        b = bucket_np(uk, mult, bits)
        per_bucket = np.bincount(b, minlength=1 << bits)
        eb = bucket_np(ek, mult, bits)
        out[name] = {"distinct_keys": int(len(uk)), "coverage_ge3": float((seen >= 3).mean()),
                     "coverage_ge1": float((seen >= 1).mean()),
                     "keys_per_used_bucket": float(len(uk) / max(1, (per_bucket > 0).sum())),
                     "eval_bucket_shared_share": float((per_bucket[eb] > 1).mean())}
    return out


def exact_r2(D, cache_codes, cache_keys, ev, lam):
    """Collision-free back-off 3 -> 2 -> 1 with exact keys, built from the first --exact-tokens fill positions."""
    k1, k2, k3, v2, v3 = cache_keys
    res = {}
    e1, e2, e3, ev2, ev3 = [x.cpu().numpy() for x in ev["keys"]]
    idx = {}
    for name, k, v, ek, evv in (("o2", k2, v2, e2, ev2), ("o3", k3, v3, e3, ev3)):
        uk, inv = np.unique(k[v], return_inverse=True)
        pos = np.searchsorted(uk, ek).clip(0, len(uk) - 1)
        hit = (uk[pos] == ek) & evv
        idx[name] = (uk, inv, np.where(hit, pos, -1))
    for li in D.layers:
        c = cache_codes[li].float().cuda()
        def table(rows_fill, n_rows, rows_eval):
            s = torch.zeros(n_rows, c.shape[1], device="cuda"); n = torch.zeros(n_rows, device="cuda")
            m = rows_fill >= 0
            rf = torch.from_numpy(rows_fill[m]).cuda()
            s.index_add_(0, rf, c[torch.from_numpy(m).cuda()]); n.index_add_(0, rf, torch.ones(len(rf), device="cuda"))
            re = torch.from_numpy(rows_eval).cuda()
            safe = re.clamp(min=0)
            nn_ = n[safe] * (re >= 0)
            mm = s[safe] * (re >= 0).unsqueeze(1) / nn_.clamp(min=1).unsqueeze(1)
            return mm, nn_
        m1, n1 = table(k1.astype(np.int64), D.V, e1.astype(np.int64))
        p = (n1 / (n1 + lam)).unsqueeze(1) * m1
        for name, v, evv in (("o2", v2, ev2), ("o3", v3, ev3)):
            uk, inv, re = idx[name]
            rows_fill = np.full(len(v), -1, dtype=np.int64); rows_fill[v] = inv
            m, n = table(rows_fill, len(uk), re)
            w = (n / (n + lam)).unsqueeze(1)
            p = w * m + (1 - w) * p
        cc = ev["codes"][li]
        res[li] = 1 - (ev["r2"][li] + float(((cc - p) ** 2).sum())) / ev["sst"][li]
        del c
    return res


def decide(r2_wiki, g_wiki, r2_exact1m, r2_hash1m, router_best, ngram_best, thr=0.05, gthr=5.0):
    if router_best > 2 * max(ngram_best, 1e-9) and router_best > thr:
        return "REDESIGN"
    if r2_wiki >= thr and g_wiki >= gthr:
        return "GO"
    if r2_wiki >= thr:
        return "KILL (predictable but does not move ppl)"
    if r2_exact1m >= thr and r2_hash1m < thr:
        return "RERUN (4x buckets)"
    if g_wiki >= gthr:
        return "INVESTIGATE"
    return "KILL"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--healed", default=str(ROOT / "cache" / "healed-Qwen3.5-2B-0.25k4.pt"))
    ap.add_argument("--config", default="0.25:4")
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--layers", default="3,7,11,15,19,23")
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--bits", type=int, default=19)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--lam", type=float, default=8.0)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--fill-tokens", type=int, default=20_000_000)
    ap.add_argument("--snapshots", default="1000000,5000000")
    ap.add_argument("--exact-tokens", type=int, default=1_000_000)
    ap.add_argument("--max-hours", type=float, default=6.0)
    ap.add_argument("--out", default=str(ROOT / "engram_diag_results.jsonl"))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.fill_tokens, a.snapshots, a.exact_tokens, a.bits, a.bs = 16384, "8192", 8192, 12, 2
        a.layers = "3,11"
    t0 = time.time()
    data = Path(a.data)
    basis = np.load(data / "fineweb_basis.npy")
    fill = np.load(data / "fineweb_fill.npy")[: a.fill_tokens]
    evs_np = {"wiki": np.load(data / "wiki_eval.npy"), "c4": np.load(data / "c4_eval.npy")}
    if a.smoke:
        basis = basis[:4096]
        evs_np = {k: v[:4096] for k, v in evs_np.items()}
    D = Diag(a)
    C.log(f"model loaded, V={D.V}, Hd={D.Hd}, probe layers {D.layers}")
    D.basis(basis)
    evs = {k: D.prep_eval(k, v) for k, v in evs_np.items()}
    T = Tables(len(D.layers), D.V, a.rank, a.bits, a.heads, "cuda")
    C.log(f"tables: {T.gb():.2f} GB")
    snaps = sorted(int(x) for x in a.snapshots.split(",") if x)
    res = {"model": Path(a.model).name, "healed": Path(a.healed).name, "layers": D.layers, "rank": a.rank,
           "bits": a.bits, "heads": a.heads, "lam": a.lam, "basis_keep": D.keep, "snapshots": {}, "smoke": a.smoke}
    keylog2, keylog3 = [], []
    cache_codes = {li: [] for li in D.layers}
    cache_keys = []
    done, truncated = 0, False
    tfill = time.time()

    def snapshot(tag):
        s = {"tokens": done, "elapsed_h": round((time.time() - t0) / 3600, 3)}
        for name, ev in evs.items():
            s[f"r2_{name}"] = {str(li): {k: round(v, 5) for k, v in d.items()} for li, d in D.r2(T, ev).items()}
        k2 = np.concatenate(keylog2) if keylog2 else np.zeros(0, np.int64)
        k3 = np.concatenate(keylog3) if keylog3 else np.zeros(0, np.int64)
        s["coverage_wiki"] = coverage(k2, k3, evs["wiki"], a.bits, MULTS[0], MULTS[a.heads])
        res["snapshots"][tag] = s
        C.log(f"snapshot {tag} @ {done} tok:", json.dumps({k: s[k] for k in ("r2_wiki",)})[:600])

    for ids in D.batches(fill):
        d, rk = D.drift(ids)
        k1, k2, k3, v2, v3 = ngram_keys(ids, D.V)
        rows = T.rows(k1, k2, k3, v2, v3)
        for i, li in enumerate(D.layers):
            c = D.codes(d[li], li)
            T.add(i, rows, c, rk[li])
            if done < a.exact_tokens:
                cache_codes[li].append(c.half().cpu())
        keylog2.append(k2[v2].cpu().numpy()); keylog3.append(k3[v3].cpu().numpy())
        if done < a.exact_tokens:
            cache_keys.append([x.cpu().numpy() for x in (k1, k2, k3, v2, v3)])
        prev = done
        done += ids.numel()
        for sn in snaps:
            if prev < sn <= done:
                snapshot(str(sn))
        if done % (1 << 21) < ids.numel():
            rate = (done) / (time.time() - tfill)
            C.log(f"fill {done}/{len(fill)} tok, {rate:.0f} tok/s, eta {(len(fill) - done) / rate / 3600:.2f} h")
        if time.time() - t0 > a.max_hours * 3600:
            truncated = True
            C.log("max-hours reached, stopping the fill")
            break
    res["truncated"] = truncated
    snapshot("final")
    res["fill_tok_per_s"] = round(done / (time.time() - tfill), 1)

    li_idx = {li: i for i, li in enumerate(D.layers)}                   # Tables index by position in D.layers

    # collision-free exact-key back-off at the --exact-tokens snapshot
    ck = [np.concatenate([b[j] for b in cache_keys]) for j in range(5)]
    ckc = {li: torch.cat(cache_codes[li]) for li in D.layers}
    res["exact_r2_wiki"] = {str(li): round(v, 5) for li, v in exact_r2(D, ckc, ck, evs["wiki"], a.lam).items()}
    del ckc, cache_codes

    # decision inputs: wiki, n-gram columns, final snapshot
    fin = res["snapshots"]["final"]["r2_wiki"]
    best_li, best_col, best_r2 = None, None, -1e9
    for li in D.layers:
        for col in ("backoff", "aread"):
            if fin[str(li)][col] > best_r2:
                best_li, best_col, best_r2 = li, col, fin[str(li)][col]
    router_best = max(fin[str(li)]["router"] for li in D.layers)
    s1m = res["snapshots"].get(str(snaps[0]), res["snapshots"]["final"])["r2_wiki"]
    hash1m = max(s1m[str(li)]["backoff"] for li in D.layers)
    exact1m = max(res["exact_r2_wiki"].values())
    C.log(f"best layer {best_li} col {best_col} R2 {best_r2:.4f}; router best {router_best:.4f}; exact@1M {exact1m:.4f} hashed@1M {hash1m:.4f}")

    # baselines + injected G + oracle on the same slices
    g = {}
    for name, arr in evs_np.items():
        D.dense(); pd_ = D.ppl(arr)
        D.healed(); ph = D.ppl(arr)
        pi = D.ppl(arr, best_li, D.add_fn(T, li_idx[best_li], best_li, best_col))
        g[name] = {"ppl_dense": pd_, "ppl_healed": ph, "ppl_injected": pi,
                   "G_pp": 100 * (ph - pi) / (ph - pd_) if ph != pd_ else 0.0}
        C.log(f"{name}: dense {pd_:.3f} healed {ph:.3f} injected {pi:.3f} G {g[name]['G_pp']:.2f} pp")
    # oracle: table filled on the eval text itself, best layer only
    for name, ev in evs.items():
        O = Tables(1, D.V, a.rank, a.bits, a.heads, "cuda")
        rows = O.rows(*ev["keys"])
        O.add(0, rows, ev["codes"][best_li], ev["rk"][best_li])
        D.healed()
        po = D.ppl(evs_np[name], best_li, D.add_fn(O, 0, best_li, best_col))
        g[name]["ppl_oracle"] = po
        g[name]["G_oracle_pp"] = 100 * (g[name]["ppl_healed"] - po) / (g[name]["ppl_healed"] - g[name]["ppl_dense"])
        del O
    res["gap"] = g
    res["best"] = {"layer": best_li, "column": best_col, "r2_wiki": best_r2, "router_best": router_best,
                   "exact_r2_1m": exact1m, "hashed_r2_1m": hash1m}
    outcome = decide(best_r2, g["wiki"]["G_pp"], exact1m, hash1m, router_best,
                     max(fin[str(li)][c] for li in D.layers for c in ("backoff", "aread")))
    if outcome == "INVESTIGATE":
        arr = np.load(data / "c4_investigate.npy")
        ev = D.prep_eval("c4_investigate", arr)
        r2i = D.r2(T, ev)
        bi = max(r2i[li][c] for li in D.layers for c in ("backoff", "aread"))
        D.dense(); pd_ = D.ppl(arr); D.healed(); ph = D.ppl(arr)
        pi = D.ppl(arr, best_li, D.add_fn(T, li_idx[best_li], best_li, best_col))
        gi = 100 * (ph - pi) / (ph - pd_)
        res["investigate"] = {"r2": bi, "G_pp": gi}
        outcome = "GO (after INVESTIGATE)" if bi >= 0.05 and gi >= 5 else "KILL (after INVESTIGATE)"
    res["outcome"] = outcome
    # fill-size suggestion: smallest snapshot reaching 80% of the final best R^2
    sizes = sorted((s["tokens"], max(s["r2_wiki"][str(li)][c] for li in D.layers for c in ("backoff", "aread")))
                   for s in res["snapshots"].values())
    res["suggested_fill_tokens"] = next((t for t, r in sizes if r >= 0.8 * best_r2), sizes[-1][0])
    res["popqa_pilot"] = D.popqa_sd(data / "popqa_pilot.jsonl")
    res["total_h"] = round((time.time() - t0) / 3600, 3)
    with open(a.out, "a") as f:
        f.write(json.dumps(res) + "\n")
    C.log("OUTCOME:", outcome, "| total", res["total_h"], "h")


if __name__ == "__main__":
    main()
