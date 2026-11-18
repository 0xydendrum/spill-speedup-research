"""Engram v2 A0b (finding F7): does a context gate, canonical token keys or an early layer rescue the n-gram memory?

Reuses engram_diag.py (model, hooks, drift, rank-128 basis, hashed tables, R² definition). Adds:
  keysets   raw token IDs (A0) and canonical IDs (NFKC + lowercase + strip; whitespace-only -> " " or "\\n";
            special and partial-UTF-8 tokens kept), one table set each
  gates     oracle clip gate alpha = clip(<c, c_hat>/||c_hat||^2, 0, 1) and unrestricted projection gate, per position
  decoy     shuffled keys: each eval position gets another position's rows (fixed permutation, seed 0)
  plant     positive control: random rank-128-span vector keyed on canonical ID, share of drift variance --plant
  CI        block bootstrap over eval sequences (1,000 resamples), paired for differences

  python engram_diag_b.py --model <dir> --stage smoke|plant|gate|pod --out <jsonl> [--plant-result <jsonl>] ...
"""
import argparse, json, math, time, unicodedata
from pathlib import Path

import numpy as np
import torch

import carve_proto as C
import engram_diag as E

KEYSETS = ("raw", "canon")
NG = ("backoff", "aread")
GATES = ("ungated", "gateclip", "gateproj")


def canon_map(tok, V):
    cmap = np.arange(V, dtype=np.int64)
    first, special = {}, set(tok.all_special_ids)
    for i in range(len(tok)):
        if i in special:
            continue
        s = tok.decode([i])
        if "�" in s:                                              # partial UTF-8 byte tokens: keep distinct
            continue
        key = ("\n" if "\n" in s else " ") if s.strip() == "" else unicodedata.normalize("NFKC", s).lower().strip()
        cmap[i] = first.setdefault(key, i)
    return cmap


class DiagB(E.Diag):
    def __init__(self, a):
        super().__init__(a)
        n = len(self.tok)
        self.cmap = torch.from_numpy(canon_map(self.tok, self.V)).cuda()
        self.canon_stats = {"raw_ids": n, "canon_ids": int(torch.unique(self.cmap[:n]).numel())}
        self.canon_stats["reduction"] = round(1 - self.canon_stats["canon_ids"] / n, 4)
        self.plant = None
        if a.trim:                                  # no ppl evals: layers after the last probe never run
            for i in range(self.stop_at + 1, len(self.dec)):
                self.dec[i].to("cpu"); self.mods[i].to("cpu")
            torch.cuda.empty_cache()

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
        self.mu, self.U, self.keep, self.var_tot = {}, {}, {}, {}
        for li in self.layers:
            mu = s1[li] / n
            ev, evec = torch.linalg.eigh(s2[li] / n - torch.outer(mu, mu))
            self.mu[li] = mu.float()
            self.U[li] = evec[:, -r:].flip(1).float().contiguous()
            self.keep[li] = float(ev[-r:].sum() / ev.clamp(min=0).sum())
            self.var_tot[li] = float(ev.clamp(min=0).sum())
        C.log("basis: variance kept by rank", r, {li: round(v, 4) for li, v in self.keep.items()})

    def set_plant(self, share, seed=1):
        g = torch.Generator(device="cuda"); g.manual_seed(seed)
        self.plant = {}
        for li in self.layers:
            sig2 = share * self.var_tot[li] / (1 - share) / self.a.rank
            self.plant[li] = torch.randn(self.V, self.a.rank, generator=g, device="cuda") * math.sqrt(sig2)

    @torch.no_grad()
    def drift(self, ids):
        d, rk = super().drift(ids)
        if self.plant is not None:
            cid = self.cmap[ids.reshape(-1)]
            for li in self.layers:
                d[li] = d[li] + self.plant[li][cid] @ self.U[li].T
        return d, rk

    def keys(self, ids, ks):
        return E.ngram_keys(ids if ks == "raw" else self.cmap[ids], self.V)

    @torch.no_grad()
    def prep_eval_b(self, name, arr):
        seq, Hd = self.a.seq, self.Hd
        acc = {k: {li: [] for li in self.layers} for k in ("c", "q", "rk", "s1")}
        toks = []
        for ids in self.batches(arr):
            d, r = self.drift(ids)
            for li in self.layers:
                x = d[li] - self.mu[li]
                c = x @ self.U[li]
                acc["c"][li].append(c); acc["q"][li].append((x * x).sum(1)); acc["rk"][li].append(r[li])
                acc["s1"][li].append(x.view(-1, seq, Hd).double().sum(1))
            toks.append(ids)
        tok = torch.cat(toks)
        n = tok.numel()
        blk = torch.arange(n, device="cuda") // seq
        nb = int(blk.max()) + 1
        ev = {"name": name, "n": n, "blk": blk, "nb": nb, "tok": tok,
              "keys": {ks: self.keys(tok, ks) for ks in KEYSETS}, "c": {}, "resid": {}, "rk": {}, "sst_b": {}}
        if self.plant is not None:
            cid = self.cmap[tok.reshape(-1)]
            ev["plant_energy"] = {li: float((self.plant[li][cid] ** 2).sum()) for li in self.layers}
        for li in self.layers:
            c = torch.cat(acc["c"][li]); q = torch.cat(acc["q"][li]).double()
            s1b = torch.cat(acc["s1"][li])                                  # (nb, Hd) sums of x per sequence
            m = s1b.sum(0) / n
            qb = torch.zeros(nb, device="cuda", dtype=torch.float64).index_add_(0, blk, q)
            ev["sst_b"][li] = qb - 2 * (s1b @ m) + seq * float(m @ m)      # sum ||x - m||^2 per sequence
            ev["c"][li] = c
            ev["resid"][li] = (q - (c.double() ** 2).sum(1))                # out-of-basis energy per position
            ev["rk"][li] = torch.cat(acc["rk"][li])
        C.log(f"eval {name}: {n} tokens cached ({nb} blocks)")
        return ev


def per_pos_sse(resid, c, p):
    """Per-position SSE (float64) for ungated, clip-gated and projection-gated predictions p of codes c."""
    c, p = c.double(), p.double()
    cc, dot, pp = (c * c).sum(1), (c * p).sum(1), (p * p).sum(1)
    a_proj = torch.where(pp > 0, dot / pp.clamp(min=1e-30), torch.zeros_like(dot))
    a_clip = a_proj.clamp(0, 1)
    return {"ungated": resid + cc - 2 * dot + pp,
            "gateclip": resid + cc - 2 * a_clip * dot + a_clip ** 2 * pp,
            "gateproj": resid + cc - a_proj * dot}


def score(D, T, ev, boot=True, nboot=1000):
    """R² (+ block-bootstrap 95% CI) for every keyset x layer x column x gate, real and shuffled-key decoy."""
    g = torch.Generator(device="cuda"); g.manual_seed(0)
    perm = torch.randperm(ev["n"], generator=g, device="cuda")
    idx = torch.randint(0, ev["nb"], (nboot, ev["nb"]), generator=g, device="cuda") if boot else None
    out, blocks, sanity = {}, {}, []
    for ks in KEYSETS:
        rows = T[ks].rows(*ev["keys"][ks])
        rows_d = [r[perm] for r in rows]
        for i, li in enumerate(D.layers):
            rk = ev["rk"][li] if ks == "raw" else None
            preds = {"real": T[ks].predict(i, rows, D.a.lam, rk),
                     "decoy": T[ks].predict(i, rows_d, D.a.lam, rk[perm] if rk is not None else None)}
            sst_b = ev["sst_b"][li]
            for kind, pr in preds.items():
                for col, p in pr.items():
                    sse = per_pos_sse(ev["resid"][li], ev["c"][li], p)
                    s = {gname: float(v.sum()) for gname, v in sse.items()}
                    sanity.append(s["gateclip"] <= s["ungated"] * (1 + 1e-9) + 1e-6 and s["gateproj"] <= s["gateclip"] * (1 + 1e-9) + 1e-6)
                    for gname in GATES if col in NG else ("ungated",):
                        sb = torch.zeros(ev["nb"], device="cuda", dtype=torch.float64).index_add_(0, ev["blk"], sse[gname])
                        key = f"{ks}|{li}|{col}|{gname}|{kind}"
                        blocks[key] = sb
                        rec = {"r2": 1 - float(sb.sum() / sst_b.sum())}
                        if boot:
                            bs = 1 - sb[idx].sum(1) / sst_b[idx].sum(1)
                            rec["lo"], rec["hi"] = float(bs.quantile(0.025)), float(bs.quantile(0.975))
                        out[key] = rec
    res = {"r2": out, "sanity_ok": all(sanity)}
    if boot:                                                            # paired differences
        diffs = {}
        for ks in KEYSETS:
            for li in D.layers:
                sst = ev["sst_b"][li][idx].sum(1)
                for col in NG:
                    for gname in GATES:
                        a, b = blocks[f"{ks}|{li}|{col}|{gname}|real"], blocks[f"{ks}|{li}|{col}|{gname}|decoy"]
                        dd = (b[idx].sum(1) - a[idx].sum(1)) / sst
                        diffs[f"excess|{ks}|{li}|{col}|{gname}"] = {"d": float((b.sum() - a.sum()) / ev["sst_b"][li].sum()),
                                                                    "lo": float(dd.quantile(0.025)), "hi": float(dd.quantile(0.975))}
                    a, b = blocks[f"raw|{li}|{col}|ungated|real"], blocks[f"canon|{li}|{col}|ungated|real"]
                    dd = (a[idx].sum(1) - b[idx].sum(1)) / sst
                    diffs[f"canon_minus_raw|{li}|{col}"] = {"d": float((a.sum() - b.sum()) / ev["sst_b"][li].sum()),
                                                            "lo": float(dd.quantile(0.025)), "hi": float(dd.quantile(0.975))}
        res["diffs"] = diffs
    return res


def summarize(sc, layers):
    """Headline numbers the preregistered rules use."""
    r2 = sc["r2"]
    ung = [(r2[f"{ks}|{li}|{col}|ungated|real"]["r2"], ks, li, col) for ks in KEYSETS for li in layers for col in NG]
    best = max(ung)
    if "diffs" in sc:
        exc = [(sc["diffs"][f"excess|{ks}|{li}|{col}|gateclip"]["d"], ks, li, col) for ks in KEYSETS for li in layers for col in NG]
    else:
        exc = [(r2[f"{ks}|{li}|{col}|gateclip|real"]["r2"] - r2[f"{ks}|{li}|{col}|gateclip|decoy"]["r2"], ks, li, col)
               for ks in KEYSETS for li in layers for col in NG]
    decoy_max = max(v["r2"] for k, v in r2.items() if k.endswith("|ungated|decoy"))
    return {"best_ungated": {"r2": best[0], "keyset": best[1], "layer": best[2], "col": best[3]},
            "best_gate_excess": {"d": max(exc)[0], "keyset": max(exc)[1], "layer": max(exc)[2], "col": max(exc)[3]},
            "decoy_ungated_max": decoy_max, "sanity_ok": sc["sanity_ok"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--healed", default=str(E.ROOT / "cache" / "healed-Qwen3.5-2B-0.25k4.pt"))
    ap.add_argument("--config", default="0.25:4")
    ap.add_argument("--data", default=str(E.DATA))
    ap.add_argument("--stage", choices=("smoke", "plant", "gate", "pod"), required=True)
    ap.add_argument("--layers", default="0,1,2,3")
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--bits", type=int, default=19)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--lam", type=float, default=8.0)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--fill-tokens", type=int, default=20_000_000)
    ap.add_argument("--snapshots", default="1000000,2000000,5000000")
    ap.add_argument("--max-hours", type=float, default=6.0)
    ap.add_argument("--plant", type=float, default=0.10)
    ap.add_argument("--plant-result", default=None, help="gate stage: the plant run's jsonl, for plant_recovery")
    ap.add_argument("--allow-investigate", action="store_true", help="pod 'full' scope: may read c4_investigate")
    ap.add_argument("--trim", action="store_true", help="move layers after the last probe to CPU (no ppl evals)")
    ap.add_argument("--smoke-ppl", action="store_true", help="smoke stage: also run the pod G-injection path on the smoke slices")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.stage == "smoke":
        a.fill_tokens, a.snapshots, a.bits, a.bs, a.layers = 16384, "8192", 12, 2, "2,3"
    if a.stage in ("smoke", "plant", "gate") and not a.smoke_ppl:
        a.trim = True
    t0 = time.time()
    data = Path(a.data)
    basis = np.load(data / "fineweb_basis.npy")
    fill = np.load(data / "fineweb_fill.npy")[: a.fill_tokens]
    evs_np = {"wiki": np.load(data / "wiki_eval.npy"), "c4": np.load(data / "c4_eval.npy")}
    if a.stage == "smoke":
        basis = basis[:4096]
        evs_np = {k: v[:4096] for k, v in evs_np.items()}
    D = DiagB(a)
    C.log(f"stage {a.stage}: model loaded, V={D.V}, probe layers {D.layers}, canonical {D.canon_stats}, "
          f"GPU {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    D.basis(basis)
    if a.stage == "plant":
        D.set_plant(a.plant)
        C.log(f"plant: share {a.plant} of drift variance, keyed on canonical id")
    evs = {k: D.prep_eval_b(k, v) for k, v in evs_np.items()}
    T = {ks: E.Tables(len(D.layers), D.V, a.rank, a.bits, a.heads, "cuda") for ks in KEYSETS}
    C.log(f"tables: {sum(t.gb() for t in T.values()):.2f} GB, GPU {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    res = {"stage": a.stage, "layers": D.layers, "bits": a.bits, "heads": a.heads, "rank": a.rank, "lam": a.lam,
           "canon": D.canon_stats, "basis_keep": D.keep, "snapshots": {}}
    snaps = sorted(int(x) for x in a.snapshots.split(",") if x)
    done, tfill, truncated = 0, time.time(), False
    for ids in D.batches(fill):
        d, rk = D.drift(ids)
        rows = {ks: T[ks].rows(*D.keys(ids, ks)) for ks in KEYSETS}
        for i, li in enumerate(D.layers):
            c = D.codes(d[li], li)
            T["raw"].add(i, rows["raw"], c, rk[li])
            T["canon"].add(i, rows["canon"], c, None)
        prev, done = done, done + ids.numel()
        for sn in snaps:
            if prev < sn <= done:
                sm = summarize(score(D, T, evs["wiki"], boot=False), D.layers)
                res["snapshots"][str(sn)] = sm
                C.log(f"snapshot {sn}: {json.dumps(sm)}")
        if done % (1 << 20) < ids.numel():
            rate = done / (time.time() - tfill)
            C.log(f"fill {done}/{len(fill)} tok, {rate:.0f} tok/s, eta {(len(fill) - done) / rate / 3600:.2f} h, "
                  f"GPU peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        if time.time() - t0 > a.max_hours * 3600:
            truncated = True; C.log("max-hours reached, stopping the fill"); break
    res["fill_tokens"], res["truncated"] = done, truncated
    res["fill_tok_per_s"] = round(done / (time.time() - tfill), 1)
    res["gpu_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    full = {name: score(D, T, ev) for name, ev in evs.items()}
    res["final"] = {name: {"summary": summarize(sc, D.layers), "r2": sc["r2"], "diffs": sc["diffs"]} for name, sc in full.items()}
    sw = res["final"]["wiki"]["summary"]
    C.log(f"final wiki: {json.dumps(sw)}")
    C.log(f"final c4: {json.dumps(res['final']['c4']['summary'])}")

    if a.stage == "plant":
        ev = evs["wiki"]
        rec = {}
        for ks in KEYSETS:
            best = max((res["final"]["wiki"]["r2"][f"{ks}|{li}|{col}|ungated|real"]["r2"], li) for li in D.layers for col in NG)
            share = ev["plant_energy"][best[1]] / float(ev["sst_b"][best[1]].sum())
            rec[ks] = {"r2": best[0], "planted_share_eval": share, "recovery": best[0] / share}
        res["plant_recovery"] = rec
        C.log(f"plant recovery: {json.dumps(rec)}")

    if a.stage in ("gate", "smoke"):
        plant = None
        if a.plant_result:
            plant = json.loads(Path(a.plant_result).read_text().strip().splitlines()[-1])["plant_recovery"]
        fails = []
        if plant is None:
            fails.append("no plant result")
        elif min(v["recovery"] for v in plant.values()) < 0.50:
            fails.append(f"plant_recovery {min(v['recovery'] for v in plant.values()):.3f} < 0.50")
        if sw["decoy_ungated_max"] > 0.005:
            fails.append(f"decoy ungated R² {sw['decoy_ungated_max']:.4f} > 0.005")
        if not all(f["summary"]["sanity_ok"] for f in res["final"].values()):
            fails.append("gate sanity check failed")
        if fails:
            outcome = "CONTROL_FAIL: " + "; ".join(fails)
        elif sw["best_ungated"]["r2"] >= 0.02 or sw["best_gate_excess"]["d"] >= 0.02:
            outcome = "PROMISING"
        else:
            outcome = "NULL"
        res["gate_outcome"] = outcome
        C.log("GATE_OUTCOME:", outcome)

    with open(a.out + ".partial", "a") as f:                           # R² results survive a crash in the G step
        f.write(json.dumps(res) + "\n")
    if a.stage == "pod" or a.smoke_ppl:
        b = sw["best_ungated"]
        ks, li, col = b["keyset"], b["layer"], b["col"]
        t_idx = D.layers.index(li)
        g = {}
        for name, arr in evs_np.items():
            D.dense(); pd_ = D.ppl(arr)
            D.healed(); ph = D.ppl(arr)
            pi = D.ppl(arr, li, D_add(D, T[ks], t_idx, li, col, ks))
            g[name] = {"ppl_dense": pd_, "ppl_healed": ph, "ppl_injected": pi, "G_pp": 100 * (ph - pi) / (ph - pd_)}
            C.log(f"{name}: dense {pd_:.3f} healed {ph:.3f} injected {pi:.3f} G {g[name]['G_pp']:.2f} pp ({ks} L{li} {col})")
        res["gap"] = g
        exc = sw["best_gate_excess"]
        inv = None
        wiki_pass = b["r2"] >= 0.05 and g["wiki"]["G_pp"] >= 5.0
        gate_pass = b["r2"] < 0.05 and exc["d"] >= 0.05
        if (wiki_pass or gate_pass) and a.allow_investigate:
            arr = np.load(data / "c4_investigate.npy")
            evi = D.prep_eval_b("c4_investigate", arr)
            sci = score(D, T, evi)
            inv = {"summary": summarize(sci, D.layers), "r2": sci["r2"]}
            if wiki_pass:
                D.dense(); pd_ = D.ppl(arr); D.healed(); ph = D.ppl(arr)
                pi = D.ppl(arr, li, D_add(D, T[ks], t_idx, li, col, ks))
                inv["arm"] = {"r2": sci["r2"][f"{ks}|{li}|{col}|ungated|real"]["r2"], "G_pp": 100 * (ph - pi) / (ph - pd_)}
            else:
                inv["gate_excess"] = sci["diffs"][f"excess|{exc['keyset']}|{exc['layer']}|{exc['col']}|gateclip"]
            C.log(f"c4_investigate: {json.dumps(inv.get('arm') or inv.get('gate_excess'))}")
        res["investigate"] = inv
        if wiki_pass and inv and "arm" in inv:
            ok = inv["arm"]["r2"] >= 0.05 and inv["arm"]["G_pp"] >= 5.0
            rec = "GO to stage A (" + f"{ks} L{li} {col})" if ok else "KILL (wiki pass did not replicate on c4_investigate)"
        elif wiki_pass:
            rec = "wiki bars passed; c4_investigate not read in this scope -> owner decides on confirmation"
        elif b["r2"] >= 0.05:
            rec = "KILL (predictable but does not move perplexity)"
        elif gate_pass:
            rec = "A0c: learned context gate on frozen tables (not GO)"
        else:
            rec = "KILL confirmed"
        res["recommendation"] = rec
        C.log("RECOMMENDATION:", rec)

    res["total_h"] = round((time.time() - t0) / 3600, 3)
    with open(a.out, "a") as f:
        f.write(json.dumps(res) + "\n")
    C.log("DONE | total", res["total_h"], "h")


def D_add(D, T, t_idx, li, col, ks):
    def f(ids):
        rows = T.rows(*D.keys(ids, ks))
        pred = T.predict(t_idx, rows, D.a.lam, None)[col]
        return (D.mu[li] + pred @ D.U[li].T).view(ids.shape[0], ids.shape[1], -1)
    return f


if __name__ == "__main__":
    main()
