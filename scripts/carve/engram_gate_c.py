"""Engram v2 A0c (finding F7): learned Engram-style gate (+ value projection) on frozen canonical n-gram tables.

Tables (canonical keys, from engram_diag_b) are filled with FineWeb fill tokens [0, --table-tokens). Small heads are
trained on fill tokens [--table-tokens, +--train-tokens) (never in the tables), then scored on held-out WikiText-2 / C4:
  gate          alpha * c_hat,  alpha = sigmoid(RMSNorm(h) . RMSNorm(W_K c_hat) / sqrt(Hd) + b), h = healed hidden state
  gatewv        alpha * (W_V c_hat + b_v), W_V initialised to identity
  gatewv-decoy  gatewv with c_hat from shuffled keys (the hidden state alone, no n-gram link)
Injected G uses the live hidden state inside the forward pass, so it is what a deployed block would do.

  python engram_gate_c.py --model <dir> --out runs/a0c/a0c.jsonl [--smoke]
"""
import argparse, json, math, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import carve_proto as C
import engram_diag as E
import engram_diag_b as B

COLS = ("aread", "backoff")
ARMS = ("gate", "gatewv", "gatewv-decoy")


class Head(torch.nn.Module):
    def __init__(self, Hd, r, wv):
        super().__init__()
        self.Hd = Hd
        self.wk = torch.nn.Linear(r, Hd, bias=False)
        torch.nn.init.normal_(self.wk.weight, std=r ** -0.5)
        self.b = torch.nn.Parameter(torch.zeros(1))
        self.wv = None
        if wv:
            self.wv = torch.nn.Linear(r, r)
            torch.nn.init.eye_(self.wv.weight); torch.nn.init.zeros_(self.wv.bias)

    def forward(self, h, chat):
        q = F.rms_norm(h, (self.Hd,))
        k = F.rms_norm(self.wk(chat), (self.Hd,))
        a = torch.sigmoid((q * k).sum(-1, keepdim=True) / math.sqrt(self.Hd) + self.b)
        v = self.wv(chat) if self.wv is not None else chat
        return a * v, a


class DiagC(B.DiagB):
    rec = None

    def _cap_hook(self, li):                      # like E.Diag's, but the injection sees the live hidden state
        def fh(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if self.inject is not None and self.inject[0] == li:
                h = h + self.inject[1](h).to(h.dtype)
                out = (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
            self.cap[li] = h
            return out
        return fh

    @torch.no_grad()
    def drift(self, ids):
        d, rk = super().drift(ids)                # healed pass runs last, so self.cap holds the healed hidden states
        if self.rec is not None:
            self.rec.append({li: self.cap[li].reshape(-1, self.Hd).to(torch.bfloat16) for li in self.layers})
        return d, rk

    def prep_eval_c(self, name, arr):
        self.rec = []
        ev = self.prep_eval_b(name, arr)
        ev["h"] = {li: torch.cat([r[li] for r in self.rec]) for li in self.layers}
        self.rec = None
        return ev


def predict_arm(head, h, chat, chunk=8192):
    out = []
    with torch.no_grad():
        for s in range(0, h.shape[0], chunk):
            out.append(head(h[s:s + chunk].float(), chat[s:s + chunk])[0])
    return torch.cat(out)


def block_sse(ev, li, p):
    c = ev["c"][li].double()
    sse = ev["resid"][li] + ((c - p.double()) ** 2).sum(1)
    return torch.zeros(ev["nb"], device="cuda", dtype=torch.float64).index_add_(0, ev["blk"], sse)


@torch.no_grad()
def ppl(D, arr, li=None, fn=None):
    D.stopping = False
    nll, n, seq = 0.0, 0, D.a.seq
    for s in range(len(arr) // seq):
        ids = torch.from_numpy(arr[s * seq:(s + 1) * seq].astype(np.int64)).view(1, seq).cuda()
        D.inject = (li, lambda h, ids=ids: fn(ids, h)) if fn is not None else None
        out = D.model(input_ids=ids, labels=ids, use_cache=False)
        nll += out.loss.float().item() * (seq - 1); n += seq - 1
    D.inject, D.stopping = None, True
    return math.exp(nll / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--healed", default=str(E.ROOT / "cache" / "healed-Qwen3.5-2B-0.25k4.pt"))
    ap.add_argument("--config", default="0.25:4")
    ap.add_argument("--data", default=str(E.DATA))
    ap.add_argument("--layers", default="0,1")
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--bits", type=int, default=19)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--lam", type=float, default=8.0)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--table-tokens", type=int, default=18_000_000)
    ap.add_argument("--train-tokens", type=int, default=2_000_000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    a.trim, a.plant, a.stage = False, 0.0, "a0c"
    if a.smoke:
        a.table_tokens, a.train_tokens, a.bits, a.bs, a.epochs = 32768, 16384, 12, 2, 1
    t0 = time.time()
    data = Path(a.data)
    basis = np.load(data / "fineweb_basis.npy")
    fill = np.load(data / "fineweb_fill.npy")
    table_arr, train_arr = fill[: a.table_tokens], fill[a.table_tokens: a.table_tokens + a.train_tokens]
    evs_np = {"wiki": np.load(data / "wiki_eval.npy"), "c4": np.load(data / "c4_eval.npy")}
    if a.smoke:
        basis = basis[:4096]; evs_np = {k: v[:4096] for k, v in evs_np.items()}
    D = DiagC(a)
    L = D.layers
    C.log(f"A0c: layers {L}, canonical {D.canon_stats}, table tokens {len(table_arr)}, train tokens {len(train_arr)}")
    D.basis(basis)
    T = E.Tables(len(L), D.V, a.rank, a.bits, a.heads, "cuda")
    C.log(f"tables {T.gb():.2f} GB, GPU {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    done, tf = 0, time.time()
    for ids in D.batches(table_arr):
        d, _ = D.drift(ids)
        rows = T.rows(*D.keys(ids, "canon"))
        for i, li in enumerate(L):
            T.add(i, rows, D.codes(d[li], li), None)
        done += ids.numel()
        if done % (1 << 22) < ids.numel():
            C.log(f"table fill {done}/{len(table_arr)} tok, {done / (time.time() - tf):.0f} tok/s")

    heads = {(li, col, arm): Head(D.Hd, a.rank, arm != "gate").cuda() for li in L for col in COLS for arm in ARMS}
    opts = {k: torch.optim.Adam(h.parameters(), lr=a.lr) for k, h in heads.items()}
    g = torch.Generator(device="cuda"); g.manual_seed(0)
    hist = []
    for ep in range(a.epochs):
        tot = {k: 0.0 for k in list(heads) + [(li, col, "ungated") for li in L for col in COLS]}
        nb = 0
        for ids in D.batches(train_arr):
            d, _ = D.drift(ids)
            rows = T.rows(*D.keys(ids, "canon"))
            n = ids.numel()
            perm = torch.randperm(n, generator=g, device="cuda")
            for i, li in enumerate(L):
                c = D.codes(d[li], li)
                h = D.cap[li].reshape(-1, D.Hd).float()
                preds = T.predict(i, rows, a.lam, None)
                for col in COLS:
                    chat = preds[col]
                    tot[(li, col, "ungated")] += float(((c - chat) ** 2).sum(1).mean())
                    for arm in ARMS:
                        out, _ = heads[(li, col, arm)](h, chat[perm] if arm.endswith("decoy") else chat)
                        loss = ((c - out) ** 2).sum(1).mean()
                        opts[(li, col, arm)].zero_grad(set_to_none=True); loss.backward(); opts[(li, col, arm)].step()
                        tot[(li, col, arm)] += float(loss.detach())
            nb += 1
        ep_loss = {f"{k[0]}|{k[1]}|{k[2]}": v / max(nb, 1) for k, v in tot.items()}
        hist.append(ep_loss)
        C.log(f"epoch {ep + 1}: mean code MSE " + json.dumps({k: round(v, 4) for k, v in ep_loss.items()}))
    if len(hist) > 1 and all(hist[-1][k] >= hist[0][k] for k in hist[0] if not k.endswith("ungated")):
        C.log("STOP: training loss did not decrease for any arm")

    gb = torch.Generator(device="cuda"); gb.manual_seed(0)
    res = {"layers": L, "bits": a.bits, "table_tokens": len(table_arr), "train_tokens": len(train_arr), "epochs": a.epochs,
           "lr": a.lr, "canon": D.canon_stats, "basis_keep": D.keep, "train_loss": hist, "eval": {}}
    for name, arr in evs_np.items():
        ev = D.prep_eval_c(name, arr)
        idx = torch.randint(0, ev["nb"], (1000, ev["nb"]), generator=gb, device="cuda")
        perm = torch.randperm(ev["n"], generator=gb, device="cuda")
        rows = T.rows(*ev["keys"]["canon"])
        out, blocks = {}, {}
        for i, li in enumerate(L):
            sst = ev["sst_b"][li]
            preds = T.predict(i, rows, a.lam, None)
            for col in COLS:
                chat = preds[col]
                ps = {"ungated": chat}
                for arm in ARMS:
                    ps[arm] = predict_arm(heads[(li, col, arm)], ev["h"][li], chat[perm] if arm.endswith("decoy") else chat)
                for arm, p in ps.items():
                    sb = block_sse(ev, li, p)
                    blocks[(li, col, arm)] = sb
                    bs = 1 - sb[idx].sum(1) / sst[idx].sum(1)
                    out[f"{li}|{col}|{arm}"] = {"r2": 1 - float(sb.sum() / sst.sum()),
                                                "lo": float(bs.quantile(0.025)), "hi": float(bs.quantile(0.975))}
                a_, b_ = blocks[(li, col, "gatewv")], blocks[(li, col, "gatewv-decoy")]
                dd = (b_[idx].sum(1) - a_[idx].sum(1)) / sst[idx].sum(1)
                out[f"{li}|{col}|decoy_excess"] = {"d": float((b_.sum() - a_.sum()) / sst.sum()),
                                                   "lo": float(dd.quantile(0.025)), "hi": float(dd.quantile(0.975))}
        res["eval"][name] = out
        C.log(f"{name}: " + json.dumps({k: round(v.get('r2', v.get('d')), 4) for k, v in out.items()}))
        del ev
        torch.cuda.empty_cache()

    with open(a.out + ".partial", "a") as f:                           # R² results survive a crash in the ppl step
        f.write(json.dumps(res) + chr(10))
    w = res["eval"]["wiki"]
    best = max((w[f"{li}|{col}|{arm}"]["r2"], li, col, arm) for li in L for col in COLS for arm in ("gate", "gatewv"))
    r2b, li, col, arm = best
    t_idx = L.index(li)
    head = heads[(li, col, arm)]

    def make_fn(use_head):
        def fn(ids, h):
            rows = T.rows(*D.keys(ids, "canon"))
            chat = T.predict(t_idx, rows, a.lam, None)[col]
            p = head(h.reshape(-1, D.Hd).float(), chat)[0] if use_head else chat
            return (D.mu[li] + p @ D.U[li].T).view(ids.shape[0], ids.shape[1], -1)
        return fn

    gap = {}
    with torch.no_grad():
        for name, arr in evs_np.items():
            D.dense(); pd_ = ppl(D, arr)
            D.healed(); ph = ppl(D, arr)
            pl = ppl(D, arr, li, make_fn(True)); pu = ppl(D, arr, li, make_fn(False))
            gap[name] = {"ppl_dense": pd_, "ppl_healed": ph, "ppl_learned": pl, "ppl_ungated": pu,
                         "G_learned_pp": 100 * (ph - pl) / (ph - pd_), "G_ungated_pp": 100 * (ph - pu) / (ph - pd_)}
            C.log(f"{name}: dense {pd_:.3f} healed {ph:.3f} learned {pl:.3f} (G {gap[name]['G_learned_pp']:.2f} pp) "
                  f"ungated {pu:.3f} (G {gap[name]['G_ungated_pp']:.2f} pp) [{arm} L{li} {col}]")
    res["best"] = {"r2_wiki": r2b, "layer": li, "col": col, "arm": arm,
                   "decoy_excess": w[f"{li}|{col}|decoy_excess"]}
    res["gap"] = gap
    passed = r2b >= 0.05 and gap["wiki"]["G_learned_pp"] >= 5.0
    if passed and not a.smoke:
        arr = np.load(data / "c4_investigate.npy")
        ev = D.prep_eval_c("c4_investigate", arr)
        rows = T.rows(*ev["keys"]["canon"])
        chat = T.predict(t_idx, rows, a.lam, None)[col]
        sb = block_sse(ev, li, predict_arm(head, ev["h"][li], chat))
        r2i = 1 - float(sb.sum() / ev["sst_b"][li].sum())
        del ev
        D.dense(); pd_ = ppl(D, arr); D.healed(); ph = ppl(D, arr); pl = ppl(D, arr, li, make_fn(True))
        res["investigate"] = {"r2": r2i, "G_pp": 100 * (ph - pl) / (ph - pd_)}
        ok = r2i >= 0.05 and res["investigate"]["G_pp"] >= 5.0
        rec = f"GO to stage A ({arm} L{li} {col})" if ok else "KILL (did not replicate on c4_investigate)"
    elif r2b >= 0.05:
        rec = "KILL (predictable but does not move perplexity)"
    else:
        rec = "KILL: Engram v2 closes as a negative finding"
    res["recommendation"] = rec
    res["total_h"] = round((time.time() - t0) / 3600, 3)
    with open(a.out, "a") as f:
        f.write(json.dumps(res) + "\n")
    C.log("RECOMMENDATION:", rec)
    C.log("DONE | total", res["total_h"], "h")


if __name__ == "__main__":
    main()
