"""Pre-trained Engram, part 2 (0xydendrum's idea, 2026-11-15): "take a snapshot of the pre-carved model, save it to the
Engram, and train the carved model off the Engram".

  carve -> layer-wise heal (cached) -> [snapshot the dense-minus-carved gap into n-gram tables] -> whole-model
  distillation with the memory attached and trainable (tables live in CPU RAM, updated sparsely with row-wise Adagrad).

Memory modes (same data, steps and carve for a fair comparison):
  none      plain distillation (more text than distill_proto.py's first run)
  snapshot  tables + projection + bias initialised from the snapshot (0xydendrum's idea)
  zero      same memory size, tables start empty and projection random (control: is the snapshot itself worth it?)

Per layer l, per token t with keys k1 (token), k2 (hashed 2-gram), k3 (hashed 3-gram):
  code = T1[l,k1] + T2[l,k2] + T3[l,k3]        (r numbers)
  out  = carved_ffn(x) + sigmoid(x.w_l + b_l) * (mu_l + U_l code)

  python distill_engram.py --model path/to/Qwen3.5-2B --memory snapshot
"""
import argparse, json, math, time
from pathlib import Path

import torch
import torch.nn.functional as F

import carve_proto as C
import heal_proto as H
from distill_proto import teacher_topk
from engram_snapshot import keys

OUT = Path(__file__).resolve().parent


class Memory:
    """n-gram memory: tables on CPU (fp32), dense parts (U, mu, gate) on GPU and trained with AdamW."""

    def __init__(self, L, H, V, r, b2, b3, dev):
        self.L, self.H, self.V, self.r, self.b2, self.b3 = L, H, V, r, b2, b3
        self.T = [torch.zeros(L, n, r) for n in (V, b2, b3)]
        self.acc = [torch.zeros(L, n) for n in (V, b2, b3)]          # row-wise Adagrad state
        self.U = torch.nn.Parameter(torch.zeros(L, H, r, device=dev))
        self.mu = torch.nn.Parameter(torch.zeros(L, H, device=dev))
        self.gw = torch.nn.Parameter(torch.zeros(L, H, device=dev))
        self.gb = torch.nn.Parameter(torch.full((L,), math.log(0.75 / 0.25), device=dev))
        self.cur = None
        self.on = True

    def params(self):
        return [self.U, self.mu, self.gw, self.gb]

    def prepare(self, ids, train):
        """Gather the table rows this batch needs onto the GPU (as leaf tensors when training)."""
        ks = keys(ids, self.V, self.b2, self.b3)
        cur = []
        for h, k in enumerate(ks):
            uk, inv = torch.unique(k, return_inverse=True)
            rows = self.T[h][:, uk].cuda(non_blocking=True)            # (L, U, r)
            if train:
                rows.requires_grad_(True)
            cur.append((uk, inv.cuda(non_blocking=True), rows))
        self.cur = cur

    def corr(self, i, x2):
        code = sum(rows[i][inv] for _, inv, rows in self.cur)            # (N, r)
        c = self.mu[i] + code @ self.U[i].T
        g = torch.sigmoid(x2.float() @ self.gw[i] + self.gb[i]).unsqueeze(1)
        return g * c

    @torch.no_grad()
    def step_tables(self, lr, eps=1e-8):
        for h, (uk, _, rows) in enumerate(self.cur):
            if rows.grad is None:
                continue
            g = rows.grad.float().cpu()                                  # (L, U, r)
            a = self.acc[h][:, uk] + g.pow(2).mean(2)
            self.acc[h][:, uk] = a
            self.T[h][:, uk] -= lr * g / (a.sqrt().unsqueeze(2) + eps)
        self.cur = None


def install(mlp, mod, mem, i):
    def fwd(x):
        shp = x.shape
        x2 = x.reshape(-1, shp[-1]).to(mod.wg.dtype)
        out = mod(x2)
        if mem is not None and mem.on and mem.cur is not None:
            out = out + mem.corr(i, x2).to(out.dtype)
        return out.to(x.dtype).reshape(shp)
    mlp._orig_forward = getattr(mlp, "_orig_forward", mlp.forward)
    mlp.forward = fwd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--memory", choices=["none", "snapshot", "zero"], default="snapshot")
    ap.add_argument("--experts", type=int, default=32)
    ap.add_argument("--config", default="0.25:4")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--b2", type=int, default=1 << 19)
    ap.add_argument("--b3", type=int, default=1 << 19)
    ap.add_argument("--min-count", type=int, default=3)
    ap.add_argument("--cal-tokens", type=int, default=65536)
    ap.add_argument("--train-tokens", type=int, default=1 << 20)
    ap.add_argument("--snap-tokens", type=int, default=2_500_000)
    ap.add_argument("--basis-tokens", type=int, default=65536)
    ap.add_argument("--eval-tokens", type=int, default=16384)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--topk-teacher", type=int, default=32)
    ap.add_argument("--heal-steps", type=int, default=600)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lr-mem", type=float, default=1e-4, help="AdamW lr for U / mu / gate")
    ap.add_argument("--lr-table", type=float, default=0.05, help="Adagrad lr, as a fraction of the typical code size")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.cal_tokens, a.train_tokens, a.eval_tokens, a.heal_steps, a.steps, a.eval_every = 8192, 4096, 2048, 20, 10, 10
        a.snap_tokens, a.basis_tokens, a.b2, a.b3 = 16384, 8192, 1 << 12, 1 << 12
    torch.backends.cuda.matmul.allow_tf32 = True
    t0 = time.time()
    eseq = 1024 if not a.smoke else 512
    model, tok = C.load_model(a.model, "cuda")
    for p in model.parameters():
        p.requires_grad_(False)
    layers = C.mlps(model)
    L, Hd = len(layers), layers[0].gate_proj.in_features
    V = model.get_input_embeddings().weight.shape[0]
    sf, k = float(a.config.split(":")[0]), int(a.config.split(":")[1])
    r = a.rank
    cal = C.text_tokens(tok, "train", max(a.cal_tokens + a.train_tokens, a.snap_tokens))
    cal_carve, cal_train, snap = cal[: a.cal_tokens], cal[a.cal_tokens: a.cal_tokens + a.train_tokens], cal[: a.snap_tokens]
    ev = C.text_tokens(tok, "test", a.eval_tokens)
    mem = None if a.memory == "none" else Memory(L, Hd, V, r, a.b2, a.b3, "cuda")

    @torch.no_grad()
    def ppl():
        nll, n = 0.0, 0
        ch = [ev[i:i + eseq] for i in range(0, len(ev) - eseq + 1, eseq)]
        for j in range(0, len(ch), 4):
            ids = torch.stack(ch[j:j + 4])
            if mem is not None:
                mem.prepare(ids, train=False)
            o = model(input_ids=ids.cuda(), labels=ids.cuda())
            kk = ids.numel() - ids.shape[0]
            nll += o.loss.float().item() * kk; n += kk
        if mem is not None:
            mem.cur = None
        return math.exp(nll / n)

    base = ppl()
    C.log(f"dense perplexity {base:.3f}  ({L} layers, H={Hd}, vocab={V}, memory={a.memory})")
    chunks, tv, ti = teacher_topk(model, cal_train, a.seq, a.topk_teacher)
    C.log(f"teacher top-{a.topk_teacher} saved for {chunks.numel()} tokens")

    # ---- carve + layer-wise heal (cached so every memory mode starts from the identical carved model) ----
    cache = OUT / "cache" / f"healed-{Path(a.model).name}-{a.config.replace(':', 'k')}{'-smoke' if a.smoke else ''}.pt"
    if cache.exists():
        ck = torch.load(cache)
        carves = ck["carves"]
        mods = []
        for i, m in enumerate(layers):
            mod = H.CarvedFFN(m, carves[i], k).to(torch.bfloat16)
            mod.load_state_dict(ck["state"][i])
            mods.append(mod)
        C.log(f"loaded healed carve from {cache.name}")
    else:
        freq, xs, acts = C.collect(model, layers, cal_carve, eseq, "cuda", 8192 if not a.smoke else 1024)
        paths = H.dump_io(model, layers, cal_carve, eseq, min(1.0, 32768 / len(cal_carve)), "distill")
        carves = [C.carve_layer(m, freq[i], xs[i], acts[i], a.experts, sf) for i, m in enumerate(layers)]
        del xs, acts
        mods = []
        for i, m in enumerate(layers):
            mod = H.CarvedFFN(m, carves[i], k)
            if a.heal_steps:
                H.heal_layer(mod, paths[i], a.heal_steps, 1e-4, 1024)
            mods.append(mod.to(torch.bfloat16))
        cache.parent.mkdir(exist_ok=True)
        torch.save({"carves": [{kk: v.cpu() for kk, v in c.items()} for c in carves],
                    "state": [{kk: v.cpu() for kk, v in m_.state_dict().items()} for m_ in mods]}, cache)
        C.log(f"healed carve cached to {cache.name}")
    for p in (p for m_ in mods for p in m_.parameters()):
        p.requires_grad_(False)

    # ---- snapshot: what the dense layer produces that the healed carve misses, per n-gram key ----
    code_std = 1.0
    holder = {}
    snap_chunks = lambda ids, n: [ids[i:i + eseq] for i in range(0, min(len(ids), n) - eseq + 1, eseq)]
    if mem is not None:
        cov = torch.zeros(L, Hd, Hd, device="cuda"); s1 = torch.zeros(L, Hd, device="cuda"); cnt = [0]
        def hook_basis(i):
            def fh(m_, inp, out):
                x2 = inp[0].reshape(-1, Hd)
                gap = (out.reshape(-1, Hd).float() - mods[i](x2.to(torch.bfloat16)).float())
                cov[i] += gap.T @ gap; s1[i] += gap.sum(0)
                if i == 0: cnt[0] += gap.shape[0]
            return fh
        hs = [m.register_forward_hook(hook_basis(i)) for i, m in enumerate(layers)]
        with torch.no_grad():
            ch = snap_chunks(snap, a.basis_tokens)
            for j in range(0, len(ch), 4):
                model(input_ids=torch.stack(ch[j:j + 4]).cuda())
        for h in hs: h.remove()
        mu = s1 / cnt[0]
        U = torch.zeros(L, Hd, r, device="cuda"); keep, top_ev = [], []
        for i in range(L):
            ev_, evec = torch.linalg.eigh(cov[i] / cnt[0] - torch.outer(mu[i], mu[i]))
            U[i] = evec[:, -r:]
            keep.append((ev_[-r:].sum() / ev_.sum().clamp(min=1e-9)).item()); top_ev.append(ev_[-r:].mean().item())
        del cov
        code_std = math.sqrt(max(sum(top_ev) / L, 1e-12))
        C.log(f"gap basis vs healed carve: rank-{r} keeps {sum(keep)/L:.1%} of the gap variance; typical code size {code_std:.4f}")
        if a.memory == "snapshot":
            T1, T2, T3 = mem.T
            N = [torch.zeros(n) for n in (V, a.b2, a.b3)]
            def hook_snap(i):
                def fh(m_, inp, out):
                    x2 = inp[0].reshape(-1, Hd)
                    code = ((out.reshape(-1, Hd).float() - mods[i](x2.to(torch.bfloat16)).float()) - mu[i]) @ U[i]
                    code = code.cpu(); k1, k2, k3 = holder["keys"]
                    T1[i].index_add_(0, k1, code); T2[i].index_add_(0, k2, code); T3[i].index_add_(0, k3, code)
                return fh
            hs = [m.register_forward_hook(hook_snap(i)) for i, m in enumerate(layers)]
            with torch.no_grad():
                ch = snap_chunks(snap, a.snap_tokens)
                for j in range(0, len(ch), 4):
                    ids = torch.stack(ch[j:j + 4])
                    kk = keys(ids, V, a.b2, a.b3)
                    holder["keys"] = kk
                    for n_, k_ in zip(N, kk):
                        n_.index_add_(0, k_, torch.ones(k_.shape[0]))
                    model(input_ids=ids.cuda())
                    if (j // 4) % 150 == 0:
                        C.log(f"snapshot {j * eseq}/{len(ch) * eseq} tokens")
            for h in hs: h.remove()
            for T, n_ in zip(mem.T, N):
                T.div_(n_.clamp(min=1).view(1, -1, 1))
            # stack as residuals so code = T1+T2+T3 reproduces back-off (3-gram, else 2-gram, else 1-gram, else 0)
            ok1 = (N[0] >= a.min_count).view(1, -1, 1).float()
            ok2 = (N[1] >= a.min_count).view(1, -1, 1).float()
            ok3 = (N[2] >= a.min_count).view(1, -1, 1).float()
            T1.mul_(ok1)
            # residuals need the lower-order mean for the same position; approximate with the table means seen in the
            # snapshot text (exact per position is impossible with hashing), computed in one more pass below
            R2 = torch.zeros_like(T2); R3 = torch.zeros_like(T3)
            with torch.no_grad():
                for j in range(0, len(ch), 4):
                    ids = torch.stack(ch[j:j + 4])
                    k1, k2, k3 = keys(ids, V, a.b2, a.b3)
                    for i in range(L):
                        c1 = T1[i][k1]; c2 = T2[i][k2] * ok2[0, k2]
                        R2[i].index_add_(0, k2, c1); R3[i].index_add_(0, k3, torch.where(ok2[0, k2].bool(), c2, c1))
            R2.div_(N[1].clamp(min=1).view(1, -1, 1)); R3.div_(N[2].clamp(min=1).view(1, -1, 1))
            T3.sub_(R3).mul_(ok3); T2.sub_(R2).mul_(ok2)
            del R2, R3
            C.log(f"snapshot done: {int((N[0] >= a.min_count).sum())} tokens, {int((N[1] >= a.min_count).sum())} 2-gram and "
                  f"{int((N[2] >= a.min_count).sum())} 3-gram buckets with >= {a.min_count} hits")
            with torch.no_grad():
                mem.U.copy_(U); mem.mu.copy_(mu)
        else:  # zero: same size, no snapshot information (random orthonormal projection, empty tables, no bias)
            with torch.no_grad():
                g = torch.Generator(device="cuda").manual_seed(0)
                for i in range(L):
                    mem.U[i].copy_(torch.linalg.qr(torch.randn(Hd, r, device="cuda", generator=g))[0])
        del U

    # ---- swap in the carved layers (+ memory) and free the dense feed-forward weights ----
    for i, m in enumerate(layers):
        for lin in (m.gate_proj, m.up_proj, m.down_proj):
            lin.weight = torch.nn.Parameter(torch.empty(0, device="cuda", dtype=torch.bfloat16), requires_grad=False)
        install(m, mods[i], mem, i)
    torch.cuda.empty_cache()
    frac = (len(carves[0]["shared"]) + k * carves[0]["experts"].shape[1]) / mods[0].wg.shape[0]
    if mem is not None:
        mem.on = False
    ppl_heal = ppl()
    if mem is not None:
        mem.on = True
    ppl_start = ppl()
    C.log(f"[{a.config}] active {frac:.3f}  healed carve {ppl_heal:.3f}  with memory at start {ppl_start:.3f}  "
          f"GPU {torch.cuda.memory_allocated()/2**30:.1f} GiB")

    # ---- whole-model distillation, memory attached ----
    params = [p for m_ in mods for p in m_.parameters()]
    for p in params:
        p.requires_grad_(True)
    import bitsandbytes as bnb
    opt = bnb.optim.AdamW8bit(params, lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    opt_m = sched_m = None
    if mem is not None:
        opt_m = torch.optim.AdamW(mem.params(), lr=a.lr_mem, weight_decay=0.0)
        sched_m = torch.optim.lr_scheduler.CosineAnnealingLR(opt_m, a.steps)
    lr_tab = a.lr_table * code_std
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.train()
    gen = torch.Generator().manual_seed(0)
    hist = [(0, round(ppl_start, 4))]
    for s in range(a.steps):
        bi = torch.randint(0, len(chunks), (a.bs,), generator=gen)
        ids = chunks[bi]
        if mem is not None:
            mem.prepare(ids, train=True)
        x = ids.cuda()
        logits = model(input_ids=x, use_cache=False).logits.float()
        lp = torch.log_softmax(logits, -1)
        t_p = torch.softmax(tv[bi].cuda().float(), -1)
        kd = -(t_p * lp.gather(-1, ti[bi].cuda().long())).sum(-1).mean()
        ce = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1))
        loss = kd + 0.25 * ce
        opt.zero_grad(set_to_none=True)
        if opt_m: opt_m.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params + (mem.params() if mem else []), 1.0)
        opt.step(); sched.step()
        if mem is not None:
            opt_m.step(); sched_m.step()
            mem.step_tables(lr_tab * 0.5 * (1 + math.cos(math.pi * s / a.steps)))
        if (s + 1) % 50 == 0:
            C.log(f"step {s+1}/{a.steps} kd {kd.item():.4f} ce {ce.item():.4f}")
        if (s + 1) % a.eval_every == 0 or s == a.steps - 1:
            model.eval()
            p_ = ppl()
            model.train()
            hist.append((s + 1, round(p_, 4)))
            C.log(f"eval at step {s+1}: perplexity {p_:.3f}")
    model.eval()
    res = {"model": Path(a.model).name, "memory": a.memory, "config": a.config, "active_ffn_frac": round(frac, 3),
           "dense_ppl": round(base, 4), "ppl_healed": round(ppl_heal, 4), "ppl_start": round(ppl_start, 4),
           "ppl_final": hist[-1][1], "ppl_best": min(h[1] for h in hist), "ratio_final": round(hist[-1][1] / base, 4),
           "history": hist, "steps": a.steps, "train_tokens": int(chunks.numel()), "rank": r, "b2": a.b2, "b3": a.b3,
           "lr": a.lr, "lr_mem": a.lr_mem, "lr_table": a.lr_table, "smoke": a.smoke, "total_s": round(time.time() - t0, 1)}
    if mem is not None:
        res["table_mb"] = round(sum(T.numel() for T in mem.T) * 2 / 2**20)      # if stored as fp16
    with open(OUT / "distill_engram_results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(res) + "\n")
    C.log("done", json.dumps(res))


if __name__ == "__main__":
    main()
