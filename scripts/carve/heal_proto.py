"""Step 2 of the dense->MoE project: carve, then 'heal' each carved feed-forward layer by training it to reproduce the
dense layer's output (layer-wise distillation), then score the whole model by WikiText-2 perplexity.

The trained forward pass is exactly what llama.cpp's qwen35moe computes, so a healed layer exports losslessly:
    out = shared(x) + sum_{e in topk(router(x))} (k * softmax_over_topk(router(x))_e) * expert_e(x)
(in the GGUF: routed down_proj weights pre-multiplied by k; shared gate weights 0 with shared down x2).

  python heal_proto.py --model path/to/Qwen3.5-2B --configs 0.25:4,0.125:4 --steps 600
"""
import argparse, json, math, time
from pathlib import Path

import torch
import torch.nn.functional as F

import carve_proto as C

CACHE = Path(__file__).resolve().parent / "cache"


@torch.no_grad()
def dump_io(model, layers, ids, seq, keep_p, tag, bs=4):
    """Save (x, y) = (MLP input, dense MLP output) for a random token subset of every layer, fp16, one file per layer."""
    CACHE.mkdir(parents=True, exist_ok=True)
    bufx = [[] for _ in layers]
    bufy = [[] for _ in layers]
    sel_holder = {}
    hooks = []
    for li, m in enumerate(layers):
        def fh(mod, inp, out, li=li):
            x = inp[0].reshape(-1, inp[0].shape[-1])
            y = out.reshape(-1, out.shape[-1])
            sel = sel_holder["sel"]
            bufx[li].append(x[sel].half().cpu())
            bufy[li].append(y[sel].half().cpu())
        hooks.append(m.register_forward_hook(fh))
    chunks = [ids[i:i + seq] for i in range(0, len(ids) - seq + 1, seq)]
    g = torch.Generator().manual_seed(1)
    for i in range(0, len(chunks), bs):
        x = torch.stack(chunks[i:i + bs])
        sel_holder["sel"] = (torch.rand(x.numel(), generator=g) < keep_p).to("cuda")
        model(input_ids=x.to("cuda"))
    for h in hooks:
        h.remove()
    paths = []
    for li in range(len(layers)):
        p = CACHE / f"{tag}-L{li:02d}.pt"
        torch.save({"x": torch.cat(bufx[li]), "y": torch.cat(bufy[li])}, p)
        bufx[li] = bufy[li] = None
        paths.append(p)
    return paths


@torch.no_grad()
def dump_one(model, layers, li, ids, seq, keep_p, bs=4):
    """(x, y) for layer li only, with whatever is currently installed in earlier layers (sequential healing).
    y is the DENSE output of layer li on those inputs (layer li itself is still dense while this runs)."""
    bx, by = [], []
    holder = {}
    def fh(mod, inp, out):
        x = inp[0].reshape(-1, inp[0].shape[-1]); y = out.reshape(-1, out.shape[-1])
        bx.append(x[holder["sel"]].half().cpu()); by.append(y[holder["sel"]].half().cpu())
    h = layers[li].register_forward_hook(fh)
    chunks = [ids[i:i + seq] for i in range(0, len(ids) - seq + 1, seq)]
    g = torch.Generator().manual_seed(1)
    for i in range(0, len(chunks), bs):
        x = torch.stack(chunks[i:i + bs])
        holder["sel"] = (torch.rand(x.numel(), generator=g) < keep_p).to("cuda")
        model(input_ids=x.to("cuda"))
    h.remove()
    p = CACHE / f"seq-L{li:02d}.pt"
    CACHE.mkdir(parents=True, exist_ok=True)
    torch.save({"x": torch.cat(bx), "y": torch.cat(by)}, p)
    return p


class CarvedFFN(torch.nn.Module):
    """Trainable carved layer kept in the dense neuron layout; experts are defined by neuron ownership."""

    def __init__(self, mlp, carve, k):
        super().__init__()
        dev = mlp.gate_proj.weight.device
        self.wg = torch.nn.Parameter(mlp.gate_proj.weight.detach().float().clone())
        self.wu = torch.nn.Parameter(mlp.up_proj.weight.detach().float().clone())
        self.wd = torch.nn.Parameter(mlp.down_proj.weight.detach().float().clone())
        self.rw = torch.nn.Parameter(carve["router_w"].to(dev).float().clone())
        self.rb = torch.nn.Parameter(carve["router_b"].to(dev).float().clone())
        n_ff = self.wg.shape[0]
        E = carve["experts"].to(dev)
        self.n_exp = E.shape[0]
        owner = torch.full((n_ff,), -1, dtype=torch.long, device=dev)
        owner[E.flatten()] = torch.arange(self.n_exp, device=dev).repeat_interleave(E.shape[1])
        self.register_buffer("onehot", F.one_hot(owner.clamp(min=0), self.n_exp).float() * (owner >= 0).unsqueeze(1).float())
        self.register_buffer("is_shared", (owner < 0).float())
        self.k = k
        self.act = mlp.act_fn
        # start with near-uniform weights so the initial output equals the plain carve (weight ~1 per chosen expert)
        with torch.no_grad():
            self.temp = 1.0
            s = (self.rw.std() * math.sqrt(self.rw.shape[1])).item()
            self.rw.mul_(0.5 / max(s, 1e-6)); self.rb.mul_(0.5 / max(s, 1e-6))

    def forward(self, h):
        a = self.act(h @ self.wg.T) * (h @ self.wu.T)                     # (T, n_ff)
        logits = h @ self.rw.T + self.rb
        topv, topi = logits.topk(self.k, dim=1)
        w = torch.softmax(topv, dim=1) * self.k                            # llama.cpp: normalised top-k weights x scale k
        we = torch.zeros(h.shape[0], self.n_exp, device=h.device, dtype=h.dtype).scatter(1, topi, w.to(h.dtype))
        scale = we @ self.onehot.T + self.is_shared                        # per-neuron multiplier
        return (a * scale) @ self.wd.T


def heal_layer(mod, path, steps, lr, bs, log_every=0):
    d = torch.load(path)
    x, y = d["x"].float(), d["y"].float()
    n = x.shape[0]
    nv = max(256, n // 10)
    xv, yv = x[-nv:].cuda(), y[-nv:].cuda()
    xt, yt = x[:-nv], y[:-nv]
    denom_v = (yv ** 2).mean()
    opt = torch.optim.AdamW(mod.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)

    def val():
        with torch.no_grad():
            return (((mod(xv) - yv) ** 2).mean() / denom_v).item()
    v0 = val()
    best, best_state = v0, None
    g = torch.Generator().manual_seed(0)
    for s in range(steps):
        idx = torch.randint(0, xt.shape[0], (bs,), generator=g)
        xb, yb = xt[idx].cuda(), yt[idx].cuda()
        loss = ((mod(xb) - yb) ** 2).mean() / (yb ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(); sched.step()
        if (s + 1) % max(1, steps // 6) == 0 or s == steps - 1:
            v = val()
            if v < best:
                best = v
                best_state = {k_: t.detach().clone() for k_, t in mod.state_dict().items()}
    if best_state is not None:
        mod.load_state_dict(best_state)
    return v0, best


def install_healed(mlp, mod):
    def fwd(x, mod=mod):
        shp = x.shape
        dt = mod.wg.dtype
        return mod(x.reshape(-1, shp[-1]).to(dt)).to(x.dtype).reshape(shp)
    mlp._orig_forward = getattr(mlp, "_orig_forward", mlp.forward)
    mlp.forward = fwd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--experts", type=int, default=32)
    ap.add_argument("--configs", default="0.25:4,0.125:4", help="shared_frac:topk pairs")
    ap.add_argument("--cal-tokens", type=int, default=131072)
    ap.add_argument("--train-tokens", type=int, default=32768)
    ap.add_argument("--sample-tokens", type=int, default=8192)
    ap.add_argument("--eval-tokens", type=int, default=16384)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--out", default="heal_results.jsonl")
    ap.add_argument("--sequential", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.cal_tokens, a.train_tokens, a.sample_tokens, a.eval_tokens, a.seq, a.steps = 8192, 4096, 1024, 2048, 512, 20
    torch.backends.cuda.matmul.allow_tf32 = True
    t0 = time.time()
    model, tok = C.load_model(a.model, "cuda")
    for p_ in model.parameters():
        p_.requires_grad_(False)
    layers = C.mlps(model)
    tag = Path(a.model).name + ("-smoke" if a.smoke else "")
    cal = C.text_tokens(tok, "train", a.cal_tokens)
    ev = C.text_tokens(tok, "test", a.eval_tokens)
    base = C.perplexity(model, ev, a.seq, "cuda")
    C.log(f"dense perplexity {base:.3f}")
    freq, xs, acts = C.collect(model, layers, cal[: a.cal_tokens // 2], a.seq, "cuda", a.sample_tokens)
    paths = None
    if not a.sequential:
        paths = dump_io(model, layers, cal, a.seq, min(1.0, a.train_tokens / len(cal)), tag)
        C.log(f"saved layer inputs/outputs ({torch.load(paths[0])['x'].shape[0]} tokens per layer)")
    res = {"model": Path(a.model).name, "dense_ppl": round(base, 4), "smoke": a.smoke, "steps": a.steps, "lr": a.lr,
           "sequential": a.sequential, "train_tokens": a.train_tokens,
           "eval_tokens": len(ev), "configs": []}
    for cfg in a.configs.split(","):
        sf, k = float(cfg.split(":")[0]), int(cfg.split(":")[1])
        carves = [C.carve_layer(m, freq[i], xs[i], acts[i], a.experts, sf) for i, m in enumerate(layers)]
        n_ff = layers[0].gate_proj.weight.shape[0]
        frac = (len(carves[0]["shared"]) + k * carves[0]["experts"].shape[1]) / n_ff
        # carve-only perplexity with the same (soft-weight) forward, before healing
        mods = [CarvedFFN(m, carves[i], k) for i, m in enumerate(layers)]
        for i, m in enumerate(layers):
            install_healed(m, mods[i])
        ppl0 = C.perplexity(model, ev, a.seq, "cuda")
        for m in layers:
            C.uninstall(m)
        C.log(f"[{cfg}] active {frac:.3f}  carve-only perplexity {ppl0:.3f}")
        errs = []
        for i, m in enumerate(layers):
            if a.sequential:      # inputs come through the already-healed earlier layers
                pth = dump_one(model, layers, i, cal, a.seq, min(1.0, a.train_tokens / len(cal)))
            else:
                pth = paths[i]
            v0, v1 = heal_layer(mods[i], pth, a.steps, a.lr, a.bs)
            errs.append((v0, v1))
            C.log(f"[{cfg}] layer {i:02d}: relative error {v0:.4f} -> {v1:.4f}")
            if a.sequential:
                install_healed(m, mods[i])
        for i, m in enumerate(layers):
            install_healed(m, mods[i])
        ppl1 = C.perplexity(model, ev, a.seq, "cuda")
        for m in layers:
            C.uninstall(m)
        out = {"shared_frac": sf, "topk": k, "active_ffn_frac": round(frac, 3), "ppl_carve": round(ppl0, 4),
               "ppl_healed": round(ppl1, 4), "ratio_healed": round(ppl1 / base, 4),
               "mean_rel_err_before": round(sum(e[0] for e in errs) / len(errs), 4),
               "mean_rel_err_after": round(sum(e[1] for e in errs) / len(errs), 4)}
        res["configs"].append(out)
        C.log(json.dumps(out))
        del mods
        torch.cuda.empty_cache()
    res["total_s"] = round(time.time() - t0, 1)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(res) + "\n")
    C.log("done")


if __name__ == "__main__":
    main()
