"""Training-free carving of a dense Qwen3.5 model's feed-forward layers into a shared expert + routed experts,
simulated in PyTorch, scored by WikiText-2 perplexity.  Prototype step 1 of the dense->MoE project.

  python carve_proto.py --model path/to/Qwen3.5-2B --experts 32 --shared-fracs 0.125 --topk 2,4,8 [--smoke]

Per layer:
  1. record |act| = |silu(gate x) * up x| for calibration tokens (forward pre-hook on down_proj)
  2. shared expert  = neurons that are most often among a token's strongest activations (CMoE-style frequency)
  3. routed experts = balanced k-means of the remaining neurons on their activation patterns (fire together)
  4. router         = ridge regression from the layer input x to each expert's activation mass
Then each MLP forward becomes: shared(x) + sum over router top-k experts of expert(x) (weight 1, like the dense sum).
"""
import argparse, json, math, time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

DATA = Path(__file__).resolve().parent / "data" / "wikitext-2-raw-v1"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_model(path, device):
    import transformers
    try:
        m = transformers.AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map=device)
    except Exception as e:  # multimodal checkpoints
        log("AutoModelForCausalLM failed, trying ImageTextToText:", type(e).__name__, str(e)[:120])
        m = transformers.AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, device_map=device)
    tok = transformers.AutoTokenizer.from_pretrained(path)
    m.eval()
    return m, tok


def mlps(model):
    out = [mod for name, mod in model.named_modules() if type(mod).__name__.endswith("MLP")
           and hasattr(mod, "gate_proj") and "visual" not in name and "mtp" not in name.lower()]
    return out


def text_tokens(tok, split, n_tokens):
    df = pd.read_parquet(DATA / f"{split}-00000-of-00001.parquet")
    txt = "\n\n".join(t for t in df["text"].tolist() if t.strip())
    ids = tok(txt[: n_tokens * 8], return_tensors="pt").input_ids[0]
    return ids[:n_tokens]


@torch.no_grad()
def perplexity(model, ids, seq, device, bs=4):
    nll, n = 0.0, 0
    chunks = [ids[i:i + seq] for i in range(0, len(ids) - seq + 1, seq)]
    for i in range(0, len(chunks), bs):
        x = torch.stack(chunks[i:i + bs]).to(device)
        out = model(input_ids=x, labels=x)
        k = x.numel() - x.shape[0]
        nll += out.loss.float().item() * k
        n += k
    return math.exp(nll / n)


@torch.no_grad()
def collect(model, layers, ids, seq, device, sample_tokens, bs=4):
    """Per layer: activation-frequency counts over all tokens, plus a random sample of (x, |act|) rows."""
    nl = len(layers)
    freq = [None] * nl
    xs = [[] for _ in range(nl)]
    acts = [[] for _ in range(nl)]
    total = [0]
    keep_p = [1.0]
    hooks = []
    for li, m in enumerate(layers):
        def pre_mlp(mod, inp, li=li):
            x = inp[0].reshape(-1, inp[0].shape[-1])
            mod._x = x
        def pre_down(mod, inp, li=li):
            a = inp[0].reshape(-1, inp[0].shape[-1]).abs().float()
            q = max(1, int(a.shape[1] * 0.1))                  # token's top 10% neurons count as "firing"
            thr = a.topk(q, dim=1).values[:, -1:]
            f = (a >= thr).float().sum(0)
            freq[li] = f if freq[li] is None else freq[li] + f
            sel = torch.rand(a.shape[0], device=a.device) < keep_p[0]
            if sel.any():
                acts[li].append(a[sel].half().cpu())
                xs[li].append(layers[li]._x[sel].float().cpu())
        hooks.append(m.register_forward_pre_hook(pre_mlp))
        hooks.append(m.down_proj.register_forward_pre_hook(pre_down))
    chunks = [ids[i:i + seq] for i in range(0, len(ids) - seq + 1, seq)]
    keep_p[0] = min(1.0, sample_tokens / max(1, len(chunks) * seq))
    for i in range(0, len(chunks), bs):
        model(input_ids=torch.stack(chunks[i:i + bs]).to(device))
    for h in hooks:
        h.remove()
    for m in layers:
        m._x = None
    return freq, [torch.cat(x) for x in xs], [torch.cat(a) for a in acts]


def balanced_kmeans(feat, k, iters=25, seed=0):
    """feat: (n_items, d) on GPU. Returns labels with exactly n_items/k items per cluster."""
    g = torch.Generator(device=feat.device).manual_seed(seed)
    n = feat.shape[0]
    cap = n // k
    feat = F.normalize(feat, dim=1)
    cent = feat[torch.randperm(n, generator=g, device=feat.device)[:k]].clone()
    labels = torch.zeros(n, dtype=torch.long, device=feat.device)
    for _ in range(iters):
        sim = feat @ cent.T                                   # (n, k)
        # greedy capacity-constrained assignment: strongest (item, cluster) pairs first
        order = torch.argsort(sim.flatten(), descending=True)
        labels.fill_(-1)
        count = torch.zeros(k, dtype=torch.long, device=feat.device)
        items, clus = order // k, order % k
        # vectorised greedy in rounds (fast enough for n <= ~20k)
        lab_cpu = [-1] * n
        cnt = [0] * k
        assigned = 0
        for it, cl in zip(items.tolist(), clus.tolist()):
            if lab_cpu[it] == -1 and cnt[cl] < cap:
                lab_cpu[it] = cl
                cnt[cl] += 1
                assigned += 1
                if assigned == n:
                    break
        labels = torch.tensor(lab_cpu, device=feat.device)
        new = torch.zeros_like(cent).index_add_(0, labels, feat)
        cent = F.normalize(new, dim=1)
    return labels


def carve_layer(mlp, freq, x, a, n_exp, shared_frac, ridge=1e-2):
    dev = mlp.gate_proj.weight.device
    n_ff = mlp.gate_proj.weight.shape[0]
    n_shared = int(round(n_ff * shared_frac / 64)) * 64
    rest = n_ff - n_shared
    n_shared += rest % n_exp                                   # make routed part divisible
    order = torch.argsort(freq, descending=True)
    shared = order[:n_shared].sort().values
    routed = order[n_shared:].sort().values
    feat = a[:, routed.cpu()].float().T.to(dev)               # (n_routed, tokens): activation pattern per neuron
    labels = balanced_kmeans(feat, n_exp)
    experts = [routed[labels == e] for e in range(n_exp)]
    # router: ridge regression x -> activation mass of each expert
    X = x.to(dev)
    Y = torch.stack([a[:, idx.cpu()].float().sum(1) for idx in experts], 1).to(dev)   # (tokens, n_exp)
    Y = Y / (Y.mean(0, keepdim=True) + 1e-6)
    Xb = torch.cat([X, torch.ones(X.shape[0], 1, device=dev)], 1)
    A = Xb.T @ Xb + ridge * X.shape[0] * torch.eye(Xb.shape[1], device=dev)
    A = torch.nan_to_num(A); B = torch.nan_to_num(Xb.T @ Y)
    try:
        W = torch.linalg.solve(A, B)                          # (H+1, n_exp)
    except Exception:
        W = torch.linalg.pinv(A) @ B
    return {"shared": shared, "experts": torch.stack(experts), "router_w": W[:-1].T.contiguous(),
            "router_b": W[-1].contiguous()}


def router_recall(c, x, a, k):
    dev = c["router_w"].device
    X = x.to(dev)
    true_mass = torch.stack([a[:, idx.cpu()].float().sum(1) for idx in c["experts"]], 1).to(dev)
    pred = X @ c["router_w"].T + c["router_b"]
    t = true_mass.topk(k, 1).indices
    p = pred.topk(k, 1).indices
    hit = (t.unsqueeze(2) == p.unsqueeze(1)).any(2).float().mean().item()
    return hit


def install(mlp, c, k, mode="router"):
    """Simulate the carved MoE: compute all neuron activations, keep shared + the k chosen experts, zero the rest.
    mode='router' uses the fitted router; mode='oracle' picks the k experts with the largest true activation mass."""
    Wg, Wu, Wd = mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight
    n_ff = Wg.shape[0]
    E = c["experts"].to(Wg.device)                             # (n_exp, m)
    n_exp = E.shape[0]
    owner = torch.full((n_ff,), -1, dtype=torch.long, device=Wg.device)   # -1 = shared neuron
    owner[E.flatten()] = torch.arange(n_exp, device=Wg.device).repeat_interleave(E.shape[1])
    onehot = F.one_hot(owner.clamp(min=0), n_exp).to(Wg.dtype) * (owner >= 0).unsqueeze(1).to(Wg.dtype)  # (n_ff, n_exp)
    rw, rb = c["router_w"].to(Wg.dtype), c["router_b"].to(Wg.dtype)
    act = mlp.act_fn

    def fwd(x, k=k):
        shp = x.shape
        h = x.reshape(-1, shp[-1])
        a = act(F.linear(h, Wg)) * F.linear(h, Wu)              # (T, n_ff)
        if mode == "oracle":
            score = a.abs() @ onehot                              # (T, n_exp) true activation mass
        else:
            score = h @ rw.T + rb
        idx = score.topk(k, dim=1).indices
        keep_e = torch.zeros(h.shape[0], n_exp, dtype=Wg.dtype, device=h.device).scatter_(1, idx, 1.0)
        keep = (keep_e @ onehot.T) + (owner < 0).to(Wg.dtype)    # (T, n_ff) 1 for kept neurons
        return F.linear(a * keep, Wd).reshape(shp)
    mlp._orig_forward = getattr(mlp, "_orig_forward", mlp.forward)
    mlp.forward = fwd


def uninstall(mlp):
    if hasattr(mlp, "_orig_forward"):
        mlp.forward = mlp._orig_forward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--experts", type=int, default=32)
    ap.add_argument("--shared-fracs", default="0.125")
    ap.add_argument("--modes", default="router,oracle")
    ap.add_argument("--topk", default="2,4,8")
    ap.add_argument("--cal-tokens", type=int, default=65536)
    ap.add_argument("--sample-tokens", type=int, default=8192)
    ap.add_argument("--eval-tokens", type=int, default=32768)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--out", default="results.jsonl")
    ap.add_argument("--save-carve", default="")
    ap.add_argument("--smoke", action="store_true", help="tiny sizes to test the pipeline end to end")
    a = ap.parse_args()
    if a.smoke:
        a.cal_tokens, a.sample_tokens, a.eval_tokens, a.seq = 2048, 1024, 2048, 512
    dev = "cuda"
    t0 = time.time()
    model, tok = load_model(a.model, dev)
    layers = mlps(model)
    log(f"loaded {Path(a.model).name}: {len(layers)} MLPs, n_ff={layers[0].gate_proj.weight.shape[0]}, "
        f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB")
    cal = text_tokens(tok, "train", a.cal_tokens)
    ev = text_tokens(tok, "test", a.eval_tokens)
    base = perplexity(model, ev, a.seq, dev)
    log(f"dense perplexity {base:.3f}")
    freq, xs, acts = collect(model, layers, cal, a.seq, dev, a.sample_tokens)
    log(f"collected activations ({xs[0].shape[0]} sampled tokens/layer)")
    n_ff = layers[0].gate_proj.weight.shape[0]
    res = {"model": Path(a.model).name, "experts": a.experts, "n_ff": n_ff, "dense_ppl": round(base, 4),
           "smoke": a.smoke, "eval_tokens": len(ev), "configs": []}
    for sf in [float(v) for v in a.shared_fracs.split(",")]:
        carves = [carve_layer(m, freq[li], xs[li], acts[li], a.experts, sf) for li, m in enumerate(layers)]
        n_sh = len(carves[0]["shared"]); m_exp = carves[0]["experts"].shape[1]
        log(f"carved shared_frac={sf}: shared {n_sh} neurons, {a.experts} experts x {m_exp} neurons")
        if a.save_carve:
            torch.save([{k: v.cpu() for k, v in c.items()} for c in carves], a.save_carve.replace(".pt", f"-sf{sf}.pt"))
        for k in [int(v) for v in a.topk.split(",")]:
            frac = (n_sh + k * m_exp) / n_ff
            rec = sum(router_recall(c, xs[i], acts[i], k) for i, c in enumerate(carves)) / len(carves)
            for mode in a.modes.split(","):
                for i, m in enumerate(layers):
                    install(m, carves[i], k, mode)
                ppl = perplexity(model, ev, a.seq, dev)
                for m in layers:
                    uninstall(m)
                cfg = {"shared_frac": sf, "topk": k, "mode": mode, "active_ffn_frac": round(frac, 3), "ppl": round(ppl, 4),
                       "ppl_ratio": round(ppl / base, 4), "router_recall": round(rec, 3)}
                res["configs"].append(cfg)
                log(json.dumps(cfg))
    res["total_s"] = round(time.time() - t0, 1)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(res) + "\n")
    log("done", json.dumps(res))


if __name__ == "__main__":
    main()
