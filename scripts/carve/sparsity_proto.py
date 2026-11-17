"""'Run a dense model like an MoE without carving it': every feed-forward neuron is its own expert and the model's
own gate projection is the router.  Per token, keep only the neurons with the largest |silu(gate x)| (CATS-style),
so their rows of up_proj and columns of down_proj are the only ones that would have to be read from RAM.
Simulated in PyTorch, scored by WikiText-2 perplexity.

  python sparsity_proto.py --model path/to/Qwen3.5-2B --keep 0.7,0.5,0.4,0.3,0.2
Bytes read per token for the FFN = gate (1/3, always read) + (up + down) * keep (2/3 * keep).
"""
import argparse, json, time
from pathlib import Path

import torch
import torch.nn.functional as F

import carve_proto as C


def install_sparse(mlp, keep):
    Wg, Wu, Wd = mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight
    act = mlp.act_fn
    n_keep = max(1, int(round(Wg.shape[0] * keep)))

    def fwd(x):
        shp = x.shape
        h = x.reshape(-1, shp[-1])
        g = act(F.linear(h, Wg))                                  # router = the real gate (always computed)
        idx = g.abs().topk(n_keep, dim=1).indices
        mask = torch.zeros_like(g).scatter_(1, idx, 1.0)
        return F.linear(g * mask * F.linear(h, Wu), Wd).reshape(shp)   # up/down only matter for kept neurons
    mlp._orig_forward = getattr(mlp, "_orig_forward", mlp.forward)
    mlp.forward = fwd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--keep", default="0.7,0.5,0.4,0.3,0.2")
    ap.add_argument("--eval-tokens", type=int, default=16384)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--out", default="sparsity_results.jsonl")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.eval_tokens, a.seq = 2048, 512
    t0 = time.time()
    model, tok = C.load_model(a.model, "cuda")
    layers = C.mlps(model)
    ev = C.text_tokens(tok, "test", a.eval_tokens)
    base = C.perplexity(model, ev, a.seq, "cuda")
    C.log(f"dense perplexity {base:.3f}")
    res = {"model": Path(a.model).name, "dense_ppl": round(base, 4), "smoke": a.smoke, "eval_tokens": len(ev), "configs": []}
    for k in [float(v) for v in a.keep.split(",")]:
        for m in layers:
            install_sparse(m, k)
        ppl = C.perplexity(model, ev, a.seq, "cuda")
        for m in layers:
            C.uninstall(m)
        cfg = {"keep": k, "ffn_bytes_frac": round(1 / 3 + 2 / 3 * k, 3), "ppl": round(ppl, 4), "ppl_ratio": round(ppl / base, 4)}
        res["configs"].append(cfg)
        C.log(json.dumps(cfg))
    res["total_s"] = round(time.time() - t0, 1)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(res) + "\n")
    C.log("done")


if __name__ == "__main__":
    main()
