"""Scale test for the dense->MoE project: the carving and 'gate as router' simulations, done with hooks only, so they
work on models bigger than the GPU (split between GPU and CPU with device_map=auto). Never reads weights directly:
it masks the feed-forward activations right before down_proj.

  python carve_scale.py --model path/to/Qwen3.5-9B --gpu-gib 9

Modes per MLP (active neurons get through, the rest are zeroed before down_proj):
  neuron:<keep>        keep the top <keep> fraction of neurons by |silu(gate x)| per token (no carving, exact router)
  carve:<sf>:<k>:<r>   shared fraction sf + top-k of 32 carved experts; r = router (fitted ridge) or oracle
"""
import argparse, json, time, types
from pathlib import Path

import torch
import torch.nn.functional as F

import carve_proto as C


def load_split(path, gpu_gib):
    import transformers
    kw = dict(dtype=torch.bfloat16, device_map="auto", max_memory={0: f"{gpu_gib}GiB", "cpu": "40GiB"})
    try:
        m = transformers.AutoModelForCausalLM.from_pretrained(path, **kw)
    except Exception as e:
        C.log("CausalLM failed, trying ImageTextToText:", type(e).__name__, str(e)[:120])
        m = transformers.AutoModelForImageTextToText.from_pretrained(path, **kw)
    tok = transformers.AutoTokenizer.from_pretrained(path)
    m.eval()
    return m, tok


class Masker:
    """Installs hooks on one MLP that zero activations of non-selected neurons before down_proj."""

    def __init__(self, mlp):
        self.mlp = mlp
        self.mode = None
        self.h = [mlp.register_forward_pre_hook(self.pre_mlp),
                  mlp.gate_proj.register_forward_hook(self.post_gate),
                  mlp.down_proj.register_forward_pre_hook(self.pre_down)]

    def pre_mlp(self, mod, inp):
        self.x = inp[0]

    def post_gate(self, mod, inp, out):
        self.g = out

    def pre_down(self, mod, inp):
        if self.mode is None:
            return None
        a = inp[0]
        shp = a.shape
        a2 = a.reshape(-1, shp[-1])
        if self.mode[0] == "neuron":
            g = self.mlp.act_fn(self.g).reshape(-1, shp[-1]).abs()
            n_keep = max(1, int(round(shp[-1] * self.mode[1])))
            mask = torch.zeros_like(a2).scatter_(1, g.topk(n_keep, dim=1).indices, 1.0)
        else:
            _, c, k, router = self.mode
            onehot = c["onehot"].to(a2.device, a2.dtype)
            if router == "oracle":
                score = a2.abs() @ onehot
            else:
                h = self.x.reshape(-1, self.x.shape[-1]).to(c["router_w"].device, torch.float32)
                score = (h @ c["router_w"].T + c["router_b"]).to(a2.device)
            idx = score.topk(k, dim=1).indices
            keep_e = torch.zeros(a2.shape[0], onehot.shape[1], device=a2.device, dtype=a2.dtype).scatter_(1, idx, 1.0)
            mask = keep_e @ onehot.T + c["is_shared"].to(a2.device, a2.dtype)
        return ((a2 * mask).reshape(shp),)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpu-gib", type=float, default=9)
    ap.add_argument("--experts", type=int, default=32)
    ap.add_argument("--modes", default="neuron:0.7,neuron:0.5,neuron:0.3,carve:0.25:4:router,carve:0.25:4:oracle,"
                                       "carve:0.375:8:router,carve:0.375:8:oracle")
    ap.add_argument("--cal-tokens", type=int, default=32768)
    ap.add_argument("--sample-tokens", type=int, default=6144)
    ap.add_argument("--eval-tokens", type=int, default=16384)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--out", default="scale_results.jsonl")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.cal_tokens, a.sample_tokens, a.eval_tokens, a.seq = 2048, 1024, 2048, 512
        a.modes = "neuron:0.5,carve:0.25:4:router"
    t0 = time.time()
    model, tok = load_split(a.model, a.gpu_gib)
    layers = C.mlps(model)
    n_ff = layers[0].gate_proj.out_features
    C.log(f"loaded {Path(a.model).name}: {len(layers)} MLPs, n_ff={n_ff}, GPU {torch.cuda.memory_allocated() / 2**30:.1f} GiB")
    dev0 = next(model.parameters()).device if next(model.parameters()).device.type != "meta" else "cuda"
    cal = C.text_tokens(tok, "train", a.cal_tokens)
    ev = C.text_tokens(tok, "test", a.eval_tokens)
    base = C.perplexity(model, ev, a.seq, "cuda")
    C.log(f"dense perplexity {base:.3f}")
    if not (base < 60):
        raise SystemExit(f"dense perplexity {base:.1f} looks broken (model mis-loaded?); stopping before wasting the run")
    res = {"model": Path(a.model).name, "n_ff": n_ff, "dense_ppl": round(base, 4), "smoke": a.smoke,
           "eval_tokens": len(ev), "configs": []}
    need_carve = any(m.startswith("carve") for m in a.modes.split(","))
    carves = {}
    if need_carve:
        freq, xs, acts = C.collect(model, layers, cal, a.seq, "cuda", a.sample_tokens)
        C.log(f"collected activations ({xs[0].shape[0]} sampled tokens/layer)")
    maskers = [Masker(m) for m in layers]
    for mode in a.modes.split(","):
        parts = mode.split(":")
        if parts[0] == "neuron":
            keep = float(parts[1])
            for mk in maskers:
                mk.mode = ("neuron", keep)
            frac_active, bytes_frac = keep, 1 / 3 + 2 / 3 * keep
        else:
            sf, k, router = float(parts[1]), int(parts[2]), parts[3]
            if sf not in carves:
                shim = lambda: types.SimpleNamespace(gate_proj=types.SimpleNamespace(weight=torch.empty((n_ff, 1), device="cuda")))
                cs = []
                for li in range(len(layers)):
                    c = C.carve_layer(shim(), freq[li].cuda(), xs[li], acts[li], a.experts, sf)
                    owner = torch.full((n_ff,), -1, dtype=torch.long, device="cuda")
                    E = c["experts"]
                    owner[E.flatten()] = torch.arange(E.shape[0], device="cuda").repeat_interleave(E.shape[1])
                    c["onehot"] = F.one_hot(owner.clamp(min=0), E.shape[0]).float() * (owner >= 0).unsqueeze(1).float()
                    c["is_shared"] = (owner < 0).float()
                    c["recall_k"] = {}
                    cs.append(c)
                carves[sf] = cs
                C.log(f"carved shared_frac={sf}: {len(cs[0]['shared'])} shared, {a.experts} x {cs[0]['experts'].shape[1]}")
            cs = carves[sf]
            for li, mk in enumerate(maskers):
                mk.mode = ("carve", cs[li], k, router)
            frac_active = (len(cs[0]["shared"]) + k * cs[0]["experts"].shape[1]) / n_ff
            bytes_frac = frac_active
        ppl = C.perplexity(model, ev, a.seq, "cuda")
        for mk in maskers:
            mk.mode = None
        cfg = {"mode": mode, "active_frac": round(frac_active, 3), "ffn_bytes_frac": round(bytes_frac, 3),
               "ppl": round(ppl, 4), "ppl_ratio": round(ppl / base, 4)}
        if parts[0] == "carve" and parts[3] == "router":
            cfg["router_recall"] = round(sum(C.router_recall(c, xs[i], acts[i], k) for i, c in enumerate(cs)) / len(cs), 3)
        res["configs"].append(cfg)
        C.log(json.dumps(cfg))
    res["total_s"] = round(time.time() - t0, 1)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(res) + "\n")
    C.log("done")


if __name__ == "__main__":
    main()
