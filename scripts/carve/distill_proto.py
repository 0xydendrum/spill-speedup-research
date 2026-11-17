"""Step 3 of the dense->MoE project: whole-model distillation of a carved model.
Carve -> quick layer-wise heal -> train ALL carved layers together so the carved model's next-token predictions match
the dense model's (teacher top-k probabilities, saved up front), then score WikiText-2 perplexity.

  python distill_proto.py --model path/to/Qwen3.5-2B --config 0.25:4 --steps 1500
"""
import argparse, json, math, time
from pathlib import Path

import torch
import torch.nn.functional as F

import carve_proto as C
import heal_proto as H

OUT = Path(__file__).resolve().parent


@torch.no_grad()
def teacher_topk(model, ids, seq, k, bs=4):
    chunks = torch.stack([ids[i:i + seq] for i in range(0, len(ids) - seq + 1, seq)])
    vals, idxs = [], []
    for i in range(0, len(chunks), bs):
        logits = model(input_ids=chunks[i:i + bs].cuda()).logits.float()
        lp = torch.log_softmax(logits, -1)
        v, ix = lp.topk(k, -1)
        vals.append(v.half().cpu()); idxs.append(ix.int().cpu())
    return chunks, torch.cat(vals), torch.cat(idxs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--experts", type=int, default=32)
    ap.add_argument("--config", default="0.25:4")
    ap.add_argument("--cal-tokens", type=int, default=65536)
    ap.add_argument("--train-tokens", type=int, default=131072)
    ap.add_argument("--eval-tokens", type=int, default=16384)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--topk-teacher", type=int, default=32)
    ap.add_argument("--heal-steps", type=int, default=600)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.cal_tokens, a.train_tokens, a.eval_tokens, a.heal_steps, a.steps, a.eval_every = 8192, 4096, 2048, 20, 10, 10
    torch.backends.cuda.matmul.allow_tf32 = True
    t0 = time.time()
    model, tok = C.load_model(a.model, "cuda")
    for p in model.parameters():
        p.requires_grad_(False)
    layers = C.mlps(model)
    sf, k = float(a.config.split(":")[0]), int(a.config.split(":")[1])
    cal = C.text_tokens(tok, "train", a.cal_tokens + a.train_tokens)
    cal_carve, cal_train = cal[: a.cal_tokens], cal[a.cal_tokens:]
    ev = C.text_tokens(tok, "test", a.eval_tokens)
    base = C.perplexity(model, ev, 1024 if not a.smoke else 512, "cuda")
    C.log(f"dense perplexity {base:.3f}")
    # 1) teacher predictions (dense model) on the training text
    chunks, tv, ti = teacher_topk(model, cal_train, a.seq, a.topk_teacher)
    C.log(f"teacher top-{a.topk_teacher} saved for {chunks.numel()} tokens")
    # 2) carve + quick layer-wise heal
    freq, xs, acts = C.collect(model, layers, cal_carve, 1024 if not a.smoke else 512, "cuda", 8192 if not a.smoke else 1024)
    paths = H.dump_io(model, layers, cal_carve, 1024 if not a.smoke else 512, min(1.0, 32768 / len(cal_carve)), "distill")
    carves = [C.carve_layer(m, freq[i], xs[i], acts[i], a.experts, sf) for i, m in enumerate(layers)]
    del xs, acts
    mods = []
    for i, m in enumerate(layers):
        mod = H.CarvedFFN(m, carves[i], k)
        if a.heal_steps:
            H.heal_layer(mod, paths[i], a.heal_steps, 1e-4, 1024)
        mod = mod.to(torch.bfloat16)                       # bf16 for whole-model training
        mods.append(mod)
        # free the original dense MLP weights: the carved copy replaces them
        for lin in (m.gate_proj, m.up_proj, m.down_proj):
            lin.weight = torch.nn.Parameter(torch.empty(0, device="cuda", dtype=torch.bfloat16), requires_grad=False)
        H.install_healed(m, mod)
    torch.cuda.empty_cache()
    frac = (len(carves[0]["shared"]) + k * carves[0]["experts"].shape[1]) / layers[0].gate_proj.out_features
    ppl_heal = C.perplexity(model, ev, 1024 if not a.smoke else 512, "cuda")
    C.log(f"[{a.config}] active {frac:.3f}  after layer-wise heal: perplexity {ppl_heal:.3f}  GPU {torch.cuda.memory_allocated()/2**30:.1f} GiB")
    # 3) whole-model distillation
    params = [p for mod in mods for p in mod.parameters()]
    for p in params:
        p.requires_grad_(True)
    try:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(params, lr=a.lr, weight_decay=0.0)
    except Exception as e:
        C.log("bitsandbytes unavailable, plain AdamW:", e)
        opt = torch.optim.AdamW(params, lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.train()
    g = torch.Generator().manual_seed(0)
    hist = []
    for s in range(a.steps):
        bi = torch.randint(0, len(chunks), (a.bs,), generator=g)
        x = chunks[bi].cuda()
        logits = model(input_ids=x, use_cache=False).logits.float()
        lp = torch.log_softmax(logits, -1)
        t_lp = tv[bi].cuda().float(); t_ix = ti[bi].cuda().long()
        t_p = torch.softmax(t_lp, -1)                      # teacher top-k, renormalised
        kd = -(t_p * lp.gather(-1, t_ix)).sum(-1).mean()
        ce = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1))
        loss = kd + 0.25 * ce
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        if (s + 1) % 50 == 0:
            C.log(f"step {s+1}/{a.steps} kd {kd.item():.4f} ce {ce.item():.4f}")
        if (s + 1) % a.eval_every == 0 or s == a.steps - 1:
            model.eval()
            p_ = C.perplexity(model, ev, 1024 if not a.smoke else 512, "cuda")
            model.train()
            hist.append((s + 1, round(p_, 4)))
            C.log(f"eval at step {s+1}: perplexity {p_:.3f}")
    model.eval()
    ppl_final = hist[-1][1]
    res = {"model": Path(a.model).name, "config": a.config, "active_ffn_frac": round(frac, 3), "dense_ppl": round(base, 4),
           "ppl_layerwise_heal": round(ppl_heal, 4), "ppl_distilled": ppl_final, "ratio_distilled": round(ppl_final / base, 4),
           "history": hist, "steps": a.steps, "lr": a.lr, "train_tokens": int(chunks.numel()), "smoke": a.smoke,
           "total_s": round(time.time() - t0, 1)}
    with open(OUT / "distill_results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(res) + "\n")
    if not a.smoke:
        torch.save({"carves": [{kk: v.cpu() for kk, v in c.items()} for c in carves],
                    "state": [{kk: v.cpu() for kk, v in m_.state_dict().items()} for m_ in mods], "k": k, "config": a.config},
                   OUT / f"distilled-{Path(a.model).name}-{a.config.replace(':', 'k')}.pt")
    C.log("done", json.dumps(res))


if __name__ == "__main__":
    main()
