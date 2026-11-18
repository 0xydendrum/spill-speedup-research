# F7. Negative result: a forward-filled n-gram memory cannot repair a healed dense-to-MoE carve, even with Engram's gate, canonical keys and early layers

## Claim

After the carve from F5 is healed layer by layer, WikiText-2 perplexity is 28.50 against 13.27 for the dense model.
A DeepSeek-Engram-style table filled with forward passes only (no training) can predict at most **2.73%** of the hidden-state
gap the carve leaves (held-out WikiText-2, bar 5%). That is with every part of Engram given its best case: canonical token
IDs, layer 0, and a learned context gate with a value projection. Adding the prediction back closes **0.43%** of the
perplexity gap (bar 5 percentage points). Even on C4, where the table predicts 8.4-9.5% of the gap, injection closes only
1.8-2.2%. Three preregistered tests (A0, A0b, A0c) all ended KILL.

## Setup

- Model: Qwen3.5-2B (HF revision 15852e8), bf16, PyTorch/transformers 5.5.0. It has 24 layers: 18 Gated DeltaNet
  linear-attention and 6 full-attention layers (3, 7, ..., 23). Linear attention used the torch fallback (no `fla`).
- Carve: F5's `0.25:4` (25% shared neurons + routed top-4, 34.4% active), then per-layer healing
  (`heal_proto.py`). Gap per probe layer L: d_L = h_dense(L) - h_healed(L), each model on its own upstream states.
- Target: rank-128 PCA code of d_L (basis from 64K FineWeb tokens). R² counts the out-of-basis part as error, pooled over
  64K held-out tokens.
- Table: filled from 20M FineWeb tokens (sample-10BT, stream order) with orders 1 (exact) and 2-3 (hashed, 2 heads,
  2^19 buckets). Readouts: back-off 3->2->1 with shrinkage lambda 8, or the mean of the head bucket means ("aread").
- Keys: raw token IDs, or canonical IDs: NFKC + lowercase + strip, with whitespace-only tokens mapped to " " or "\n".
  That is 248,077 -> 193,925 IDs (-21.8%; Engram reports -23% for its 128K tokenizer).
- Gates (A0b): per-position oracle gates on the table prediction, clip alpha to [0, 1] or unrestricted, minus the same
  gate on shuffled keys. Gates (A0c): a learned Engram gate, alpha = sigmoid(RMSNorm(h) . RMSNorm(W_K c_hat)/sqrt(2048) + b),
  with h the healed hidden state, optionally with a learned W_V. Tables come from fill tokens 0-18M and the gate is
  trained on tokens 18M-20M, which the tables never saw.
- Injected G: perplexity with mu + U c_hat added at that layer inside the healed forward pass, then
  (ppl_healed - ppl_injected) / (ppl_healed - ppl_dense).
- Controls: a shuffled-key decoy for every arm, and a planted signal (10% of the drift variance, keyed on canonical ID)
  that must be recovered. 95% intervals are a block bootstrap over 64 eval sequences.
- Runs 2026-11-17/18 (EST) on the RTX 3060 and rented A6000/A100 GPUs. Preregistered bars: R² >= 5% and G >= 5 pp.
- The forward-filled memory idea is 0xydendrum's.

## Result

| Test | Best arm (held-out WikiText-2) | R² % [95% CI] | G pp | C4 R² % | C4 G pp | Outcome |
|---|---|---|---|---|---|---|
| A0 | raw keys, aread, layer 3 (probes 3-23) | 0.85 | 0.44 | 3.95 | 5.79 | KILL |
| A0b | canonical keys, aread, layer 0 (probes 0-7) | 2.18 | 0.26 | 8.40 | 1.81 | KILL |
| A0b | same + oracle clip gate, excess over decoy | 3.15 | - | 8.65 | - | KILL |
| A0c | canonical, learned gate, layer 0 | 2.38 [2.21, 2.58] | - | 8.40 | - | KILL |
| A0c | canonical, learned gate + W_V, layer 0 | 2.73 [2.52, 2.96] | 0.43 | 9.49 | 2.22 | KILL |

| Control | Result |
|---|---|
| planted 10% signal | 84% (raw) / 85% (canonical) recovered |
| shuffled-key decoys | every ungated and learned decoy below 0 (A0b max -0.95%, A0c -0.40%); oracle-gate decoys sit above 0 by construction and are subtracted |
| A0b raw-key layer-3 arm vs A0 | 0.85% = 0.85% (same pipeline result) |
| oracle table filled on the eval text itself (A0, layer 3) | closes only 11.3% (wiki) / 18.0% (C4) of the gap |

What helped, each by a little: canonical IDs (+0.2-0.4 pp at every layer), earlier layers (layer 0 > 1 > ... > 7), the
head-mean readout over back-off (back-off suffers from hash collisions), and the learned gate (+0.6 pp).

## Evidence

- `data/engram_v2_r2.csv`: every R² (run x eval x keyset x layer x column x gate x real/decoy/planted), with CIs where
  bootstrapped.
- `data/engram_v2_gap.csv`: dense, healed and injected perplexity and G per run, plus A0's oracle.
- Scripts: `scripts/carve/engram_diag.py` (A0), `engram_diag_b.py` (A0b), `engram_gate_c.py` (A0c),
  `prep_engram_data.py` (data), `run_manifest.py` (weights, tokenizer and module-tree fingerprint per run).

## Caveats

- One small model (2B) and one carve configuration, scored on perplexity only.
- The gap was measured after layer-by-layer healing, not after whole-model distillation (F5 found no gain from a trained
  memory there either).
- Tables have rank 128. That keeps only 22-27% of the gap's variance at layers 0-3, which caps R² (observed values sit
  far below that cap) and limits injection.
- 2 hash heads, not Engram's 8, and a mean or back-off readout, not a learned projection over concatenated heads.
- 20M fill tokens, against 262B training tokens for Engram.
- Not tested: a table trained end to end with the backbone, which is how Engram is built (paper: arXiv:2601.07372). This
  finding covers training-free filling plus a small learned gate on frozen tables.
- WikiText-2 decided; C4 is web text like the FineWeb fill and scores about 4x higher on R², yet still under 2.3 pp G.

## Prior work

- DeepSeek, "Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models" (Engram,
  arXiv:2601.07372, January 2026): hashed n-gram tables with tokenizer compression, a context gate and a short conv,
  trained end to end. This finding borrows its keys, gate and placement.
- F5 in this repo: snapshot memory and distillation on the same carve.
- Dense-to-MoE carving: Zhang et al., "MoEfication" (Findings of ACL 2022); Zhu et al., "LLaMA-MoE" (2024).
