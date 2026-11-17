# F5. Negative result: an n-gram "snapshot" memory does not repair a dense model carved into experts

## Claim

Carving the feed-forward layers of the dense Qwen3.5-2B into a mixture of experts (34% of the feed-forward weights
used per token) without training raised WikiText-2 perplexity from 11.85 to 30.58. A training-free memory that stores,
per recent-token pattern, what the dense layers produced that the carved layers miss (a "snapshot" of the dense model,
in the spirit of DeepSeek's Engram) brought it to 28.66 at best, against 29.92 for a plain fixed correction. After
whole-model distillation on 1M tokens, the memory made no difference: 16.05 with the snapshot memory, 16.01 with no
memory and 16.01 with an empty one. Twice the training text (2M tokens, no memory) gave 15.16.

## Setup

- Model Qwen3.5-2B (bf16, transformers), PyTorch simulation on the RTX 3060.
- Carve: each feed-forward layer split into a shared part (25% of neurons) plus routed experts, top 4 per token
  (config `0.25:4`, 34.4% active). Router = the model's own gate projection.
- Score: WikiText-2 (raw, v1) perplexity on held-out text (`eval_tokens` 16,384).
- Scripts `scripts/carve/*.py`. Runs on 2026-11-15 and 2026-11-16, EST.
- The snapshot idea (store the dense-minus-carved gap in n-gram tables, then train the carved model with them
  attached) is 0xydendrum's.

## Result

| Step | Memory | Training text | Perplexity | vs dense |
|---|---|---|---|---|
| Dense model | - | - | 11.85 | 1.00x |
| Carved, no training | none | - | 30.58 | 2.58x |
| Carved + fixed per-layer correction | bias | - | 29.92 | 2.53x |
| Carved + snapshot memory (unigram) | snapshot | - | 29.04 | 2.45x |
| Carved + snapshot memory (back-off, scale 0.75) | snapshot | - | 28.66 | 2.42x |
| Carved, layer-by-layer healing | none | 65K tokens | 22.76 | 1.92x |
| Carved, whole-model distillation | none | 131K tokens | 20.63 | 1.74x |
| Carved, distillation | none | 1M tokens | 16.01 | 1.35x |
| Carved, distillation | empty table | 1M tokens | 16.01 | 1.35x |
| Carved, distillation | snapshot | 1M tokens | 16.05 | 1.35x |
| Carved, distillation | none | 2M tokens | 15.16 | 1.28x |

Related baselines from the same project: skipping the least active 30% of feed-forward neurons per token without
carving cost little (11.85 to 12.37), and a perfect router ("oracle") at the same 34% active gave 25.44 instead of
30.60, so the router explains only part of the carving loss.

## Evidence

- `data/moe_carve_engram.csv` (every real run; smoke tests removed), experiments `engram_snapshot_training_free`,
  `carve_distill_plus_memory`, `carve_training_free`, `carve_layerwise_heal*`, `carve_whole_model_distill`,
  `neuron_skip_no_carve`.

## Caveats

- One small model (2B), one carve configuration for the memory tests, perplexity only (no task scores).
- The memory tables were rank 32 and built from 2.5M tokens; a much larger table or text might behave differently.
  The result says the snapshot did not help at this scale, not that it cannot.
- Training runs are single seeds.

## Prior work

- Turning dense feed-forward layers into experts: Zhang et al., "MoEfication: Transformer Feed-forward Layers are
  Mixtures of Experts" (Findings of ACL 2022); Zhu et al., "LLaMA-MoE: Building Mixture-of-Experts from LLaMA with
  Continual Pre-training" (2024).
- Activation-sparse inference with hot/cold neurons: Song et al., "PowerInfer" (2023).
- N-gram lookup memory for language models: DeepSeek's Engram work (January 2026), which this experiment borrows the
  table idea from. A community model combines Qwen with Engram tables (summerAIO/Cyber-9B-Qwen-AllGDN-Engram).
