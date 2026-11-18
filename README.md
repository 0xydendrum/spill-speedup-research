# Speeding up LLMs that don't fit in a 12 GB GPU: measurements

Measurements of what makes llama.cpp faster when a model is bigger than the GPU and part of it has to live in system
RAM ("spilled"). One consumer PC, five models, every number backed by a data file in `data/`.
Collected 2026-11-14 to 2026-11-18 (EST) by 0xydendrum. The tool that applies these results automatically is
[spilltune](https://github.com/0xydendrum/spilltune).

## TL;DR

1. **A confidence cut-off fixes slow prose with a guess-ahead helper.** Gemma 4 31B + DFlash helper (16 guesses):
   explanation 2.68 to 5.19 tokens/sec and writing 2.21 to 4.37 with `--spec-draft-p-min 0.7`, math and code faster
   too; quality 72.0% vs 72.2% fully in VRAM (n=450). [F1](findings/F1-confidence-cutoff.md),
   data `data/speed_setups.csv`.
2. **An MTP helper makes a mixture-of-experts model with its experts in RAM 1.5x faster.** Gemma 4 26B-A4B:
   37.5 to 56.0 tokens/sec (MTP, 3 guesses, cut-off 0.7), every prompt kind faster; quality 74.9% vs 75.3% (n=450).
   DFlash managed 1.14x. [F2](findings/F2-mtp-on-offloaded-moe.md), data `data/spilltune_trials.csv`.
3. **On a spilled dense model, MTP is fastest at 6 guesses; with the cut-off it reaches 2.61x.** Gemma 4 31B:
   3.42 to 8.92 tokens/sec. 8 to 12 guesses were slower on every prompt kind. [F3](findings/F3-guess-count-edge.md),
   data `data/speed_setups.csv`.
4. **Many requests at once beat any single-chat setting for total speed.** 31B: 20.7 tokens/sec total with 32
   requests vs 3.3 for one chat (6.2x); 26B-A4B: 98.4 with 32 requests; quality unchanged with 16 slots and an 8-bit
   cache. [F4](findings/F4-many-requests.md), data `data/parallel.csv`.
5. **Negative result: an n-gram snapshot memory does not repair a dense model carved into experts.** Qwen3.5-2B
   carved to 34% active: after distillation, perplexity 16.05 with the memory vs 16.01 without (dense 11.85).
   [F5](findings/F5-engram-snapshot-negative.md), data `data/moe_carve_engram.csv`.
6. **Negative result: a forward-filled Engram-style memory cannot repair the healed carve either.** Best case
   (canonical token keys, layer 0, learned context gate + value projection): it predicts 2.73% of the hidden-state gap
   on held-out WikiText-2 and closes 0.43% of the perplexity gap (bars 5% and 5 pp; three preregistered tests).
   [F7](findings/F7-engram-forward-filled-negative.md), data `data/engram_v2_r2.csv`, `data/engram_v2_gap.csv`.

Also measured (threads, context length, driver spill, other helpers, a smaller quant): [F6](findings/F6-other-measurements.md).

## Hardware and software

| | This PC (all speed tests) | Quality reference |
|---|---|---|
| GPU | NVIDIA GeForce RTX 3060 12 GB, driver 621.14 | NVIDIA RTX A6000 48 GB (rented), driver 580.126.09 |
| CPU / RAM | Intel Core i5-14600KF, 32 GB DDR4-3200 (2 x 16 GB) | - |
| OS | Windows 11 Pro (build 26200) | Linux |
| llama.cpp | release b11798 (commit ec0da70e9), CUDA build | release b11798, built from source |

Models (GGUF): `gemma-4-31B-it-Q4_K_M` (17.1 GB, dense, 60 layers), `gemma-4-26B-A4B-it-UD-Q4_K_M` (mixture of
experts), `Qwen3.5-35B-A3B-UD-IQ4_XS`, `Qwen3.8-27B-UD-Q4_K_M`, `gemma-4-12b-it-Q4_K_M` (control, fits in VRAM),
`gemma-4-31B-it-UD-IQ3_XXS` (smaller-file comparison). Helpers: `mtp-gemma-4-31B-it`, `mtp-gemma-4-26B-A4B-it`,
`gemma4-31b-it-dflash-Q4_K_M`, `gemma4-26b-a4b-it-dflash-Q4_K_M`, `gemma-4-31B-it-speculator.eagle3-Q4_K_M`.
Carving experiments (F5, F7): Qwen3.5-2B in PyTorch / transformers, bf16; F7 also on rented A6000/A100 GPUs.

How things were measured:
- **Speed** = spilltune's four prompts (math, code, an explanation, a story), 192 generated tokens each, temperature 0,
  one chat at a time, context 4096. "Overall" = total tokens / total time over the four prompts.
- **Quality** = 300 MMLU-Pro + 150 GSM8K test questions (fixed seed), temperature 0, thinking off. Each setup is
  compared with the same GGUF fully in VRAM on the reference GPU: paired difference with a 95% margin. "No loss" means
  difference + margin >= 0.

## Files

| Path | What it is |
|---|---|
| `findings/F1..F6-*.md` | One finding each: Claim / Setup / Result / Evidence / Caveats / Prior work |
| `findings.json` | The findings as data (one object each: claim, metric, value, baseline, n, data file, status) |
| `data/*.csv`, `data/quality_answers.jsonl` | All measurements; columns described in [DATA.md](DATA.md) |
| `scripts/bench/` | Speed and quality benchmarks (llama-server) |
| `scripts/carve/` | Dense-to-MoE carving and memory experiments (PyTorch) |
| [REPRODUCE.md](REPRODUCE.md) | Exact commands |
| `llms.txt` | Short index for AI readers |
| `CITATION.cff` | How to cite |

## Glossary

- **Spilled / offloaded**: part of the model's weights are in system RAM, not GPU memory.
- **Guess-ahead helper (speculative decoding)**: a small model or extra head guesses the next few tokens; the big model
  checks all guesses in one pass and keeps the ones it agrees with. In principle the output matches running without
  the helper; in practice small numerical differences change a few answers (see the "same answer" columns).
- **MTP, DFlash, EAGLE3**: three kinds of helper (multi-token-prediction head, a block-drafting model, a feature-level
  draft head), each trained for one specific model.
- **Guesses**: how many tokens the helper proposes per pass (`--spec-draft-n-max`).
- **Confidence cut-off**: the helper stops guessing when it is less sure than this probability (`--spec-draft-p-min`).
- **Weight streaming**: for large enough batches llama.cpp copies RAM-resident weights to the GPU and computes there
  (threshold `GGML_OP_OFFLOAD_MIN_BATCH`).
- **Mixture of experts (MoE)**: a model where each token uses only a few of many feed-forward "experts"; "26B-A4B"
  means 26B weights in total, about 4B used per token.
- **Carving**: splitting a dense model's feed-forward layers into experts after training.

## License and credit

Text and data: [CC BY 4.0](LICENSE). Scripts in `scripts/`: [PolyForm Noncommercial 1.0.0](scripts/LICENSE).
Please cite with [CITATION.cff](CITATION.cff). Prior work is cited in each finding; none of the methods measured here
is claimed as new. The measurements, the comparisons and the tooling are the contribution.
