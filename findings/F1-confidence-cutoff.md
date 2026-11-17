# F1. A confidence cut-off roughly doubles prose speed for a guess-ahead helper on a spilled model

## Claim

On Gemma 4 31B (Q4_K_M, 17.1 GB) running on a 12 GB GPU with the rest in system RAM, adding a confidence cut-off
(`--spec-draft-p-min 0.7`) to a DFlash helper with 16 guesses raised explanation speed from 2.68 to 5.19 tokens/sec
and writing from 2.21 to 4.37, and also raised math (7.29 to 8.07) and code (6.76 to 7.97). Without the cut-off the
helper made prose slower than no helper at all (3.4). Answer quality did not change: 72.0% vs 72.2% for the same
model fully in VRAM on 450 questions.

## Setup

- Hardware and software: see the README (RTX 3060 12 GB, 32 GB DDR4-3200, llama.cpp b11798, Windows 11).
- Model `gemma-4-31B-it-Q4_K_M.gguf`, helper `gemma4-31b-it-dflash-Q4_K_M.gguf`, `--spec-type draft-dflash
  --spec-draft-n-max 16`, `--fit on --fit-target 2098` (VRAM held back for the helper), 21 of 61 layers on the GPU.
- Speed: `scripts/bench/kinds_bench.py`, spilltune's four prompts (math, code, explanation, story), 192 generated
  tokens each, one chat at a time, context 4096, temperature 0.
- Quality: `scripts/bench/eval_quality.py` on 300 MMLU-Pro + 150 GSM8K questions; reference = same GGUF and llama.cpp
  build fully in VRAM on a rented RTX A6000 48 GB.
- Runs on 2026-11-16 between 10:30 PM and 11:50 PM EST.

## Result

Tokens/sec per prompt kind; "overall" = total tokens / total time over the four prompts.

| Setup | Math | Code | Explain | Writing | Overall | Guesses accepted |
|---|---|---|---|---|---|---|
| No helper (defaults) | 3.4 | 3.4 | 3.4 | 3.4 | 3.4 | - |
| DFlash 16, no cut-off | 7.29 | 6.76 | 2.68 | 2.21 | 3.60 | 15.4% |
| DFlash 16, cut-off 0.5 | 7.94 | 7.26 | 5.04 | 4.58 | 5.88 | 48.1% |
| DFlash 16, cut-off 0.7 | 8.07 | 7.97 | 5.19 | 4.37 | 5.96 | 62.8% |
| DFlash 16, cut-off 0.85 | 9.29 | 7.44 | 5.03 | 4.33 | 5.95 | 75.2% |
| DFlash 16, cut-off 0.95 | 9.46 | 7.91 | 4.69 | 3.86 | 5.68 | 87.2% |
| Repeat: DFlash 16, no cut-off | 7.29 | 6.72 | 2.67 | 2.21 | 3.60 | 15.4% |
| Repeat: DFlash 16, cut-off 0.5 | 8.06 | 7.40 | 5.20 | 4.60 | 5.98 | 48.1% |

Why it works: without the cut-off the helper offers all 16 guesses every time. On the explanation prompt it drafted
1156 tokens to get 111 accepted, and on writing 1380 for 94. With cut-off 0.7 it drafted 168 and 156 and kept 99 and
79. Each wasted guess still costs the big model work, so fewer, surer guesses win.

Quality (450 questions, paired 95% margin):

| Comparison | Score | Reference | Difference | Same answer |
|---|---|---|---|---|
| DFlash 16 + cut-off 0.7 vs fully in VRAM | 72.0% | 72.2% | -0.2 +/- 1.0 | 98.4% |
| DFlash 16 + cut-off 0.7 vs DFlash 16 without | 72.0% | 71.8% | +0.2 +/- 1.2 | 98.0% |

The same cut-off on the MTP helper (6 guesses) traded some math speed for prose: 13.41 / 11.40 / 6.51 / 5.56 without
it, 11.98 / 11.76 / 7.44 / 6.59 with 0.7 (overall 8.07 to 8.80).

On the mixture-of-experts Gemma 4 26B-A4B with experts in RAM, the cut-off helped DFlash (overall 28.5 to 33.2 with
8 guesses and 0.5) but not enough to beat running without a helper (about 37 to 40). See F2 for what does work there.

## Evidence

- `data/speed_setups.csv` and `data/speed_by_kind.csv`, labels `s1-dflash16-*`, `s1-moe-dflash8-*`, `s2-mtp6-ctrl`,
  `x-mtp6-ctrl-pmin*` (`draft_n` / `draft_ok` columns give the drafted and accepted counts).
- `data/quality_summary.csv`, comparisons naming "p-min 0.7"; answers in `data/quality_answers.jsonl`, labels
  `g31b-q4-dflash16-pmin07`, `g31b-q4-dflash16`, `g31b-q4-vram`.

## Caveats

- One GPU, one RAM speed, one model family. The best cut-off value depends on the helper and the model.
- Four prompts of 192 tokens each; speeds on your text will differ. The repeat runs agree within 0.1 tokens/sec
  overall, but most setups were run once.
- The cut-off needs a llama.cpp build with `--spec-draft-p-min`.
- The quality reference ran on a different GPU (RTX A6000). The hardware-only control (Gemma 4 12B fully in VRAM on
  both machines) differed by -0.4 +/- 1.5 points, so small differences of that size are not about the setup.

## Prior work

- The cut-off itself is an existing llama.cpp option (`--spec-draft-p-min`); this finding is a measurement of its
  effect on a model that spills into system RAM, not a new method.
- Speculative decoding: Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (ICML 2023);
  Chen et al., "Accelerating Large Language Model Decoding with Speculative Sampling" (2023).
- Speculative decoding for models offloaded to RAM: Svirschevski et al., "SpecExec: Massively Parallel Speculative
  Decoding for Interactive LLM Inference on Consumer Devices" (2024).
