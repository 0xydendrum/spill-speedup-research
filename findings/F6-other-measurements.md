# F6. Smaller measurements: what did not matter, and what to watch

Each item uses the same hardware and llama.cpp build as the README. Numbers are tokens/sec unless noted.

## Claim

Several settings people often tune made little or no difference on a spilled model, and a few cost more than expected:
CPU threads, `--fit-target` margins and tensor-level offload were flat; letting the NVIDIA driver spill VRAM into RAM
halved speed; a 32K-token context cost 7 to 11%; the n-gram and small-model helpers did nothing.

## Setup

Same as F1 to F4. Sources per item are listed under Evidence.

## Result

| Measurement | Result |
|---|---|
| CPU threads, 26B-A4B + MTP (6 / 8 / 10 / default 14 / 20) | 55.84 to 56.71 overall: flat (RAM speed is the limit) |
| `--fit-target` margin, 31B single chat (default / 256 MiB) | 3.32 / 3.53 |
| Tensor-level offload (`-ngl 99 -ot`, feed-forward weights of layers 16-59 or 20-59 in RAM, everything else on GPU) | 3.14 / 3.43 vs 3.32 with `--fit`: a tie |
| Every layer forced onto the GPU (`-ngl 99`), driver spills the overflow into RAM | 1.55 vs 3.32 with `--fit`: 2.1x slower |
| 31B + MTP 6 + cut-off 0.7, context 4K vs 32K (f16 cache) vs 32K (8-bit cache) | 8.90 / 7.89 / 8.24 overall (28 / 24 / 25 of 61 layers on GPU) |
| n-gram lookup helper (`ngram`), 31B, math prompts | 3.39 vs 3.40 without, 2.7% of guesses accepted |
| Small same-family model as helper (Gemma 4 E2B Q3_K_M, 4 guesses), 31B, math prompts | 2.72 vs 3.40 without: slower |
| Smaller 31B file (UD-IQ3_XXS) instead of Q4_K_M, single chat `--fit` | 8.51 (47 of 61 layers fit) vs 3.32 |
| Same IQ3 file vs Q4_K_M, both fully in VRAM, 450 questions | 72.9% vs 72.2% (+0.7 +/- 2.6), but only 86.9% same answers |
| Hardware-only control: Gemma 4 12B fully in VRAM, this PC vs the reference GPU | 69.8% vs 70.2% (-0.4 +/- 1.5), 94.2% same answers |

The smaller IQ3 file is fast because more of it fits on the GPU, and its score is within the margin, but it changes
about one answer in eight. It is listed as a measurement, not a recommendation.

Unattended spilltune results for every model tried (best setup / defaults):

| Model | Helper | Defaults | Best | Speed-up |
|---|---|---|---|---|
| Gemma 4 31B Q4_K_M | MTP | 3.42 | 8.92 | 2.61x |
| Gemma 4 31B Q4_K_M | DFlash | 3.42 | 6.55 | 1.92x |
| Gemma 4 31B Q4_K_M | EAGLE3 | 3.42 | 5.90 | 1.73x |
| Gemma 4 26B-A4B | MTP | 37.45 | 56.03 | 1.50x |
| Gemma 4 26B-A4B | DFlash | 37.18 | 42.33 | 1.14x |
| Qwen3.5-35B-A3B IQ4_XS | none | 47.17 | 50.16 | 1.06x |
| Qwen3.8-27B Q4_K_M | none | 4.41 | 5.03 | 1.14x |

## Evidence

- Threads and context: `data/speed_setups.csv`, labels `thr-*`, `ctx*`.
- Placement, driver spill, IQ3 speed: `data/single_stream.csv`, sources `placement`, `side_by_side`, `avenues`.
- Helpers on math prompts: `data/single_stream.csv`, source `spec`, labels `ngram`, `e2b-n4`, `s0-fit`.
- Quality: `data/quality_summary.csv`.
- spilltune runs: `data/spilltune_trials.csv`.

## Caveats

- The `spec` rows use GSM8K math prompts only, which flatter helpers compared with the four-kind test (for example,
  26B-A4B + DFlash 8 ran at 55.8 on math prompts but 28.5 overall on the four kinds).
- Single runs unless stated.

## Prior work

- The driver's spill behaviour is NVIDIA's "CUDA - Sysmem Fallback Policy" (default on since driver 536.40). The
  companion tool [spillsafe](https://github.com/0xydendrum/spillsafe) budgets for it.
