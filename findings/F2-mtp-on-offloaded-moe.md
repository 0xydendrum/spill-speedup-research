# F2. An MTP helper makes a mixture-of-experts model with its experts in RAM 1.5x faster on every prompt kind

## Claim

Gemma 4 26B-A4B (a mixture-of-experts model) with its experts in system RAM ran at 37.5 tokens/sec with llama-server's
defaults on a 12 GB GPU. With its MTP helper (3 guesses, cut-off 0.7) it ran at 56.0 tokens/sec (1.50x), and every
prompt kind got faster, prose included. A repeat run gave 55.5 (1.49x). The DFlash helper reached only 42.3 (1.14x).
Answer quality did not change: 74.9% vs 75.3% for the same model fully in VRAM on 450 questions.

## Setup

- Model `gemma-4-26B-A4B-it-UD-Q4_K_M.gguf`, helper `mtp-gemma-4-26B-A4B-it.gguf` (0.43 GB), `--spec-type draft-mtp`,
  `--load-mode none`, `--fit on`, all 31 layers' shared parts on the GPU and the expert weights in RAM.
- Settings chosen unattended by [spilltune](https://github.com/0xydendrum/spilltune) 0.2.0: 12 setups tried per run
  (7 minutes). Speed test = spilltune's four prompts x 192 tokens, one chat at a time.
- Runs on 2026-11-17 at 2:32 AM (first) and 3:41 AM EST (repeat). Quality run 4:41 AM EST.

## Result

| Setup (spilltune run) | Math | Code | Explain | Writing | Overall | Speed-up |
|---|---|---|---|---|---|---|
| Defaults | about 37 each | | | | 37.45 | 1.00x |
| MTP 3 guesses + cut-off 0.7 (run 2) | 66.65 | 66.21 | 49.67 | 47.27 | 56.03 | 1.50x |
| MTP 3 guesses + cut-off 0.7 (run 1) | 66.15 | 65.47 | 49.32 | 46.62 | 55.46 | 1.49x |
| DFlash 6 guesses + cut-off 0.7 | 50.24 | 59.27 | 37.47 | 32.24 | 42.33 | 1.14x |
| DFlash 2 guesses, no cut-off (spilltune 0.1.0) | 50.84 | 51.80 | 37.74 | 31.83 | 41.28 | 1.11x |

CPU threads made no difference to the MTP setup (overall 55.84 to 56.71 tokens/sec with 6, 8, 10, 14 (default) or
20 threads), so the run is limited by how fast weights come out of RAM, not by compute.

Quality (450 questions, paired 95% margin):

| Comparison | Score | Reference | Difference | Same answer |
|---|---|---|---|---|
| MTP 3 + cut-off 0.7 vs fully in VRAM | 74.9% | 75.3% | -0.4 +/- 1.4 | 96.0% |
| MTP 3 + cut-off 0.7 vs no helper (both in RAM) | 74.9% | 75.1% | -0.2 +/- 1.6 | 95.6% |

Why MTP wins where DFlash does not (from the llama-server timing logs, not in `data/`): on this model a plain token
cost about 25 ms; a helper pass cost about 31 ms plus 6.5 ms per guessed token, so a guess only pays if it is accepted
more than about a quarter of the time. DFlash's acceptance by guess position fell 0.60 / 0.38 / 0.22 / 0.15 / 0.11,
so only its first two guesses paid. On a mixture-of-experts model each extra guess can also need different experts,
which have to be read from RAM.

## Evidence

- `data/spilltune_trials.csv`, runs `g26b-moe-mtp-v021`, `g26b-moe-mtp-v020`, `g26b-moe-dflash-v020`, `g26b-moe-gpu`.
- `data/speed_setups.csv`, labels `thr-*` (thread sweep with the MTP setup).
- `data/quality_summary.csv`, comparisons naming "MTP helper (3 guesses"; answers in `data/quality_answers.jsonl`,
  labels `g26b-moe-mtp3-pmin07`, `g26b-moe`, `g26b-moe-vram`.

## Caveats

- One model and one GPU. Whether MTP beats DFlash depends on how well each helper was trained for its model.
- The pass-cost numbers above come from server logs read by hand and are approximate.
- The two speed runs were made an hour apart with the same settings; quality was measured once.

## Prior work

- Multi-token prediction heads: Gloeckle et al., "Better & Faster Large Language Models via Multi-token Prediction"
  (2024); used for speculative decoding in DeepSeek-V3 (DeepSeek-AI, 2024).
- Running MoE experts from host RAM in llama.cpp: PR #29963 "ggml : pipeline parallelism with MoE experts in host
  RAM" and PR #30594 "llama : hybrid MoE expert cache with profile prefilling and zero-upload inference" (both open
  as of 2026-11-17). This finding measures an existing helper type on an existing setup; neither is new here.
