# F4. Serving many requests at once gives spilled models far more total speed than any single-chat setting

## Claim

With part of the model in system RAM, total throughput grows with the number of requests served together. Gemma 4
31B (Q4_K_M) reached 20.72 tokens/sec in total with 32 requests at once, 6.2x a single chat with `--fit` (3.32).
Gemma 4 26B-A4B with experts in RAM reached 83.49 tokens/sec with 16 requests and 98.35 with 32, against about
37 for a single chat. Answer quality with 16 slots and an 8-bit memory cache was unchanged: 74.4% vs 75.3% fully in
VRAM on 450 questions.

## Setup

- `scripts/bench/par_bench.py`: one llama-server with N slots, the same prompt sent N times at once, total generated
  tokens / wall-clock time (prompt reading included). Levels 1, 4, 8, 16, 32.
- 31B: `--fit on -np 32 -c 10240 -ctk q8_0 -ctv q8_0 --load-mode none` (23 of 61 layers on the GPU).
- 26B-A4B: `-ngl 99 --n-cpu-moe 18 -np 16 -c 16384 -ctk q8_0 -ctv q8_0 --load-mode none` (and `-np 32` for 24/32).
- Runs on 2026-11-15 (31B, Qwen) and 2026-11-16 / 2026-11-17 (26B-A4B), EST.

## Result

Total tokens/sec by number of requests at once:

| Model and setup | 1 | 4 | 8 | 16 | 24 | 32 |
|---|---|---|---|---|---|---|
| 31B Q4, `--fit`, f16 cache, `-c 16384` | 2.25 | 6.43 | 6.31 | 9.79 | | 12.63 |
| 31B Q4, `--fit`, 8-bit cache, `-c 10240` | 2.72 | 8.40 | 8.12 | 13.89 | | 20.72 |
| 26B-A4B, experts in RAM, 16 slots | 23.04 | 52.81 | 65.12 | 83.49 | | |
| 26B-A4B, experts in RAM, 32 slots | | | | 82.86 | 89.05 | 98.35 |
| 26B-A4B, spilltune stage 3 (16 slots) | 28.37 | 37.76 | 59.94 | 86.54 | | |
| Qwen3.8-27B Q4, `--fit` | 2.45 | 6.48 | 6.77 | 10.03 | | 13.92 |
| Qwen3.8-27B Q4, `-ngl 27` by hand | 3.08 | 7.79 | 8.78 | 12.61 | | 17.17 |

Single-chat baselines (`data/single_stream.csv`): 31B `--fit` 3.32, 31B with every layer forced onto the GPU and the
NVIDIA driver spilling the overflow into RAM 1.55, 26B-A4B `--fit` 36.65.

On the 31B the total is flat from 4 to 8 requests and climbs again from 16. In llama.cpp, large enough batches copy
RAM-resident weights to the GPU and compute there instead of on the CPU (the threshold is
`GGML_OP_OFFLOAD_MIN_BATCH`); below it the CPU does the RAM part.

On Qwen3.8-27B (a hybrid model), `--fit` with many slots left about 5 GB of VRAM unused; setting `-ngl` by hand gave
17.17 instead of 13.92 at 32 requests. This was a llama.cpp bug in the build used (b11798), fixed upstream by PR
#31087 "llama : respect no_alloc in all memory caches" (merged 2026-10-25).

Quality (450 questions, paired 95% margin):

| Comparison | Score | Reference | Difference | Same answer |
|---|---|---|---|---|
| 26B-A4B, experts in RAM, 16 at once, 8-bit cache vs fully in VRAM | 74.4% | 75.3% | -0.9 +/- 1.4 | 96.7% |

## Evidence

- `data/parallel.csv` (every level of every run), `data/single_stream.csv` (single-chat baselines).
- `data/quality_summary.csv`, comparison "16 requests at once, compressed cache"; answers label `g26b-moe-np16`.
- `data/batch_eval_settings.csv`: the 8-bit cache scored 64 to 67 of 80 and the f16 cache 65 to 66 of 80 on the same
  80 questions, and the 8-bit runs finished faster (175 to 224 s vs 218 to 353 s).

## Caveats

- Totals are for many identical-length requests. Interactive use with one person waiting still gets the single-chat
  speed.
- Wall-clock totals include reading the prompts. On the 26B-A4B with 16 at once, prompt reading took about a quarter
  of the time (from server logs).

## Prior work

- Throughput-oriented inference with weights offloaded to CPU memory: Sheng et al., "FlexGen: High-Throughput
  Generative Inference of Large Language Models with a Single GPU" (ICML 2023).
- Batching many requests in a server: Yu et al., "Orca" (OSDI 2022); Kwon et al., "vLLM / PagedAttention" (SOSP 2023).
- The batching mechanism is llama.cpp's; this finding is a measurement on consumer hardware, not a new method.
