# F3. On a spilled dense model, more guesses lose past 6 for MTP; together with the cut-off MTP reaches 2.61x

## Claim

For Gemma 4 31B (Q4_K_M) with part of the model in system RAM, the MTP helper was fastest at 6 guesses. With 8, 10 or
12 guesses it was slower on every prompt kind (overall 5.19 to 5.32 tokens/sec vs 8.07). Turning on weight streaming
(`GGML_OP_OFFLOAD_MIN_BATCH=8`) won back part of the loss (7.04 to 7.31) but never beat 6 guesses. Combining 6 guesses
with the confidence cut-off 0.7 (F1) gave 8.92 tokens/sec, 2.61x the defaults' 3.42, picked unattended by spilltune.

## Setup

- Model `gemma-4-31B-it-Q4_K_M.gguf`, helper `mtp-gemma-4-31B-it.gguf`, `--spec-type draft-mtp`, 27 of 61 layers on
  the GPU, `--fit on`.
- Speed test = spilltune's four prompts x 192 tokens, one chat at a time (`scripts/bench/kinds_bench.py`).
- Hand runs on 2026-11-16 11:20 to 11:50 PM EST; spilltune run on 2026-11-17 at 3:08 AM EST.

## Result

| Setup | Math | Code | Explain | Writing | Overall | Guesses accepted |
|---|---|---|---|---|---|---|
| MTP 6 | 13.41 | 11.40 | 6.51 | 5.56 | 8.07 | 46.7% |
| MTP 8 | 8.58 | 7.85 | 4.16 | 3.49 | 5.19 | 37.6% |
| MTP 8 + streaming | 11.21 | 10.67 | 5.73 | 4.74 | 7.04 | 37.5% |
| MTP 10 | 9.66 | 7.83 | 4.07 | 3.42 | 5.20 | 30.8% |
| MTP 10 + streaming | 11.95 | 10.55 | 5.68 | 4.72 | 7.06 | 30.6% |
| MTP 12 | 10.17 | 8.33 | 4.08 | 3.46 | 5.32 | 26.5% |
| MTP 12 + streaming | 12.90 | 12.16 | 5.71 | 4.71 | 7.31 | 26.8% |
| MTP 6 + cut-off 0.7 | 11.98 | 11.76 | 7.44 | 6.59 | 8.80 | 71.0% |
| spilltune's pick: MTP 6 + cut-off 0.7, 384 MiB less VRAM held back | 12.78 | 12.52 | 7.73 | 6.21 | 8.92 | - |

For comparison, the same spilltune run found DFlash at best 6.55 (1.92x) and EAGLE3 at best 5.90 (1.73x) on this
model. Earlier math-only runs (GSM8K chat prompts, `data/single_stream.csv`) showed the same shape for MTP: 2 guesses
7.78, 3 guesses 9.44, 4 guesses 10.38, 6 guesses 11.96 tokens/sec.

Why (from the llama-server timing logs, not in `data/`): a full MTP pass took about 447 ms while the batch of guessed
tokens stayed under about 8, and about 750 ms once it went past that. With streaming the pass cost stayed near 553 ms.
So 6 guesses sits just under the step.

## Evidence

- `data/speed_setups.csv`, labels `s2-mtp*`, `x-mtp6-ctrl-pmin*`.
- `data/spilltune_trials.csv`, runs `g31b-q4-mtp-v021`, `g31b-q4-dflash-v020`, `g31b-q4-eagle3-v021`.
- `data/single_stream.csv`, source `spec`, labels `mtp-n*`.

## Caveats

- The step at about 8 tokens is a property of this llama.cpp build, GPU and model split; another setup may step
  elsewhere.
- The quality check for MTP 6 + cut-off 0.7 on this model was cut short by a PC crash: the 300 MMLU-Pro questions
  finished (60.7% vs 59.7% fully in VRAM, +1.0 +/- 1.5) but only 11 of 150 GSM8K questions did. It is listed as
  partial in `findings.json`.

## Prior work

- Speculative decoding and offloading: see F1. Choosing the number of guesses adaptively is studied in, for example,
  SpecExec (Svirschevski et al., 2024). This finding is a measurement on one setup, not a new method.
