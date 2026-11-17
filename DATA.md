# Data dictionary

All files are UTF-8. CSV files have a header row. Times are local time on the test PC, EST (UTC-5), written
`YYYY-MM-DD HH:MM:SS`. Speeds are generated tokens per second. Empty cell = not applicable or not recorded.
`server_args` is the llama-server command line with local paths and ports removed.

## data/speed_setups.csv: hand-measured setups, four prompt kinds (one row per setup)

Made by `scripts/bench/kinds_bench.py`.

| Column | Meaning | Unit |
|---|---|---|
| label | Run name (`s1-` cut-off sweep, `s2-` MTP guess count, `x-` MTP + cut-off, `thr-` CPU threads, `ctx` context length) | - |
| time_est | When the run finished | EST |
| model | Main model file | - |
| llama_cpp_build | llama.cpp release | - |
| helper | Helper type: none, dflash, mtp, eagle3 | - |
| helper_file | Helper file name | - |
| guesses | Guesses per pass (`--spec-draft-n-max`) | tokens |
| p_min | Confidence cut-off (`--spec-draft-p-min`); empty = off | probability |
| stream_min_batch | `GGML_OP_OFFLOAD_MIN_BATCH` if set (weight streaming) | tokens |
| ctx | Context size per slot (`-c`) | tokens |
| kv_type | Memory cache type (`-ctk`) | - |
| threads, threads_batch | CPU threads (`-t`, `-tb`); "default" = llama.cpp's choice | count |
| fit_target_mib | VRAM kept free by `--fit` (`--fit-target`) | MiB |
| load_mode | `--load-mode` | - |
| gpu_layers | Layers on the GPU / total, from the server log | - |
| accept_rate | Accepted guesses / drafted guesses over all four prompts | fraction |
| tps_overall | Total tokens / total time over the four prompts | tokens/sec |
| math_tps, code_tps, explain_tps, writing_tps | Speed on each prompt kind | tokens/sec |
| server_args | Full llama-server arguments | - |

## data/speed_by_kind.csv: the same runs, one row per prompt kind

| Column | Meaning | Unit |
|---|---|---|
| label | Joins to `speed_setups.csv` | - |
| kind | math, code, explain or writing | - |
| tps | Speed on this prompt | tokens/sec |
| tokens | Tokens generated | tokens |
| draft_n | Tokens the helper drafted | tokens |
| draft_ok | Drafted tokens the main model accepted | tokens |

## data/spilltune_trials.csv: every trial of every real-GPU spilltune run

Runs named `*-v020` used spilltune 0.2.0 before the cut-off retry fix; `*-v021` used the fixed code (still versioned
0.2.0); `g26b-moe-gpu` and `g31b-q4-gpu` used 0.1.0. Runs that silently fell back to the CPU are not included.

| Column | Meaning | Unit |
|---|---|---|
| run | Run name | - |
| spilltune_version | Version printed by the run | - |
| trial | Trial number within the run | - |
| stage | 1 placement, 2 helper, 3 many requests | - |
| label | spilltune's description of the setup | - |
| model | Main model file | - |
| helper | Helper type | - |
| tps_overall | Total tokens / total time over the four prompts | tokens/sec |
| math_tps, code_tps, explain_tps, writing_tps | Speed per prompt kind | tokens/sec |
| accept_rate | Accepted / drafted guesses | fraction |
| gpu_layers | Layers on the GPU / total | - |
| gpu_mib | GPU buffers llama.cpp reported | MiB |
| over_mib | gpu_mib minus free VRAM; negative = room to spare | MiB |
| load_s | Server load time | seconds |
| gpu_backend | True if the server log showed a GPU backend | - |
| env | Environment variables set for the trial | - |
| server_args | llama-server arguments | - |

## data/parallel.csv: total throughput vs requests at once (one row per level)

| Column | Meaning | Unit |
|---|---|---|
| source | `par_bench <label>` or `spilltune <run> trial <n>` | - |
| time_est | When the run finished (par_bench only) | EST |
| model | Main model file | - |
| gpu_layers | Layers on the GPU / total | - |
| stream_min_batch | `GGML_OP_OFFLOAD_MIN_BATCH` if set | tokens |
| concurrent | Requests sent at once | count |
| tokens | Total tokens generated at this level | tokens |
| wall_s | Wall-clock time, prompt reading included | seconds |
| total_tps | tokens / wall_s | tokens/sec |
| server_args | llama-server arguments | - |

## data/single_stream.csv: one-chat speed runs (placement, helpers on math prompts, file types)

| Column | Meaning | Unit |
|---|---|---|
| source | `side_by_side`, `speed_v2`, `placement`, `avenues` (fixed prompt, `bench_llama.py`) or `spec` (GSM8K chat prompts, `spec_bench.py`) | - |
| label | Run name | - |
| time_est | When the run finished | EST |
| model | Main model file | - |
| gpu_layers | Layers on the GPU / total | - |
| cuda0_mib, host_mib | GPU and pinned host buffers from the server log | MiB |
| prompt_tps | Prompt reading speed (median) | tokens/sec |
| gen_tps | Generation speed (median for bench_llama, overall for spec) | tokens/sec |
| accept_rate | Accepted / drafted guesses (spec only) | fraction |
| server_args | llama-server arguments | - |

## data/quality_answers.jsonl: one graded answer per line

Answer text is not included. Question ids refer to the set built by `scripts/bench/prep_items.py` (fixed seed).

| Field | Meaning |
|---|---|
| label | Setup name (`-vram` = reference fully in VRAM). `g31b-q4-mtp6-pmin07` is partial (311 of 450). |
| id | `mmlu-<question_id>` or `gsm-<index>` |
| set | `mmlu_pro` or `gsm8k` |
| category | MMLU-Pro category (null for GSM8K) |
| gold, pred | Correct answer and the model's parsed answer |
| correct | true / false |
| latency_s | Time for this answer (seconds) |
| prompt_tokens, completion_tokens | Token counts |
| finish | Why generation stopped (`stop`, `length`) |
| hardware | `rtx-3060-12gb-this-pc` or `rtx-a6000-48gb-all-in-vram` |

## data/quality_summary.csv: paired comparisons (output of `compare_quality.py`)

| Column | Meaning | Unit |
|---|---|---|
| comparison | Plain-English name of the setup and its reference | - |
| question_set | `all`, `mmlu_pro` or `gsm8k` | - |
| n | Questions answered by both | count |
| accuracy_pct, reference_pct | Scores | percent |
| diff_pts | accuracy minus reference | percentage points |
| margin_95_pts | 95% margin of the paired difference | percentage points |
| same_answer_pct | Questions where both gave the same answer | percent |
| no_loss | yes if diff_pts + margin_95_pts >= 0 | - |

## data/batch_eval_settings.csv: memory-cache and batch settings on 80 questions

Gemma 4 26B-A4B, 16 slots, context 20480 unless noted, the same 80 questions sent at once.

| Column | Meaning | Unit |
|---|---|---|
| label | `bt-<cache>-<setting>`: cache q8 (8-bit) or f16; ctrl = `-ngl 99 --n-cpu-moe 18`; fit = `--fit` placement; ub2048 = `--fit -b 4096 -ub 2048`; np32 = `--fit`, 32 slots, context 40960; no suffix = `--fit` | - |
| n | Questions answered | count |
| correct | Correct answers | count |
| wall_s | Wall-clock time for all n | seconds |

## data/moe_carve_engram.csv: dense-to-MoE carving and memory experiments (Qwen3.5-2B)

| Column | Meaning | Unit |
|---|---|---|
| experiment | `carve_training_free`, `neuron_skip_no_carve`, `carve_layerwise_heal[_sequential]`, `carve_whole_model_distill`, `engram_snapshot_training_free`, `carve_distill_plus_memory` | - |
| model | Model | - |
| config | Shared fraction, experts per token, router or oracle; or memory mode and scale g | - |
| memory | none, zero (empty table), snapshot, bias, uni, backoff, carve (no memory) | - |
| active_ffn_frac | Fraction of feed-forward weights used per token | fraction |
| train_tokens, steps | Training text and optimizer steps (empty = no training) | tokens, count |
| dense_ppl | Perplexity of the uncarved model on the same eval text | - |
| ppl | WikiText-2 perplexity | - |
| ppl_ratio_vs_dense | ppl / dense_ppl | ratio |
| router_recall | Share of the truly most active experts the router picked | fraction |
