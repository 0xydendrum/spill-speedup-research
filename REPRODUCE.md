# Reproduce

Versions used: llama.cpp release **b11798** (commit ec0da70e9) with CUDA, Python 3.12, spilltune 0.2.0, NVIDIA driver
621.14 on Windows 11. For F5: PyTorch with CUDA and transformers (Qwen3.5 support). Close other GPU programs first.

```
pip install git+https://github.com/0xydendrum/spilltune
export LLAMA_SERVER=/path/to/llama-server          # or put llama-server on PATH
export CUDA_DLL_DIR=/path/to/cuda/dlls             # Windows only, if llama-server needs cudart/cublas on PATH
```

Run the speed scripts from the folder that holds the GGUF files.

## Automatic tuning (F2, F3, F6 tables)

```
spilltune gemma-4-31B-it-Q4_K_M.gguf --draft mtp-gemma-4-31B-it.gguf
spilltune gemma-4-26B-A4B-it-UD-Q4_K_M.gguf --draft mtp-gemma-4-26B-A4B-it.gguf
spilltune gemma-4-26B-A4B-it-UD-Q4_K_M.gguf --draft gemma4-26b-a4b-it-dflash-Q4_K_M.gguf --parallel 16
```

## Four-prompt-kind speed (F1, F3, F6)

```
# F1: DFlash 16 guesses, without and with the cut-off
python scripts/bench/kinds_bench.py --label s1-dflash16-ctrl --model gemma-4-31B-it-Q4_K_M.gguf -- \
  --fit on --fit-target 2098 -md gemma4-31b-it-dflash-Q4_K_M.gguf -ngld 99 --spec-type draft-dflash --spec-draft-n-max 16
python scripts/bench/kinds_bench.py --label s1-dflash16-pmin0.7 --model gemma-4-31B-it-Q4_K_M.gguf -- \
  --fit on --fit-target 2098 -md gemma4-31b-it-dflash-Q4_K_M.gguf -ngld 99 --spec-type draft-dflash --spec-draft-n-max 16 --spec-draft-p-min 0.7

# F3: MTP guess count, with weight streaming
python scripts/bench/kinds_bench.py --label s2-mtp8-stream --model gemma-4-31B-it-Q4_K_M.gguf --env GGML_OP_OFFLOAD_MIN_BATCH=8 -- \
  <server args from data/speed_setups.csv, label s2-mtp8-stream>
```

Every row of `data/speed_setups.csv` has its full `server_args`; pass them after `--`, minus the base arguments
`kinds_bench.py` adds itself (`-c 4096 -np 1 --no-webui --cache-ram 0 -lv 4`).

## Many requests at once (F4)

```
python scripts/bench/par_bench.py --label par-tuned --model gemma-4-31B-it-Q4_K_M.gguf --out par.jsonl --levels 1,4,8,16,32 -- \
  --load-mode none --fit on -np 32 -c 10240 -ctk q8_0 -ctv q8_0
python scripts/bench/par_bench.py --label moe-par --model gemma-4-26B-A4B-it-UD-Q4_K_M.gguf --out par.jsonl --levels 1,4,8,16 -- \
  --load-mode none -ngl 99 --n-cpu-moe 18 -np 16 -c 16384 -ctk q8_0 -ctv q8_0
```

## Quality (F1 to F4)

```
uv run --with pyarrow python scripts/bench/prep_items.py         # once: writes data/items.jsonl (fixed seed)
python scripts/bench/eval_quality.py --label g26b-moe-mtp3-pmin07 --model gemma-4-26B-A4B-it-UD-Q4_K_M.gguf \
  --out results/quality.jsonl -- --fit on --fit-target 868 -md mtp-gemma-4-26B-A4B-it.gguf -ngld 99 \
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.7
# reference: the same command with the model fully in VRAM on a larger GPU (label <name>-vram, -ngl 99)
python scripts/bench/compare_quality.py
```

## Carving and memory (F5)

WikiText-2 (raw, v1) goes in `scripts/carve/data/wikitext-2-raw-v1/`.

```
python scripts/carve/carve_proto.py --model path/to/Qwen3.5-2B --experts 32 --shared-fracs 0.25 --topk 4,8
python scripts/carve/sparsity_proto.py --model path/to/Qwen3.5-2B --keep 0.7,0.5,0.4,0.3,0.2
python scripts/carve/heal_proto.py --model path/to/Qwen3.5-2B --configs 0.25:4,0.125:4 --steps 2000 --sequential
python scripts/carve/distill_proto.py --model path/to/Qwen3.5-2B --config 0.25:4 --steps 1500
python scripts/carve/engram_snapshot.py --model path/to/Qwen3.5-2B --config 0.25:4 --rank 32
python scripts/carve/distill_engram.py --model path/to/Qwen3.5-2B --memory snapshot   # also --memory none / zero
```

A 12 GB GPU is enough for the 2B model. With a 248K-token vocabulary, evaluation logits dominate memory: use small
eval batches.

## Forward-filled memory (F7)

Needs the healed carve from `heal_proto.py` (`cache/healed-Qwen3.5-2B-0.25k4.pt`). Data first (FineWeb sample-10BT,
C4 validation, WikiText-2 test, PopQA), tokenized with the Qwen3.5 tokenizer:

```
QWEN35_2B=path/to/Qwen3.5-2B python scripts/carve/prep_engram_data.py
python scripts/carve/engram_diag.py   --model path/to/Qwen3.5-2B                                  # A0
python scripts/carve/engram_diag_b.py --model path/to/Qwen3.5-2B --stage plant --layers 2 --fill-tokens 1000000 --out plant.jsonl
python scripts/carve/engram_diag_b.py --model path/to/Qwen3.5-2B --stage gate --layers 0,1,2,3 --bits 18 --plant-result plant.jsonl --out gate.jsonl
python scripts/carve/engram_diag_b.py --model path/to/Qwen3.5-2B --stage pod --layers 0,1,2,3,4,5,6,7 --bits 19 --out a0b.jsonl   # 48 GB+ GPU
python scripts/carve/engram_gate_c.py --model path/to/Qwen3.5-2B --out a0c.jsonl                     # A0c, fits 12 GB
python scripts/carve/run_manifest.py  --model path/to/Qwen3.5-2B --out manifest.json                 # fingerprint a run
```

The local stages fit the RTX 3060 (`--stage gate` moves layers after the last probe to CPU; peak 9 GB). The full A0b
run with layers 0-7 at bits 19 used 27.6 GB on an A100 (25,800 tokens/s, 15 min). Quick checks: `--smoke` on
`engram_diag.py`, `engram_gate_c.py` and `prep_engram_data.py`; `--stage smoke [--smoke-ppl]` on `engram_diag_b.py`.
