"""Fast setups vs the same model run normally, fully in VRAM (the reference), on the same 450 questions.

Reads results/quality.jsonl (local runs) + results/vram_ref/out/quality_vram_gpu*.jsonl (rented RTX A6000 48 GB, same llama.cpp b11798 and GGUF files, model fully in VRAM).
For each pair: both scores, the paired difference (fast - reference) with a 95% margin, and how often
the two gave the same answer. Writes results/quality_vs_vram.json for the report.

  python compare_quality.py
"""
import glob, json, math
from pathlib import Path

HERE = Path(__file__).resolve().parent
# (fast label, reference label, plain-English name of the fast setup)
PAIRS = [
    ("g12b", "g12b-vram", "Gemma 4 12B on this PC's GPU (hardware-only control: both fully in VRAM)"),
    ("g26b-moe", "g26b-moe-vram", "Gemma 4 26B-A4B, experts in RAM"),
    ("g26b-moe-dflash8", "g26b-moe-vram", "Gemma 4 26B-A4B, experts in RAM + DFlash helper (8 guesses)"),
    ("g26b-moe-np16", "g26b-moe-vram", "Gemma 4 26B-A4B, experts in RAM, 16 requests at once, compressed cache"),
    ("qwen35-a3b", "qwen35-a3b-vram", "Qwen3.5-35B-A3B, part in RAM (--fit)"),
    ("g31b-iq3", "g31b-iq3-vram", "Gemma 4 31B IQ3, part in RAM (--fit)"),
    ("g31b-iq3-dflash16", "g31b-iq3-vram", "Gemma 4 31B IQ3, part in RAM + DFlash helper (16 guesses)"),
    ("g31b-q4-dflash16", "g31b-q4-vram", "Gemma 4 31B Q4, part in RAM + DFlash helper (16 guesses)"),
    ("g31b-q4-dflash16-pmin07", "g31b-q4-vram", "Gemma 4 31B Q4, part in RAM + DFlash helper (16 guesses, p-min 0.7)"),
    ("g31b-q4-dflash16-pmin07", "g31b-q4-dflash16", "Gemma 4 31B Q4 + DFlash 16: with p-min 0.7 vs without"),
    ("g26b-moe-mtp3-pmin07", "g26b-moe-vram", "Gemma 4 26B-A4B, experts in RAM + MTP helper (3 guesses, p-min 0.7)"),
    ("g26b-moe-mtp3-pmin07", "g26b-moe", "Gemma 4 26B-A4B in RAM: MTP 3 + p-min 0.7 vs no helper"),
    ("g31b-q4-mtp6-pmin07", "g31b-q4-vram", "Gemma 4 31B Q4, part in RAM + MTP helper (6 guesses, p-min 0.7)"),
    # the smaller IQ3 file is itself a speed choice: compare it with the Q4 file run normally
    ("g31b-iq3-dflash16", "g31b-q4-vram", "Gemma 4 31B IQ3 + DFlash on this PC vs the Q4 file fully in VRAM"),
    # both fully in VRAM on the pod: the cost of the smaller file alone, no RAM, no helper
    ("g31b-iq3-vram", "g31b-q4-vram", "Gemma 4 31B: smaller IQ3 file vs the Q4 file, both fully in VRAM"),
]
SETS = ("mmlu_pro", "gsm8k")


def load():
    rows = {}
    files = [HERE / "results" / "quality.jsonl"] + [Path(p) for p in glob.glob(str(HERE / "results" / "vram_ref" / "out" / "quality_vram_gpu*.jsonl"))]
    for f in files:
        if f.exists():
            for line in open(f, encoding="utf-8"):
                r = json.loads(line)
                rows.setdefault(r["label"], {})[r["id"]] = r
    return rows


def paired(fast, ref, ids):
    d = [int(fast[i]["correct"]) - int(ref[i]["correct"]) for i in ids]
    n = len(d)
    mean = sum(d) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1)) if n > 1 else 0.0
    return {
        "n": n,
        "fast_pct": round(100 * sum(fast[i]["correct"] for i in ids) / n, 1),
        "ref_pct": round(100 * sum(ref[i]["correct"] for i in ids) / n, 1),
        "diff_pts": round(100 * mean, 1),
        "margin_pts": round(100 * 1.96 * sd / math.sqrt(n), 1),
        "same_answer_pct": round(100 * sum(str(fast[i]["pred"]) == str(ref[i]["pred"]) for i in ids) / n, 1),
        "fast_only_right": sum(fast[i]["correct"] and not ref[i]["correct"] for i in ids),
        "ref_only_right": sum(ref[i]["correct"] and not fast[i]["correct"] for i in ids),
    }


def main():
    rows = load()
    out = []
    for fl, rl, name in PAIRS:
        if fl not in rows or rl not in rows:
            print(f"--  {name}: waiting for {', '.join(l for l in (fl, rl) if l not in rows)}")
            continue
        f, r = rows[fl], rows[rl]
        res = {"fast": fl, "ref": rl, "name": name}
        for s in SETS + ("all",):
            ids = sorted(i for i in f if i in r and (s == "all" or f[i]["set"] == s))
            if ids:
                res[s] = paired(f, r, ids)
        out.append(res)
        a = res.get("all", {})
        print(f"{name}\n    all {a.get('n')}: fast {a.get('fast_pct')}% vs VRAM {a.get('ref_pct')}%  "
              f"diff {a.get('diff_pts'):+} ±{a.get('margin_pts')} pts, same answer {a.get('same_answer_pct')}%"
              + "".join(f"\n    {s}: {res[s]['fast_pct']}% vs {res[s]['ref_pct']}% ({res[s]['diff_pts']:+} ±{res[s]['margin_pts']}), "
                        f"same {res[s]['same_answer_pct']}%, n={res[s]['n']}" for s in SETS if s in res))
    (HERE / "results" / "quality_vs_vram.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
