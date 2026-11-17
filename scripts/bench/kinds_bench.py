"""Per-prompt-kind speed test (spilltune's 4 prompts: math, code, explain, writing; 4 x 192 tokens), one setup per call.
  python kinds_bench.py --label X --model M.gguf [--env K=V ...] -- <server args>
Run from the folder that holds the GGUF files (model/helper paths are relative to it). Base args match spilltune
(-c 4096 -np 1 --no-webui --cache-ram 0 -lv 4), so results compare with spilltune-runs/*/trials.jsonl.
Appends one row to kinds.jsonl next to this script and prints a one-line summary."""
import argparse, json, os, sys, time
from pathlib import Path

BENCH = Path(__file__).resolve().parent
# needs spilltune installed: pip install git+https://github.com/0xydendrum/spilltune
sys.path.insert(0, str(BENCH))
from spilltune import server, prompts, hw  # noqa: E402
import bench_llama as B  # noqa: E402

BASE = ["-c", "4096", "-np", "1", "--no-webui", "--cache-ram", "0", "-lv", "4"]


def main():
    argv = sys.argv[1:]
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--env", action="append", default=[], help="K=V for the server environment")
    ap.add_argument("--max-tokens", type=int, default=192)
    ap.add_argument("--out", default=str(BENCH / "kinds.jsonl"))
    a = ap.parse_args(argv[:argv.index("--")] if "--" in argv else argv)
    os.environ["PATH"] = B.TORCH_LIB + os.pathsep + os.environ.get("PATH", "")   # CUDA DLLs, else CPU
    env = dict(kv.split("=", 1) for kv in a.env)
    log = BENCH / "logs-kinds" / f"server-{a.label}.log"
    r = server.run_trial([B.SERVER], a.model, BASE + extra, env, prompts.pick(4), a.max_tokens, log, hw.gpus())
    row = {"label": a.label, "model": Path(a.model).name, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
           "args": BASE + extra, "env": env, "server": "b11798", "tps_avg": r.get("tps"), "accept": r.get("accept"),
           "gpu_layers": r.get("gpu_layers"), "over_mib": r.get("over_mib"), "load_s": r.get("load_s"),
           "total_s": r.get("total_s"), "error": r.get("error"),
           "kinds": {x["kind"]: {"tps": round(x["tps"], 2), "tokens": x["tokens"], "draft_n": x["draft_n"],
                                 "draft_ok": x["draft_ok"], "tail": x["text"].strip()[-60:]}
                     for x in r.get("runs", [])}}
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    k = row["kinds"]
    print(f"RESULT {a.label}: " + (f"ERROR {row['error']}" if row["error"] else
          " ".join(f"{n}={k[n]['tps']}" for n in k) + f" avg={row['tps_avg']} acc={row['accept']} gpu={row['gpu_layers']}"))
    return 1 if row["error"] else 0


if __name__ == "__main__":
    sys.exit(main())
