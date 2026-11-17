"""Aggregate generation throughput vs number of concurrent requests (one server, many slots).
  python par_bench.py --label X --model M.gguf --out results.jsonl --levels 1,4,8,16,32 -- <server args incl. -np>
Shows where llama.cpp stops computing RAM-resident weights on the CPU and starts streaming them to the GPU."""
import argparse, json, os, subprocess, sys, threading, time
from pathlib import Path

BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH))
import bench_llama as B  # noqa: E402


def main():
    argv = sys.argv[1:]
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=8093)
    ap.add_argument("--levels", default="1,4,8,16,32")
    ap.add_argument("--max-tokens", type=int, default=96)
    a = ap.parse_args(argv[:argv.index("--")] if "--" in argv else argv)
    items = [json.loads(l) for l in open(BENCH / "data" / "items.jsonl", encoding="utf-8")]
    prompts = [i["prompt"] for i in items if i["set"] == "gsm8k"]

    env = dict(os.environ, PATH=B.TORCH_LIB + os.pathsep + os.environ.get("PATH", ""))
    log_path = Path(a.out).with_name(f"server-{a.label}.log")
    base = [x for x in B.BASE_ARGS]
    i = base.index("-np"); del base[i:i + 2]          # slots come from the caller's args
    i = base.index("-c"); del base[i:i + 2]
    cmd = [B.SERVER, "-m", a.model, "--port", str(a.port)] + base + extra
    rec = {"label": a.label, "server_args": base + extra, "env_offload_min_batch": os.environ.get("GGML_OP_OFFLOAD_MIN_BATCH"),
           "time": time.strftime("%Y-%m-%d %H:%M:%S"), "levels": []}
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            while not B.healthy(a.port):
                if proc.poll() is not None:
                    raise RuntimeError(f"server exited {proc.returncode} during load")
                if time.time() - t0 > 1200:
                    raise RuntimeError("load timeout")
                time.sleep(2)
            rec["load_s"] = round(time.time() - t0, 1)
            rec["loaded"] = B.gpu_mem()
            text = log_path.read_text(encoding="utf-8", errors="replace")
            import re
            m = re.search(r"offloaded (\d+)/(\d+) layers", text)
            rec["gpu_layers"] = f"{m.group(1)}/{m.group(2)}" if m else None
            B.post(a.port, "/completion", {"prompt": "Hello", "n_predict": 8, "temperature": 0})
            for n in [int(x) for x in a.levels.split(",")]:
                res = [None] * n

                def one(k):
                    c = B.post(a.port, "/v1/chat/completions", {
                        "messages": [{"role": "user", "content": prompts[k % len(prompts)]}],
                        "max_tokens": a.max_tokens, "temperature": 0, "cache_prompt": False}, timeout=3600)
                    res[k] = c.get("timings", {}).get("predicted_n") or c["usage"]["completion_tokens"]

                ts = time.time()
                th = [threading.Thread(target=one, args=(k,)) for k in range(n)]
                for t in th: t.start()
                for t in th: t.join()
                wall = time.time() - ts
                toks = sum(r or 0 for r in res)
                lv = {"concurrent": n, "tokens": toks, "wall_s": round(wall, 1), "agg_tps": round(toks / wall, 2)}
                rec["levels"].append(lv)
                print(json.dumps(lv), flush=True)
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
    rec["total_s"] = round(time.time() - t0, 1)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
