"""Generation-speed bench on real chat prompts (GSM8K items), for comparing speculative decoding vs none.
  python spec_bench.py --label X --model M.gguf --out results.jsonl [--items 4 --max-tokens 256] -- <server args>
Reuses the server launch settings of bench_llama.py."""
import argparse, json, os, subprocess, sys, time
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
    ap.add_argument("--port", type=int, default=8092)
    ap.add_argument("--items", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=256)
    a = ap.parse_args(argv[:argv.index("--")] if "--" in argv else argv)
    items = [json.loads(l) for l in open(BENCH / "data" / "items.jsonl", encoding="utf-8")]
    items = [i for i in items if i["set"] == "gsm8k"][:a.items]

    env = dict(os.environ, PATH=B.TORCH_LIB + os.pathsep + os.environ.get("PATH", ""))
    log_path = Path(a.out).with_name(f"server-{a.label}.log")
    cmd = [B.SERVER, "-m", a.model, "--port", str(a.port)] + B.BASE_ARGS + extra
    rec = {"label": a.label, "server_args": B.BASE_ARGS + extra, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
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
            B.post(a.port, "/completion", {"prompt": "Hello", "n_predict": 8, "temperature": 0})
            runs = []
            for it in items:
                c = B.post(a.port, "/v1/chat/completions", {
                    "messages": [{"role": "user", "content": it["prompt"]}],
                    "max_tokens": a.max_tokens, "temperature": 0, "cache_prompt": False})
                t = c.get("timings", {})
                txt = (c["choices"][0]["message"].get("content") or "")
                runs.append({"id": it["id"], "gen_n": t.get("predicted_n"),
                             "tg_tps": round(t.get("predicted_per_second", 0), 2),
                             "draft_n": t.get("draft_n"), "draft_acc": t.get("draft_n_accepted"),
                             "tail": txt.strip()[-60:]})
            rec["runs"] = runs
            rec["during"] = B.gpu_mem()
            n = sum(r["gen_n"] or 0 for r in runs)
            secs = sum((r["gen_n"] or 0) / r["tg_tps"] for r in runs if r["tg_tps"])
            rec["tg_tps_overall"] = round(n / secs, 2) if secs else None
            dn = sum(r["draft_n"] or 0 for r in runs)
            rec["draft_accept_rate"] = round(sum(r["draft_acc"] or 0 for r in runs) / dn, 3) if dn else None
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
