"""Quick llama-server benchmark: load time, GPU layer split, VRAM + shared (spilled) GPU memory,
prompt-processing and generation speed. One JSON line per run is appended to --out.

  python bench_llama.py --label fit-auto --model X.gguf --out results.jsonl -- --fit on
Everything after "--" is passed to llama-server unchanged.
"""
import argparse, json, os, re, subprocess, sys, time, urllib.request
from pathlib import Path

SERVER = os.environ.get("LLAMA_SERVER", "llama-server")
TORCH_LIB = os.environ.get("CUDA_DLL_DIR", "")  # folder with cudart/cublas DLLs, if llama-server needs them on PATH
BASE_ARGS = ["-c", "4096", "-np", "1", "--cache-ram", "0", "--load-mode", "none", "--no-webui", "-lv", "4"]
PARA = ("The river bends twice before it reaches the old mill, where the water slows and turns brown with "
        "silt from the fields upstream. Farmers along its banks have argued for a century about who owns the "
        "right to draw from it during dry summers, and the county records hold a stack of disputes. ")


def post(port, path, body, timeout=600):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def healthy(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def gpu_mem():
    """Dedicated VRAM used (nvidia-smi) and Windows shared GPU memory in use (the sysmem-fallback spill), GiB."""
    out = {}
    try:
        used = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=20).stdout.split()[0]
        out["vram_used_gib"] = round(int(used) / 1024, 2)
    except Exception:
        out["vram_used_gib"] = None
    ps = ("$c = Get-Counter '\\GPU Adapter Memory(*)\\Shared Usage','\\GPU Adapter Memory(*)\\Dedicated Usage' "
          "-ErrorAction SilentlyContinue; $s = ($c.CounterSamples | ? Path -like '*shared usage' | Measure-Object "
          "CookedValue -Maximum).Maximum; $d = ($c.CounterSamples | ? Path -like '*dedicated usage' | "
          "Measure-Object CookedValue -Maximum).Maximum; \"$s $d\"")
    try:
        s, d = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True,
                              timeout=60).stdout.split()
        out["shared_gpu_gib"] = round(float(s) / 2**30, 2)
        out["dedicated_counter_gib"] = round(float(d) / 2**30, 2)
    except Exception:
        out["shared_gpu_gib"] = None
    return out


def main():
    argv = sys.argv[1:]
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--n-predict", type=int, default=128)
    ap.add_argument("--para-reps", type=int, default=7, help="prompt = paragraph x N (7 ~ 400 tokens, 35 ~ 2000)")
    a = ap.parse_args(argv[:argv.index("--")] if "--" in argv else argv)

    env = dict(os.environ, PATH=TORCH_LIB + os.pathsep + os.environ.get("PATH", ""))
    log_path = Path(a.out).with_name(f"server-{a.label}.log")
    cmd = [SERVER, "-m", a.model, "--port", str(a.port)] + BASE_ARGS + extra
    rec = {"label": a.label, "model": Path(a.model).name, "server_args": BASE_ARGS + extra,
           "before": gpu_mem(), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            while not healthy(a.port):
                if proc.poll() is not None:
                    rec["error"] = f"server exited {proc.returncode} during load"
                    break
                if time.time() - t0 > 1200:
                    rec["error"] = "load timeout"
                    break
                time.sleep(2)
            if "error" not in rec:
                rec["load_s"] = round(time.time() - t0, 1)
                rec["loaded"] = gpu_mem()
                text = log_path.read_text(encoding="utf-8", errors="replace")
                m = re.search(r"offloaded (\d+)/(\d+) layers", text)
                rec["gpu_layers"] = f"{m.group(1)}/{m.group(2)}" if m else None
                rec["buffers"] = sorted(set(re.findall(r"(\S+) model buffer size\s*=\s*([\d.]+ MiB)", text)))
                post(a.port, "/completion", {"prompt": "Hello", "n_predict": 8, "temperature": 0})  # warm-up
                prompt = PARA * a.para_reps
                runs = []
                for _ in range(a.runs):
                    r = post(a.port, "/completion", {"prompt": prompt, "n_predict": a.n_predict, "temperature": 0,
                                                     "ignore_eos": True, "cache_prompt": False})
                    t = r["timings"]
                    runs.append({"prompt_n": t["prompt_n"], "pp_tps": round(t["prompt_per_second"], 2),
                                 "gen_n": t["predicted_n"], "tg_tps": round(t["predicted_per_second"], 2)})
                rec["runs"] = runs
                rec["during"] = gpu_mem()
                rec["pp_tps_median"] = sorted(x["pp_tps"] for x in runs)[len(runs) // 2]
                rec["tg_tps_median"] = sorted(x["tg_tps"] for x in runs)[len(runs) // 2]
                c = post(a.port, "/v1/chat/completions", {
                    "messages": [{"role": "user", "content": "What is 17 * 23? Reply with only the number."}],
                    "max_tokens": 2048, "temperature": 0})
                msg = c["choices"][0]["message"]
                rec["sanity_17x23"] = (msg.get("content") or "").strip()[:200]
                rec["sanity_thinking_chars"] = len(msg.get("reasoning_content") or "")
                rec["sanity_finish"] = c["choices"][0].get("finish_reason")
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
    return 1 if "error" in rec else 0


if __name__ == "__main__":
    sys.exit(main())
