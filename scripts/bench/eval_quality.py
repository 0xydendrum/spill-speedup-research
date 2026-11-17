"""Ask one model every question in data/items.jsonl through llama-server; one JSON line per answer in --out.
Resumable: questions already in --out are skipped. Thinking is off, temperature 0.

  python eval_quality.py --label g12b --model X.gguf --out results/quality.jsonl -- --fit on
"""
import argparse, json, re, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from bench_llama import SERVER, TORCH_LIB, BASE_ARGS, post, healthy
import os

HERE = Path(__file__).resolve().parent
MAX_TOKENS = {"mmlu_pro": 16, "gsm8k": 768}


def grade(item, text):
    t = text.strip()
    if item["set"] == "mmlu_pro":
        L = item["letters"]
        # only an explicit "Answer: X" or a reply that starts with the letter counts; no guessing inside prose
        m = re.search(rf"[Aa]nswer\s*[:is]*\s*\(?([{L}])\b", t) or re.match(rf"\**\(?([{L}])(?:\)|\.|:|\*|\s*$)", t)
        pred = m.group(1) if m else None
        return pred, pred == item["gold"]
    m = re.findall(r"[Aa]nswer\s*:\s*\$?\s*(-?[\d,]*\.?\d+)", t) or re.findall(r"-?[\d,]*\.?\d+", t)
    pred = m[-1].replace(",", "") if m else None
    try:
        ok = pred is not None and abs(float(pred) - float(item["gold"])) < 1e-6
    except ValueError:
        ok = False
    return pred, ok


def main():
    argv = sys.argv[1:]
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=8092)
    ap.add_argument("--sets", default="mmlu_pro,gsm8k")
    ap.add_argument("--concurrency", type=int, default=1, help="questions in flight at once (for -np N setups)")
    ap.add_argument("--limit", type=int, default=0, help="first N questions of each set (smoke tests)")
    a = ap.parse_args(argv[:argv.index("--")] if "--" in argv else argv)

    items = [json.loads(l) for l in open(HERE / "data" / "items.jsonl", encoding="utf-8")]
    items = [i for i in items if i["set"] in a.sets.split(",")]
    if a.limit:
        items = [i for s in a.sets.split(",") for i in [x for x in items if x["set"] == s][:a.limit]]
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {(r["label"], r["id"]) for r in map(json.loads, open(out, encoding="utf-8"))}
    todo = [i for i in items if (a.label, i["id"]) not in done]
    print(f"{a.label}: {len(todo)} of {len(items)} questions to ask", flush=True)
    if not todo:
        return 0

    env = dict(os.environ, PATH=TORCH_LIB + os.pathsep + os.environ.get("PATH", ""))
    log = open(out.with_name(f"server-quality-{a.label}.log"), "a", encoding="utf-8", errors="replace")
    proc = subprocess.Popen((SERVER if isinstance(SERVER, list) else [SERVER]) + ["-m", a.model, "--port", str(a.port)] + BASE_ARGS + extra,
                            stdout=log, stderr=subprocess.STDOUT, env=env)
    t0 = time.time()
    try:
        while not healthy(a.port):
            if proc.poll() is not None or time.time() - t0 > 1200:
                print("server failed to start", flush=True)
                return 1
            time.sleep(2)
        print(f"server up in {time.time() - t0:.0f} s", flush=True)
        state = {"ok": 0, "k": 0}
        lock = threading.Lock()
        f = open(out, "a", encoding="utf-8")

        def ask(it):
            t = time.time()
            r = post(a.port, "/v1/chat/completions", {
                "messages": [{"role": "user", "content": it["prompt"]}], "temperature": 0,
                "max_tokens": MAX_TOKENS[it["set"]], "chat_template_kwargs": {"enable_thinking": False}},
                timeout=1800)
            dt = time.time() - t
            ch = r["choices"][0]
            text = ch["message"].get("content") or ""
            pred, ok = grade(it, text)
            u = r.get("usage", {})
            line = json.dumps({"label": a.label, "id": it["id"], "set": it["set"],
                               "category": it.get("category"), "gold": it["gold"], "pred": pred, "correct": ok,
                               "latency_s": round(dt, 2), "prompt_tokens": u.get("prompt_tokens"),
                               "completion_tokens": u.get("completion_tokens"),
                               "finish": ch.get("finish_reason"),
                               "thinking_chars": len(ch["message"].get("reasoning_content") or ""),
                               "output": text[:3000]}) + chr(10)
            with lock:
                f.write(line); f.flush()
                state["ok"] += ok; state["k"] += 1
                k = state["k"]
                if k % 25 == 0 or k == len(todo):
                    print(f"{a.label}: {k}/{len(todo)} done, {state['ok']} correct, {time.time() - t0:.0f} s", flush=True)

        try:
            with ThreadPoolExecutor(max(1, a.concurrency)) as ex:
                list(ex.map(ask, todo))
        finally:
            f.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
