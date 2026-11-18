"""Data for Engram v2 step A0 (finding F7). Tokenised with the Qwen3.5 tokenizer, saved as uint32.

  python prep_engram_data.py [--fill-tokens 20000000] [--smoke]

Writes to data/engram/:
  fineweb_basis.npy   64K tokens   (rank-128 drift basis)
  fineweb_fill.npy    20M tokens   (table fill), disjoint from the basis
  c4_eval.npy         64K tokens   (C4 en validation, eval slice)
  c4_investigate.npy  64K tokens   (C4 en validation, disjoint slice for the INVESTIGATE rerun)
  wiki_eval.npy       64K tokens   (WikiText-2 raw test, the same text as wiki.test.raw)
  wiki.test.raw                    (WikiText-2 raw test text, for llama-perplexity in stage A)
  popqa_pilot.jsonl   200 items    (PopQA long tail: lowest s_pop first)
  manifest.json                    (counts, sources, tokenizer)
Documents are joined with the EOS token, so sequences cut at fixed 1024-token windows can cross document boundaries.
"""
import argparse, json, os, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "engram"
TOK = os.environ.get("QWEN35_2B", "Qwen/Qwen3.5-2B")   # local dir or HF id
WIKI = ROOT / "data" / "wikitext-2-raw-v1"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def stream_tokens(tok, texts, n_tokens, batch=256):
    """Tokenise an iterator of documents until n_tokens are collected; documents joined with EOS."""
    eos = tok.eos_token_id
    buf, have, docs = [], 0, 0
    pending = []
    for t in texts:
        if not t or not t.strip():
            continue
        pending.append(t)
        if len(pending) < batch:
            continue
        for ids in tok(pending, add_special_tokens=False).input_ids:
            buf.append(np.asarray(ids + [eos], dtype=np.uint32)); have += len(ids) + 1; docs += 1
        pending = []
        if have >= n_tokens:
            break
    if have < n_tokens and pending:
        for ids in tok(pending, add_special_tokens=False).input_ids:
            buf.append(np.asarray(ids + [eos], dtype=np.uint32)); have += len(ids) + 1; docs += 1
    arr = np.concatenate(buf)[:n_tokens]
    if len(arr) < n_tokens:
        raise SystemExit(f"only {len(arr)} tokens available, wanted {n_tokens}")
    return arr, docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fill-tokens", type=int, default=20_000_000)
    ap.add_argument("--eval-tokens", type=int, default=65536)
    ap.add_argument("--basis-tokens", type=int, default=65536)
    ap.add_argument("--pilot-items", type=int, default=200)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.fill_tokens = 262_144
    OUT.mkdir(parents=True, exist_ok=True)
    import transformers
    from datasets import load_dataset
    tok = transformers.AutoTokenizer.from_pretrained(TOK)
    man = {"tokenizer": TOK, "eos": tok.eos_token_id, "vocab": len(tok), "created": time.strftime("%Y-%m-%d %H:%M:%S")}

    # FineWeb sample-10BT: first basis_tokens -> basis, the following fill_tokens -> fill (disjoint by construction)
    log("FineWeb sample-10BT (streaming)")
    fw = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
    arr, docs = stream_tokens(tok, (r["text"] for r in fw), a.basis_tokens + a.fill_tokens)
    np.save(OUT / "fineweb_basis.npy", arr[: a.basis_tokens])
    np.save(OUT / "fineweb_fill.npy", arr[a.basis_tokens:])
    man["fineweb"] = {"source": "HuggingFaceFW/fineweb sample-10BT train (stream order)", "docs": docs,
                      "basis_tokens": a.basis_tokens, "fill_tokens": a.fill_tokens}
    log(f"FineWeb: {docs} docs, basis {a.basis_tokens}, fill {a.fill_tokens}")

    # C4 en validation: two disjoint consecutive slices
    log("C4 en validation (streaming)")
    c4 = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    arr, docs = stream_tokens(tok, (r["text"] for r in c4), 2 * a.eval_tokens, batch=64)
    np.save(OUT / "c4_eval.npy", arr[: a.eval_tokens])
    np.save(OUT / "c4_investigate.npy", arr[a.eval_tokens:])
    man["c4"] = {"source": "allenai/c4 en validation (stream order)", "docs": docs, "eval_tokens": a.eval_tokens,
                 "investigate_tokens": a.eval_tokens}

    # WikiText-2 raw test: same text as wiki.test.raw (llama-perplexity reads that file)
    df = pd.read_parquet(WIKI / "test-00000-of-00001.parquet")
    raw = "".join(df["text"].tolist())
    (OUT / "wiki.test.raw").write_text(raw, encoding="utf-8", newline="\n")
    ids = np.asarray(tok(raw, add_special_tokens=False).input_ids, dtype=np.uint32)
    if len(ids) < a.eval_tokens:
        raise SystemExit(f"WikiText-2 test has only {len(ids)} tokens")
    np.save(OUT / "wiki_eval.npy", ids[: a.eval_tokens])
    man["wiki"] = {"source": "wikitext-2-raw-v1 test parquet, joined as wiki.test.raw", "total_tokens": int(len(ids)),
                   "eval_tokens": a.eval_tokens}

    # PopQA long tail: lowest subject popularity first
    log("PopQA")
    pq = load_dataset("akariasai/PopQA", split="test").to_pandas().sort_values("s_pop", kind="stable")
    rows = []
    for _, r in pq.head(a.pilot_items).iterrows():
        ans = json.loads(r["possible_answers"]) if isinstance(r["possible_answers"], str) else list(r["possible_answers"])
        rows.append({"id": int(r["id"]), "question": r["question"], "answers": ans, "s_pop": int(r["s_pop"])})
    with open(OUT / "popqa_pilot.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    man["popqa"] = {"source": "akariasai/PopQA test, sorted by s_pop ascending", "pilot_items": len(rows),
                    "total": int(len(pq)), "s_pop_max_in_pilot": rows[-1]["s_pop"]}

    (OUT / "manifest.json").write_text(json.dumps(man, indent=1))
    log("done:", json.dumps(man)[:400])


if __name__ == "__main__":
    main()
