"""Freeze the question set: 300 MMLU-Pro test questions (stratified over its 14 categories) and 150 GSM8K test
questions, sampled with a fixed seed. Run once:  uv run --with pyarrow python prep_items.py
Writes data/items.jsonl; every model is then asked exactly these questions."""
import json, random, urllib.request
from collections import defaultdict
from pathlib import Path
import pyarrow.parquet as pq

SEED = 20261115
N_MMLU, N_GSM = 300, 150
D = Path(__file__).resolve().parent / "data"
D.mkdir(exist_ok=True)
SRC = {"mmlu_pro.parquet": "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/main/data/test-00000-of-00001.parquet",
       "gsm8k.parquet": "https://huggingface.co/datasets/openai/gsm8k/resolve/main/main/test-00000-of-00001.parquet"}
for name, url in SRC.items():
    if not (D / name).exists():
        urllib.request.urlretrieve(url, D / name)

rng = random.Random(SEED)
items = []

mmlu = pq.read_table(D / "mmlu_pro.parquet").to_pylist()
by_cat = defaultdict(list)
for r in mmlu:
    by_cat[r["category"]].append(r)
for rows in by_cat.values():
    rng.shuffle(rows)
cats = sorted(by_cat)
picked, i = [], 0
while len(picked) < N_MMLU:  # round-robin over categories -> as even a split as the categories allow
    rows = by_cat[cats[i % len(cats)]]
    k = i // len(cats)
    if k < len(rows):
        picked.append(rows[k])
    i += 1
for r in picked:
    letters = "ABCDEFGHIJ"[:len(r["options"])]
    opts = "\n".join(f"{l}. {o}" for l, o in zip(letters, r["options"]))
    items.append({"id": f"mmlu-{r['question_id']}", "set": "mmlu_pro", "category": r["category"],
                  "prompt": (f"The following is a multiple choice question about {r['category']}.\n\n"
                             f"Question: {r['question']}\n\nOptions:\n{opts}\n\n"
                             f"Reply with only the letter ({letters[0]}-{letters[-1]}) of the correct option. "
                             "Do not explain."),
                  "gold": r["answer"], "letters": letters})

gsm = pq.read_table(D / "gsm8k.parquet").to_pylist()
for idx in sorted(rng.sample(range(len(gsm)), N_GSM)):
    r = gsm[idx]
    items.append({"id": f"gsm-{idx}", "set": "gsm8k",
                  "prompt": (f"{r['question']}\n\nSolve it step by step, then give the final answer on the last line "
                             "in the form 'Answer: <number>'."),
                  "gold": r["answer"].split("####")[-1].strip().replace(",", "")})

with open(D / "items.jsonl", "w", encoding="utf-8") as f:
    for it in items:
        f.write(json.dumps(it) + "\n")
print(f"{len(items)} items: {sum(i['set'] == 'mmlu_pro' for i in items)} MMLU-Pro over {len(cats)} categories, "
      f"{sum(i['set'] == 'gsm8k' for i in items)} GSM8K")
