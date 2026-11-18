"""Record exactly what a test ran with: weights, tokenizer, every module in the model, environment, data, script.

Writes one JSON file. Run it before every test (locally and on pods) and keep the JSON next to the results;
TESTS.md / tests.jsonl point at it. Nothing here loads real weights: the module tree is built on the meta device.

  python run_manifest.py --model <hf dir> --script engram_diag.py --args "<the run's argv>" \
      --artifact cache/healed-Qwen3.5-2B-0.25k4.pt --data data/engram --out runs/a0/manifest.json [--no-hash]

Hashing the 4.5 GB safetensors takes ~15 s on NVMe; --no-hash records size + mtime only.
"""
import argparse, collections, hashlib, importlib, json, os, platform, subprocess, sys, time
from pathlib import Path

PROBE = "The quick brown fox, 1234 — Ünïcode café!\n\tdef f(x): return x**2  # 日本語"


def sha256(path, do_hash=True):
    p = Path(path)
    st = p.stat()
    rec = {"path": str(p), "bytes": st.st_size, "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime))}
    if do_hash:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        rec["sha256"] = h.hexdigest()
    return rec


def versions():
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("torch", "transformers", "accelerate", "numpy", "pandas", "safetensors", "tokenizers", "huggingface_hub",
                "fla", "causal_conv1d", "flash_attn", "bitsandbytes"):
        try:
            out[mod] = getattr(importlib.import_module(mod), "__version__", "installed")
        except Exception:
            out[mod] = None                                     # None = not installed (e.g. fla -> torch fallback kernels)
    try:
        import torch
        out["cuda"] = torch.version.cuda
        out["cudnn"] = torch.backends.cudnn.version()
        out["gpu"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        out["tf32_matmul_default"] = torch.backends.cuda.matmul.allow_tf32
    except Exception as e:
        out["torch_error"] = str(e)[:200]
    try:
        out["nvidia_smi"] = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                            "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        out["nvidia_smi"] = None
    return out


def hf_revision(model_dir):
    """Commit sha the files were downloaded at (huggingface_hub local_dir metadata or snapshot path), else None."""
    md = Path(model_dir) / ".cache" / "huggingface" / "download"
    revs = set()
    if md.is_dir():
        for f in md.glob("*.metadata"):
            first = f.read_text(errors="replace").splitlines()[:1]
            if first:
                revs.add(first[0].strip())
    parts = Path(model_dir).resolve().parts
    if "snapshots" in parts:
        revs.add(parts[parts.index("snapshots") + 1])
    return sorted(revs) or None


def model_section(model_dir, do_hash):
    import torch, transformers
    d = Path(model_dir)
    cfg_raw = json.loads((d / "config.json").read_text())
    text = cfg_raw.get("text_config", cfg_raw)
    lt = text.get("layer_types")
    sec = {
        "dir": str(d.resolve()), "hf_revision": hf_revision(d),
        "files": [sha256(p, do_hash and p.suffix in (".safetensors", ".bin", ".json", ".jinja", ".txt", ".model"))
                  for p in sorted(d.iterdir()) if p.is_file()],
        "config": {k: cfg_raw.get(k) for k in ("architectures", "model_type", "tie_word_embeddings", "torch_dtype", "dtype")},
        "text_config": {k: text.get(k) for k in ("model_type", "num_hidden_layers", "hidden_size", "intermediate_size",
                                                  "num_attention_heads", "num_key_value_heads", "head_dim", "vocab_size",
                                                  "full_attention_interval", "linear_num_key_heads", "linear_num_value_heads",
                                                  "hidden_act", "rms_norm_eps", "rope_theta", "max_position_embeddings")},
        "layer_types": lt, "layer_type_counts": dict(collections.Counter(lt)) if lt else None,
        "has_vision_config": "vision_config" in cfg_raw,
    }
    # module tree on the meta device: which classes run, per layer, in forward order of registration
    try:
        cfg = transformers.AutoConfig.from_pretrained(d)
        # mirror carve_proto.load_model: AutoModelForCausalLM first. A multimodal checkpoint (config with text_config)
        # loads as the text-only causal LM there, so build from text_config; ImageTextToText only as the last resort.
        m = None
        with torch.device("meta"):
            for label, build in (("causal_lm(text_config)", lambda: transformers.AutoModelForCausalLM.from_config(cfg.text_config)),
                                 ("causal_lm(config)", lambda: transformers.AutoModelForCausalLM.from_config(cfg)),
                                 ("image_text_to_text(config)", lambda: transformers.AutoModelForImageTextToText.from_config(cfg))):
                if label.endswith("(text_config)") and not hasattr(cfg, "text_config"):
                    continue
                try:
                    m = build(); sec["built_via"] = label; break
                except Exception as e:
                    sec.setdefault("build_errors", {})[label] = f"{type(e).__name__}: {str(e)[:160]}"
        sec["loaded_class"] = type(m).__name__
        sec["module_class_counts"] = dict(collections.Counter(type(x).__name__ for x in m.modules()))
        sec["params_total"] = sum(p.numel() for p in m.parameters())
        layers = next((mod for name, mod in m.named_modules() if name.endswith("layers") and len(mod) > 4), None)
        if layers is not None:
            per_layer = []
            for i, layer in enumerate(layers):
                per_layer.append({"i": i, "class": type(layer).__name__,
                                  "children": [f"{n}:{type(c).__name__}" for n, c in layer.named_children()]})
            sec["decoder_layers"] = per_layer
        top = [f"{n}:{type(c).__name__}" for n, c in m.named_children()]
        sec["top_level"] = top
        del m
    except Exception as e:
        sec["module_tree_error"] = f"{type(e).__name__}: {str(e)[:300]}"
    return sec


def tokenizer_section(model_dir, do_hash):
    import transformers
    t = transformers.AutoTokenizer.from_pretrained(model_dir)
    ids = t(PROBE, add_special_tokens=False)["input_ids"]
    sec = {"class": type(t).__name__, "len": len(t), "vocab_size": getattr(t, "vocab_size", None),
           "eos": [t.eos_token, t.eos_token_id], "bos": [t.bos_token, t.bos_token_id], "pad": [t.pad_token, t.pad_token_id],
           "unk": [t.unk_token, t.unk_token_id], "n_added_tokens": len(t.get_added_vocab()),
           "special_tokens": t.all_special_tokens[:40], "add_bos_default": getattr(t, "add_bos_token", None),
           "probe_text": PROBE, "probe_ids": ids, "probe_roundtrip_ok": t.decode(ids) == PROBE}
    for fn in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja"):
        p = Path(model_dir) / fn
        if p.exists():
            sec[fn] = sha256(p, do_hash).get("sha256", p.stat().st_size)
    return sec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--script", default=None)
    ap.add_argument("--args", default="")
    ap.add_argument("--artifact", action="append", default=[], help="extra files to fingerprint (healed .pt, LoRA, ...)")
    ap.add_argument("--data", default=None, help="data dir; every file is fingerprinted, manifest.json copied in")
    ap.add_argument("--note", action="append", default=[], help="free text: what the script changes in the model, venue, ...")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-hash", action="store_true")
    a = ap.parse_args()
    do_hash = not a.no_hash
    t0 = time.time()
    rec = {"created": time.strftime("%Y-%m-%d %H:%M:%S %z"), "host": platform.node(), "cwd": os.getcwd(),
           "argv_of_run": a.args, "notes": a.note, "env": versions()}
    if a.script:
        rec["script"] = sha256(a.script, True)
        for dep in ("carve_proto.py", "heal_proto.py"):              # local modules the moe-carve scripts import
            p = Path(a.script).resolve().parent / dep
            if p.exists():
                rec.setdefault("script_deps", []).append(sha256(p, True))
    rec["model"] = model_section(a.model, do_hash)
    rec["tokenizer"] = tokenizer_section(a.model, do_hash)
    rec["artifacts"] = [sha256(p, do_hash) for p in a.artifact]
    if a.data:
        dd = Path(a.data)
        rec["data"] = {"dir": str(dd.resolve()), "files": [sha256(p, do_hash) for p in sorted(dd.iterdir()) if p.is_file()]}
        if (dd / "manifest.json").exists():
            rec["data"]["manifest"] = json.loads((dd / "manifest.json").read_text())
            m = rec["data"]["manifest"]
            rec["data"]["tokenizer_matches_manifest"] = (m.get("vocab") == rec["tokenizer"]["len"]
                                                         and m.get("eos") == rec["tokenizer"]["eos"][1])
    rec["manifest_seconds"] = round(time.time() - t0, 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
    print(f"manifest -> {a.out} ({rec['manifest_seconds']} s): model rev {rec['model']['hf_revision']}, "
          f"class {rec['model'].get('loaded_class')}, tokenizer len {rec['tokenizer']['len']}, "
          f"data/tokenizer match {rec.get('data', {}).get('tokenizer_matches_manifest')}")


if __name__ == "__main__":
    main()
