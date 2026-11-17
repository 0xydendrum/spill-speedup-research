"""Dependency-free GGUF tensor lister: prints per-group byte totals for placement planning.
usage: python -I gguf_tensors.py model.gguf"""
import re, struct, sys
from collections import defaultdict

# ggml type -> (block elements, block bytes)
T = {0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24), 8: (32, 34), 9: (32, 36),
     10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176), 14: (256, 210), 15: (256, 292),
     16: (256, 66), 17: (256, 74), 18: (256, 98), 19: (256, 50), 20: (32, 18), 21: (256, 110), 22: (256, 82),
     23: (256, 136), 24: (1, 1), 25: (1, 2), 26: (1, 4), 27: (1, 8), 28: (1, 8), 29: (256, 56), 30: (1, 2),
     34: (256, 54), 35: (256, 66), 39: (32, 17)}


def main(path):
    f = open(path, "rb")
    rd = lambda fmt: struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]
    rs = lambda: f.read(rd("Q")).decode("utf-8", "replace")
    assert f.read(4) == b"GGUF"
    rd("I"); nt = rd("Q"); nkv = rd("Q")
    sz = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
    def val(t):
        if t == 8: return rs()
        if t == 9:
            et = rd("I"); n = rd("Q")
            return [val(et) for _ in range(n)] if n < 64 else (n, [val(et) for _ in range(n)][:0])
        return rd(sz[t])
    meta = {}
    for _ in range(nkv):
        k = rs(); meta[k] = val(rd("I"))
    arch = meta.get("general.architecture")
    print("arch", arch, {k: v for k, v in meta.items() if k.startswith(f"{arch}.") and not isinstance(v, tuple)
                         and "chat" not in k})
    groups = defaultdict(int); types = defaultdict(set); total = 0
    for _ in range(nt):
        name = rs(); nd = rd("I"); dims = [rd("Q") for _ in range(nd)]; t = rd("I"); rd("Q")
        n = 1
        for d in dims: n *= d
        be, bb = T[t]; b = n // be * bb; total += b
        m = re.match(r"blk\.(\d+)\.(.*)", name)
        key = f"L{int(m.group(1)):02d}." + ("ffn" if m.group(2).startswith("ffn_") and "norm" not in m.group(2)
                                            else "attn+other") if m else name
        groups[key] += b; types[key].add(t)
    G = 1e9
    print(f"total {total/G:.2f} GB")
    for k in sorted(k for k in groups if not k.startswith("L")): print(f"  {k:28s} {groups[k]/G:6.3f} GB types {sorted(types[k])}")
    ffn = sum(v for k, v in groups.items() if k.endswith(".ffn")); att = sum(v for k, v in groups.items() if k.endswith("attn+other"))
    print(f"  all layers ffn {ffn/G:.2f} GB, attn+other {att/G:.2f} GB")
    for k in sorted(k for k in groups if k.startswith("L")): print(f"  {k} {groups[k]/1e6:7.1f} MB")


if __name__ == "__main__":
    main(sys.argv[1])
