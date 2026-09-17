# -*- coding: utf-8 -*-
"""Spearman co-expression matrices for the genes that carry STRING support.

Reference Fig. 8 is a grid of per-cancer correlation matrices over a PPI gene set. We use the
twenty highest-degree genes of our own STRING-supported network and mark the cells that
correspond to a causal edge we actually called in that cancer type.

Output: results/_corr_matrix.json
"""
import os, sys, json
import numpy as np
from scipy.stats import rankdata
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _gene_symbols import canonical, index

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results")
# TCGA matrices: ./data/ inside the package, or the folder named by MULTIBATCH_DATA
_LOCAL = os.path.join(ROOT, "data")
DATA = os.environ.get("MULTIBATCH_DATA") or _LOCAL
TOP_N = 20

A = json.load(open(os.path.join(RES, "_string_channel_analysis.json"), encoding="utf-8"))
OV = A["overlaps"]

deg = Counter()
for k in OV:
    a, b = k.split("|")
    deg[a] += 1
    deg[b] += 1

genes = [canonical(g) for g, _ in deg.most_common(TOP_N)]
print("top %d genes by supported-edge degree:" % TOP_N)
print("  ", ", ".join("%s(%d)" % (g, deg[g]) for g in genes))

# A Xena row is tied back to one of these genes case-insensitively and across legacy aliases; an
# exact test would miss both "C11orf86" and the renamed "C9orf84". See _gene_symbols.py.
gene_idx = index(genes)

# cancer types ranked by how many supported edges they carry
byc = sorted(A["overlap_by_cancer"].items(), key=lambda kv: (-kv[1], kv[0]))
cancers = [c for c, _ in byc]
print("\ncancers by supported-edge count: %s" % ", ".join("%s:%d" % t for t in byc[:14]))

def read_genes(cancer):
    """stream one Xena table and pull just the rows we need"""
    p = os.path.join(DATA, "TCGA_%s_HiSeqV2.tsv" % cancer)
    if not os.path.exists(p):
        return None
    out = {}
    with open(p, encoding="utf-8", errors="replace") as f:
        hdr = f.readline().rstrip("\n").split("\t")
        ns = len(hdr) - 1
        for ln in f:
            parts = ln.rstrip("\n").split("\t")
            g = gene_idx.get(canonical(parts[0].strip()).upper())
            if g is not None:
                v = np.array([float(x) if x not in ("", "NA", "NaN") else np.nan
                              for x in parts[1:1 + ns]], dtype=float)
                out[g] = v
    return out


res = {}
for ci, c in enumerate(cancers, 1):
    rows = read_genes(c)
    if not rows:
        print("  !! %s: expression table not found" % c)
        continue
    have = [g for g in genes if g in rows]
    miss = [g for g in genes if g not in rows]
    M = np.vstack([rows[g] for g in have])
    keep = np.isfinite(M).all(axis=0)             # samples complete for the whole gene set
    M = M[:, keep]
    n = M.shape[1]
    if n < 30:
        print("  !! %s: only %d complete samples" % (c, n))
        continue

    R = np.apply_along_axis(rankdata, 1, M)
    Z = (R - R.mean(axis=1, keepdims=True)) / np.maximum(R.std(axis=1, keepdims=True), 1e-12)
    C = (Z @ Z.T) / n

    res[c] = {"n": int(n), "genes": have, "rho": C.tolist(),
              "missing": miss}
    lo = C[np.tril_indices_from(C, -1)]
    print("  %-5s n=%5d genes=%2d | rho mean %+.3f | pos %d / neg %d"
          % (c, n, len(have), lo.mean(), int((lo > 0).sum()), int((lo < 0).sum())),
          flush=True)

json.dump({"genes_all": genes, "cancers": cancers, "matrices": res},
          open(os.path.join(RES, "_corr_matrix.json"), "w", encoding="utf-8"))
print("\nwrote results\\_corr_matrix.json (%d cancers)" % len(res))
