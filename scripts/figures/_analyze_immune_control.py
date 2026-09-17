# -*- coding: utf-8 -*-
"""Matched control for the immune-correlation figure.

Some hub genes are themselves immune-lineage markers (CD79A, MZB1, FCRL5, CD14, CHGB), so a
positive expression-immune correlation could be circular.  This script asks whether the hub
behaves differently from the other 99 genes of its own inferred network, on the same samples,
against the same 28 immune cell types -- i.e. a paired comparison that holds the cohort, the
sample set and the cell type fixed.

Output: results/_immune_control.json
"""
import os, sys, json, zipfile, tempfile, time
import numpy as np
import pyreadr
from scipy.stats import rankdata, wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _gene_symbols import canonical

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOCAL = os.path.join(ROOT, "data")
# the immune downloads are fetched rather than shipped; MULTIBATCH_IMMUNE names where they were put
IMM = os.environ.get("MULTIBATCH_IMMUNE") or os.path.join(LOCAL, "immune")
DATA = os.environ.get("MULTIBATCH_DATA") or LOCAL
OUT = os.path.join(ROOT, "results", "_immune_control.json")

NET = json.load(open(os.path.join(ROOT, "results", "_per_cancer_network.json"), encoding="utf-8"))
HUBDISP = json.load(open(os.path.join(ROOT, "results", "_hub_gene_ids.json"), encoding="utf-8"))["hubs"]
ORDER = ["ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "ESCA", "GBM", "HNSC", "KICH",
         "KIRC", "KIRP", "LGG", "LIHC", "LUAD", "LUSC", "MESO", "OV", "PAAD", "PCPG",
         "PRAD", "READ", "SARC", "SKCM", "STAD", "TGCT", "THCA", "UCEC", "UCS", "UVM"]


def norm(s):
    s = str(s).replace(".", "-")
    return s[:15] if len(s) >= 15 else s


# ---------- TIL ----------
z = zipfile.ZipFile(os.path.join(IMM, "TIL_abundance.zip"))
tmp = tempfile.mkdtemp(prefix="til_")
til = {}
for m in [m for m in z.namelist() if m.endswith("_TIL.RData")]:
    c = m.replace("_TIL.RData", "")
    if c in ORDER:
        z.extract(m, tmp)
        df = pyreadr.read_r(os.path.join(tmp, m))["TIL"]
        df.columns = [norm(x) for x in df.columns]
        til[c] = df
CELLS = list(til["BRCA"].index)
print("TIL %d cohorts x %d cells" % (len(til), len(CELLS)), flush=True)


def spearman_block(X, Y):
    """X: genes x n, Y: cells x n -> rho genes x cells (Pearson on average ranks)."""
    Xr = rankdata(X, axis=1).astype(np.float64)
    Yr = rankdata(Y, axis=1).astype(np.float64)
    Xr -= Xr.mean(axis=1, keepdims=True)
    Yr -= Yr.mean(axis=1, keepdims=True)
    Xr /= np.maximum(np.linalg.norm(Xr, axis=1, keepdims=True), 1e-12)
    Yr /= np.maximum(np.linalg.norm(Yr, axis=1, keepdims=True), 1e-12)
    return Xr @ Yr.T


def expr_rows(cancer, want_up):
    p = os.path.join(DATA, "TCGA_%s_HiSeqV2.tsv" % cancer)
    if not os.path.exists(p):
        return {}, []
    out, cols = {}, []
    with open(p, encoding="utf-8", errors="replace") as f:
        hdr = f.readline().rstrip("\n").split("\t")
        cols = [norm(h) for h in hdr[1:]]
        idx = list(range(1, len(hdr)))
        for ln in f:
            parts = ln.rstrip("\n").split("\t")
            # canonical + upper: the hub of ACC is now printed as SHOC1 while Xena labels the row
            # C9orf84. See _gene_symbols.py.
            u = canonical(parts[0].strip()).upper()
            if u in want_up:
                out[u] = np.array([float(parts[i]) if parts[i] not in ("", "NA") else np.nan
                                   for i in idx], dtype=float)
                if len(out) == len(want_up):
                    break
    return out, cols


ONLY = sys.argv[1:]
if ONLY:
    ORDER = [c for c in ORDER if c in ONLY]
    print("restricted to:", ORDER, flush=True)

res, all_hub, all_ctl = {}, [], []
t0 = time.time()
for ci, c in enumerate(ORDER, 1):
    net = NET[c]
    genes = [g for g in net["genes"] if g and g.upper() not in ("?", "NA")]
    hub = net["hub"]
    # canonical + upper on both sides, so the lookup cannot fail on a rename or on case
    want = {canonical(g).upper() for g in genes} | {canonical(hub).upper()}
    rows, cols = expr_rows(c, want)
    if not rows:
        print("[%2d/30] %-5s no expression rows" % (ci, c), flush=True)
        continue
    t = til[c]
    keep = [s for s in t.columns if s in set(cols)]
    if len(keep) < 30:
        print("[%2d/30] %-5s too few shared samples (%d)" % (ci, c, len(keep)), flush=True)
        continue
    cidx = {s: i for i, s in enumerate(cols)}
    jj = [cidx[s] for s in keep]

    names = sorted(rows)
    X = np.vstack([rows[g][jj] for g in names])
    Y = np.vstack([np.asarray(t.loc[cell, keep], dtype=float) for cell in CELLS])
    ok = np.isfinite(X).all(axis=0) & np.isfinite(Y).all(axis=0)
    X, Y = X[:, ok], Y[:, ok]
    if X.shape[1] < 30:
        print("[%2d/30] %-5s too few complete samples (%d)" % (ci, c, X.shape[1]), flush=True)
        continue
    # drop constant rows
    sd = X.std(axis=1)
    good = sd > 1e-9
    names_k = [n for n, g_ in zip(names, good) if g_]
    X = X[good]
    hub_u = canonical(hub).upper()
    if hub_u not in names_k:
        print("[%2d/30] %-5s hub %s not in expression table" % (ci, c, hub), flush=True)
        continue
    R = np.abs(spearman_block(X, Y))                      # genes x cells
    hi = names_k.index(hub_u)
    hub_abs = R[hi]                                       # 28 values
    ctl_abs = np.median(np.delete(R, hi, axis=0), axis=0)  # 28 median-of-controls
    pct = np.array([(R[:, k] < R[hi, k]).mean() for k in range(len(CELLS))])
    res[c] = {"hub": hub, "n_samples": int(X.shape[1]), "n_genes": int(X.shape[0]),
              "hub_abs": hub_abs.tolist(), "ctl_abs": ctl_abs.tolist(),
              "pct": pct.tolist()}
    all_hub.extend(hub_abs.tolist())
    all_ctl.extend(ctl_abs.tolist())
    print("[%2d/30] %-5s %-9s n=%-5d genes=%-4d hub|rho| med %.3f vs control med %.3f | "
          "hub above control in %d/28 | %.0fs"
          % (ci, c, hub, X.shape[1], X.shape[0], np.median(hub_abs), np.median(ctl_abs),
             int((hub_abs > ctl_abs).sum()), time.time() - t0), flush=True)

all_hub = np.array(all_hub)
all_ctl = np.array(all_ctl)
w = wilcoxon(all_hub, all_ctl)
n_up = int((all_hub > all_ctl).sum())
n_tot = len(all_hub)
print("\n=== paired (cohort x cell type) hub vs matched non-hub network genes ===")
print("pairs %d | hub > control in %d (%.1f%%) | Wilcoxon p = %.3g"
      % (n_tot, n_up, 100.0 * n_up / n_tot, w.pvalue))
print("median |rho|  hub %.3f  control %.3f" % (np.median(all_hub), np.median(all_ctl)))

# split by whether the hub is itself an immune-lineage marker
IMMUNE_HUB = {"CD79A", "MZB1", "MGC29506", "FCRL5", "CD14", "CHGB", "CD19", "MS4A1"}
grp = {"immune": ([], []), "non-immune": ([], [])}
for c in res:
    h = res[c]["hub"].upper()
    k = "immune" if h in IMMUNE_HUB else "non-immune"
    grp[k][0].extend(res[c]["hub_abs"])
    grp[k][1].extend(res[c]["ctl_abs"])
for k, (a, b) in grp.items():
    if not a:
        continue
    a, b = np.array(a), np.array(b)
    print("%-12s n=%3d | hub med %.3f vs ctl med %.3f | hub>ctl %d/%d (%.0f%%) | Wilcoxon p=%.3g"
          % (k, len(a), np.median(a), np.median(b), int((a > b).sum()), len(a),
             100.0 * (a > b).mean(), wilcoxon(a, b).pvalue))

summary = {"pairs": n_tot, "hub_above": n_up, "frac_above": n_up / n_tot,
           "wilcoxon_p": float(w.pvalue),
           "median_hub": float(np.median(all_hub)), "median_ctl": float(np.median(all_ctl)),
           "immune_hub_set": sorted(IMMUNE_HUB)}
json.dump({"cells": CELLS, "result": res, "summary": summary},
          open(OUT, "w", encoding="utf-8"), indent=1)
print("\nwrote", OUT)
