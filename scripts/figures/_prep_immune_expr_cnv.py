# -*- coding: utf-8 -*-
"""Spearman correlation between each cohort's own hub gene and 28 immune cell types,
for the expression and CNV modalities. Mirrors the layout of the reference paper's Fig 5.

Output: results/_immune_corr_expr_cnv.json
  { cancer: { "hub": SYMBOL, "n": n_samples,
              "expr": {"<cell>": [rho, p], ...},
              "cnv":  {"<cell>": [rho, p], ...} } }
"""
import os, sys, json, gzip, zipfile, tempfile
import numpy as np
import pyreadr
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _gene_symbols import canonical, index

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# The immune downloads (TISIDB TIL archive, HM450 manifest) are fetched rather than shipped, so
# this directory is empty in a fresh checkout: MULTIBATCH_IMMUNE points at wherever they were put.
LOCAL = os.path.join(ROOT, "data")
IMM = os.environ.get("MULTIBATCH_IMMUNE") or os.path.join(LOCAL, "immune")
# TCGA matrices: ./data/ inside the package, or the folder named by MULTIBATCH_DATA
DATA = os.environ.get("MULTIBATCH_DATA") or LOCAL
OUT = os.path.join(ROOT, "results", "_immune_corr_expr_cnv.json")

HUB = {"ACC": "SHOC1", "BLCA": "SFRP2", "BRCA": "ADH1B", "CESC": "SPRR2E", "CHOL": "MUC5B",
       "COAD": "C11orf86", "ESCA": "KRT6A", "GBM": "KCNV1", "HNSC": "CHGB", "KICH": "GSTM1",
       "KIRC": "SLC22A6", "KIRP": "KDM5D", "LGG": "GABRA1", "LIHC": "DPT", "LUAD": "TEKT1",
       "LUSC": "NAPSA", "MESO": "GSTM1", "OV": "MZB1", "PAAD": "RPS4Y1", "PCPG": "PLP1",
       "PRAD": "LTF", "READ": "CXCL5", "SARC": "JPH2", "SKCM": "FCRL5", "STAD": "RPS4Y1",
       "TGCT": "GSTM1", "THCA": "CD79A", "UCEC": "TBX18", "UCS": "MYH8", "UVM": "PCDHGA10"}


def norm(s):
    s = s.replace(".", "-")
    return s[:15] if len(s) >= 15 else s


# ---------- 1. TIL ----------
z = zipfile.ZipFile(os.path.join(IMM, "TIL_abundance.zip"))
tmp = tempfile.mkdtemp(prefix="til_")
members = [m for m in z.namelist() if m.endswith("_TIL.RData")]
til = {}
for m in members:
    c = m.replace("_TIL.RData", "")
    if c not in HUB:
        continue
    z.extract(m, tmp)
    df = pyreadr.read_r(os.path.join(tmp, m))["TIL"]
    df.columns = [norm(c_) for c_ in df.columns]
    til[c] = df
print("TIL cohorts loaded:", len(til), sorted(til))

CELLS = list(til["BRCA"].index)
print("%d immune cell types: %s" % (len(CELLS), CELLS))


# ---------- 2. expression / CNV extraction ----------
# A row is claimed by canonical symbol, compared case-insensitively, so the approved name finds
# Xena's legacy spellings ("MGC29506" for MZB1, "C9orf84" for SHOC1). See _gene_symbols.py.


def expr_row(cancer, gene, keep):
    """stream the Xena HiSeqV2 table for one gene, return (sample_list, values)"""
    p = os.path.join(DATA, "TCGA_%s_HiSeqV2.tsv" % cancer)
    if not os.path.exists(p):
        return None, None
    want = canonical(gene)
    with open(p, encoding="utf-8", errors="replace") as f:
        hdr = f.readline().rstrip("\n").split("\t")
        colmap = {norm(h): i for i, h in enumerate(hdr) if i > 0}
        idx = [colmap[s] for s in keep if s in colmap]
        cols = [s for s in keep if s in colmap]
        for ln in f:
            parts = ln.rstrip("\n").split("\t")
            if canonical(parts[0].strip()) == want:
                vals = np.array([float(parts[i]) if parts[i] not in ("", "NA") else np.nan
                                 for i in idx], dtype=float)
                return cols, vals
    return cols, np.full(len(cols), np.nan)


cnv_rows = None


def cnv_loader():
    """GISTIC is one file for all cohorts; load only what we need, once."""
    global cnv_rows
    if cnv_rows is not None:
        return cnv_rows
    p = os.path.join(DATA, "Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes.gz")
    hdr, rows = None, {}
    with gzip.open(p, "rt", encoding="utf-8", errors="replace") as f:
        hdr = [h.strip().strip('"') for h in f.readline().rstrip("\n").split("\t")]
        colmap = {norm(h): i for i, h in enumerate(hdr) if i > 0}
        want = {canonical(g) for g in HUB.values()}
        want_idx = {}
        for g in want:
            want_idx.setdefault(g.upper(), g)
        for ln in f:
            parts = ln.rstrip("\n").split("\t")
            # GISTIC carries the same symbols as the Xena matrices, so this lookup has to be
            # canonical and case-insensitive as well, or a hub's copy-number row is dropped.
            g = want_idx.get(canonical(parts[0].strip().strip('"')).upper())
            if g is not None:
                rows[g] = parts
                if len(rows) == len(want):
                    break
    cnv_rows = (colmap, rows)
    print("GISTIC rows found for %d/%d hub symbols: %s"
          % (len(rows), len(want), sorted(rows)))
    return cnv_rows


result, rows_for_fdr = {}, []
for cancer in sorted(til):
    gene = HUB[cancer]
    t = til[cancer]
    keep = list(t.columns)
    entry = {"hub": gene, "n": len(keep), "expr": {}, "cnv": {}}

    cols_e, vals_e = expr_row(cancer, gene, keep)
    have_e = {c: v for c, v in zip(cols_e, vals_e)} if cols_e else {}

    colmap_c, rows_c = cnv_loader()
    have_c = {}
    if gene in rows_c:
        parts = rows_c[gene]
        for s in keep:
            i = colmap_c.get(s)
            if i is not None and i < len(parts) and parts[i] not in ("", "NA"):
                try:
                    have_c[s] = float(parts[i])
                except ValueError:
                    pass

    n_bad = sum(1 for s in keep if s in have_e and not np.isfinite(have_e[s]))
    if n_bad == len(keep):
        print("  !! %s: gene row for %s is entirely missing from the expression table"
              % (cancer, gene))
    for cell in CELLS:
        y_all = t.loc[cell]
        # expression
        s_e = [s for s in keep if s in have_e and np.isfinite(have_e[s]) and np.isfinite(y_all[s])]
        if len(s_e) >= 30:
            r, p = stats.spearmanr([have_e[s] for s in s_e], [y_all[s] for s in s_e])
            entry["expr"][cell] = [float(r), float(p)]
            rows_for_fdr.append(("expr", cancer, cell, p))
        # cnv
        s_c = [s for s in keep if s in have_c and np.isfinite(y_all[s])]
        if len(s_c) >= 30 and len(set(have_c[s] for s in s_c)) > 2:
            r, p = stats.spearmanr([have_c[s] for s in s_c], [y_all[s] for s in s_c])
            entry["cnv"][cell] = [float(r), float(p)]
            rows_for_fdr.append(("cnv", cancer, cell, p))

    result[cancer] = entry
    ne = len(entry["expr"])
    nc = len(entry["cnv"])
    print("  %-6s hub=%-10s n=%-5d expr %2d/%d  cnv %2d/%d"
          % (cancer, gene, len(keep), ne, len(CELLS), nc, len(CELLS)))

# ---------- 3. BH-FDR per modality ----------
def bh(pvals):
    p = np.asarray(pvals, float)
    o = np.argsort(p)
    n = len(p)
    q = np.empty(n)
    prev = 1.0
    for rank, i in enumerate(o[::-1]):
        k = n - rank
        prev = min(prev, p[i] * n / k)
        q[i] = prev
    return q


for mod in ("expr", "cnv"):
    idx = [i for i, r in enumerate(rows_for_fdr) if r[0] == mod]
    q = bh([rows_for_fdr[i][3] for i in idx])
    for j, i in enumerate(idx):
        _, c, cell, _ = rows_for_fdr[i]
        result[c][mod][cell].append(float(q[j]))

summary = {}
for mod in ("expr", "cnv"):
    tot = sum(len(result[c][mod]) for c in result)
    sig = sum(1 for c in result for cell in result[c][mod] if result[c][mod][cell][2] < 0.05)
    summary[mod] = {"total": tot, "fdr_sig": sig,
                    "median_rho": float(np.median([result[c][mod][k][0] for c in result
                                                   for k in result[c][mod]])) if tot else 0.0}
print("\nsummary:", json.dumps(summary, indent=1))

with open(OUT, "w", encoding="utf-8") as f:
    json.dump({"cells": CELLS, "result": result, "summary": summary}, f, indent=1)
print("wrote", OUT)
