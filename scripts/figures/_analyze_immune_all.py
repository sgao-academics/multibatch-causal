# -*- coding: utf-8 -*-
"""Unified correlation of each cohort's own hub gene against the 28 TISIDB immune cell
types, across the three modalities (expression / CNV / methylation).
Writes results/_immune_corr_all.json and prints the summary that decides the figure."""
import os, sys, json, zipfile, tempfile
import numpy as np
import pyreadr
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _gene_symbols import canonical

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# The immune downloads are fetched rather than shipped: MULTIBATCH_IMMUNE points at wherever the
# TIL archive was put, and ./data/immune is the default place.
IMM = os.environ.get("MULTIBATCH_IMMUNE") or os.path.join(ROOT, "data", "immune")
OUT = os.path.join(ROOT, "results", "_immune_corr_all.json")

HUB = {"ACC": "SHOC1", "BLCA": "SFRP2", "BRCA": "ADH1B", "CESC": "SPRR2E", "CHOL": "MUC5B",
       "COAD": "C11orf86", "ESCA": "KRT6A", "GBM": "KCNV1", "HNSC": "CHGB", "KICH": "GSTM1",
       "KIRC": "SLC22A6", "KIRP": "KDM5D", "LGG": "GABRA1", "LIHC": "DPT", "LUAD": "TEKT1",
       "LUSC": "NAPSA", "MESO": "GSTM1", "OV": "MZB1", "PAAD": "RPS4Y1", "PCPG": "PLP1",
       "PRAD": "LTF", "READ": "CXCL5", "SARC": "JPH2", "SKCM": "FCRL5", "STAD": "RPS4Y1",
       "TGCT": "GSTM1", "THCA": "CD79A", "UCEC": "TBX18", "UCS": "MYH8", "UVM": "PCDHGA10"}
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
    if c not in HUB:
        continue
    z.extract(m, tmp)
    df = pyreadr.read_r(os.path.join(tmp, m))["TIL"]
    df.columns = [norm(x) for x in df.columns]
    til[c] = df
CELLS = list(til["BRCA"].index)
print("TIL: %d cohorts, %d cell types" % (len(til), len(CELLS)))

# ---------- expression (already computed) ----------
expr_json = json.load(open(os.path.join(ROOT, "results", "_immune_corr_expr_cnv.json"),
                           encoding="utf-8"))["result"]

# ---------- CNV ----------
cna = json.load(open(os.path.join(ROOT, "results", "_immune_cna_samples.json"), encoding="utf-8"))
cna_vals = {c: {norm(k): v for k, v in d["values"].items()} for c, d in cna.items()}

# ---------- methylation ----------
mz = np.load(os.path.join(ROOT, "results", "_immune_meth.npz"), allow_pickle=True)
mprobes, mgenes, msamples, MM = list(mz["probes"]), list(mz["genes"]), [norm(s) for s in mz["samples"]], mz["M"]
midx = {}
for i, g in enumerate(mgenes):
    midx.setdefault(g, []).append(i)
mcol = {s: j for j, s in enumerate(msamples)}
meth_avg = {}
for g, ii in midx.items():
    v = MM[ii].astype(np.float64)
    v[~np.isfinite(v)] = np.nan
    cnt = np.isfinite(v).sum(axis=0)
    s = np.where(np.isfinite(v), v, 0.0).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
    meth_avg[g] = m


def corr(x, y, min_n=30, min_uniq=3):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < min_n or len(np.unique(x[ok])) < min_uniq:
        return None
    r, p = stats.spearmanr(x[ok], y[ok])
    return (float(r), float(p), int(ok.sum()))


res, fdr_rows = {}, []
for c in ORDER:
    t = til[c]
    # the methylation table carries its own spelling, so the row is claimed canonically
    hub_c = canonical(HUB[c])
    gene_m = next((g for g in meth_avg if canonical(g) == hub_c), None)
    entry = {"hub": hub_c, "n": len(t.columns), "expr": {}, "cnv": {}, "meth": {}}
    for cell in CELLS:
        y = t.loc[cell]
        # expression: reuse
        e = expr_json.get(c, {}).get("expr", {}).get(cell)
        if e:
            entry["expr"][cell] = e[:2] + [None]
        # CNV
        if c in cna_vals:
            common = [s for s in t.columns if s in cna_vals[c]]
            if common:
                r = corr([cna_vals[c][s] for s in common], [y[s] for s in common])
                if r:
                    entry["cnv"][cell] = list(r)
        # methylation
        if gene_m:
            common = [s for s in t.columns if s in mcol]
            if common:
                r = corr([meth_avg[gene_m][mcol[s]] for s in common], [y[s] for s in common])
                if r:
                    entry["meth"][cell] = list(r)
    res[c] = entry
    print("  %-6s %-9s n=%-5d expr %2d | cnv %2d | meth %2d" %
          (c, HUB[c], len(t.columns), len(entry["expr"]), len(entry["cnv"]), len(entry["meth"])))


def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    n = len(p)
    q = np.empty(n)
    prev = 1.0
    for rank, i in enumerate(o[::-1]):
        k = n - rank
        prev = min(prev, p[i] * n / k)
        q[i] = prev
    return q


summary = {}
for mod in ("expr", "cnv", "meth"):
    rows = [(c, cell) for c in res for cell in res[c][mod]]
    if not rows:
        summary[mod] = {"total": 0}
        continue
    q = bh([res[c][mod][cell][1] for c, cell in rows])
    for (c, cell), qv in zip(rows, q):
        res[c][mod][cell][2] = float(qv)
    rho = np.array([res[c][mod][cell][0] for c, cell in rows])
    sig = np.array([res[c][mod][cell][2] < 0.05 for c, cell in rows])
    summary[mod] = {
        "total": len(rows), "sig": int(sig.sum()),
        "median_abs_rho": float(np.median(np.abs(rho))),
        "sig_pos": int((sig & (rho > 0)).sum()), "sig_neg": int((sig & (rho < 0)).sum()),
        "cohorts": int(len({c for c, _ in rows})),
    }
    print("\n%s: %d tests | FDR<0.05 %d (%.1f%%) | median |rho| %.3f | +%d / -%d | %d cohorts"
          % (mod, len(rows), sig.sum(), 100 * sig.mean(), np.median(np.abs(rho)),
             summary[mod]["sig_pos"], summary[mod]["sig_neg"], summary[mod]["cohorts"]))

# per-cohort standout tally for the expression modality
print("\nexpression: number of immune cell types with FDR<0.05, per cohort")
for c in ORDER:
    n = sum(1 for cell in res[c]["expr"] if res[c]["expr"][cell][2] < 0.05)
    print("   %-6s %2d/28" % (c, n))

json.dump({"cells": CELLS, "order": ORDER, "result": res, "summary": summary},
          open(OUT, "w", encoding="utf-8"), indent=1)
print("\nwrote", OUT)
