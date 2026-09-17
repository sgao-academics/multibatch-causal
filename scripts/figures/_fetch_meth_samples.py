# -*- coding: utf-8 -*-
"""Stream the 5 GB PANCAN methylation matrix once and keep only
(a) probes of our hub genes and (b) samples that have an immune-abundance profile.

usage:  python fetch_meth_samples.py [--limit N]     (--limit only peeks)
output: results/_immune_meth.npz   arrays: probes(str), samples(str), M(float32)
"""
import os, sys, json, time, zipfile
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMM = os.path.join(ROOT, "data", "immune")
P = os.path.join(os.environ.get("MULTIBATCH_DATA", os.path.join(ROOT, "data")), "PANCAN_methylation.tsv")
OUT = os.path.join(ROOT, "results", "_immune_meth.npz")

LIMIT = None
if "--limit" in sys.argv:
    LIMIT = int(sys.argv[sys.argv.index("--limit") + 1])

# ---- probes of interest ----
man = pd.read_csv(os.path.join(IMM, "HM450_gencode.tsv.gz"), sep="\t",
                  usecols=["probeID", "genesUniq"], low_memory=False).dropna(subset=["genesUniq"])
man["genesUniq"] = man["genesUniq"].astype(str)
WANT = ["ADH1B", "C11orf86", "C9orf84", "CD79A", "CD14", "CHGB", "CXCL5", "DPT", "FCRL5",
        "GABRA1", "GSTM1", "GSTT1", "JPH2", "KCNV1", "KDM5D", "KRT6A", "LTF", "MZB1",
        "MUC5B", "MYH8", "NAPSA", "PCDHGA10", "PLP1", "RPS4Y1", "SFRP2", "SPATA13",
        "SPRR2E", "TBX18", "TEKT1", "SHOC1"]
p2g = {}
for w in WANT:
    idx = man.index[man["genesUniq"].str.split(";").apply(lambda s: w in s)]
    for pid in man.loc[idx, "probeID"]:
        p2g[pid] = w
print("hub probes: %d for %d genes" % (len(p2g), len(WANT)))

# ---- samples that have TIL profiles ----
til_samples = set()
z = zipfile.ZipFile(os.path.join(IMM, "TIL_abundance.zip"))
import tempfile, pyreadr
tmp = tempfile.mkdtemp(prefix="til_")
for m in [m for m in z.namelist() if m.endswith("_TIL.RData")]:
    z.extract(m, tmp)
    df = pyreadr.read_r(os.path.join(tmp, m))["TIL"]
    til_samples |= {str(c).replace(".", "-")[:15] for c in df.columns}
print("samples with TIL profile: %d" % len(til_samples))

# ---- stream ----
t0 = time.time()
with open(P, encoding="utf-8", errors="replace") as f:
    hdr = [c.strip().strip('"') for c in f.readline().rstrip("\n").split("\t")]
    col_of = {}
    for i, c in enumerate(hdr):
        if i == 0:
            continue
        n = c.replace(".", "-")[:15]
        if n in til_samples:
            col_of.setdefault(n, i)
    print("PANCAN columns: %d | matched to a TIL sample: %d" % (len(hdr), len(col_of)))
    order = sorted(col_of)
    idx = [col_of[s] for s in order]

    rows, hits, nline = {}, 0, 0
    for ln in f:
        nline += 1
        if LIMIT and nline > LIMIT:
            break
        pid = ln.split("\t", 1)[0].strip().strip('"')
        g = p2g.get(pid)
        if g is None:
            continue
        parts = ln.rstrip("\n").split("\t")
        vals = np.full(len(idx), np.nan, dtype=np.float32)
        for k, j in enumerate(idx):
            if j < len(parts):
                v = parts[j].strip().strip('"')
                if v and v != "NA":
                    try:
                        vals[k] = float(v)
                    except ValueError:
                        pass
        rows[pid] = (g, vals)
        hits += 1
        if hits % 50 == 0:
            el = time.time() - t0
            print("  lines %d | probes hit %d | %.0f lines/s | %.0f s elapsed"
                  % (nline, hits, nline / el, el), flush=True)

el = time.time() - t0
print("scanned %d lines in %.0f s (%.0f lines/s), hit %d probes" % (nline, el, nline / el, len(rows)))
if LIMIT:
    print("LIMIT mode - nothing written")
    raise SystemExit(0)

probes = sorted(rows)
M = np.vstack([rows[p][1] for p in probes])
genes = [rows[p][0] for p in probes]
np.savez_compressed(OUT, probes=np.array(probes), genes=np.array(genes),
                    samples=np.array(order), M=M.astype(np.float32))
print("wrote %s  M=%s  (%.1f MB)" % (OUT, M.shape, os.path.getsize(OUT) / 1e6))
