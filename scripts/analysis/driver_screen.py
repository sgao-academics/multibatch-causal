# -*- coding: utf-8 -*-
"""W2-F: does variance-based selection systematically exclude driver genes?

For every cohort the MAD is computed as in the main pipeline (mad = median(|x - median(x)|)),
canonical driver genes are located within the resulting ranking, and the dispersion
percentile they occupy is compared against the cut used to select the panel.

Output: results/_driver_screen.json
"""
import os, json, glob
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
DATA = os.path.join(_ROOT, 'data')
RES = os.path.join(_ROOT, 'results')
PANELS = os.path.join(DATA, 'panels')
FIG = os.path.join(_ROOT, 'figures')

OUT = os.path.join(RES, '_driver_screen.json')
N_TOP = 100

DRIVERS = """TP53 KRAS MYC PIK3CA PTEN EGFR BRAF NRAS APC CTNNB1 RB1 CDKN2A SMAD4
ARID1A BRCA1 BRCA2 ERBB2 MET ALK IDH1 IDH2 VHL NF1 AKT1 CCND1 MDM2 GATA3 FOXA1
ESR1 AR TP63 SOX2 NFE2L2 KEAP1 STK11 ATM CHEK2 FAT1 NOTCH1 NOTCH2 CASP8 HRAS
PTPN11 MLH1 MSH2 MSH6 PMS2 BAP1 SETD2 B2M HLA-A JAK2 STAT3 MTOR TSC1 TSC2
CTNNB1 EZH2 KMT2D CREBBP EP300 FGFR1 FGFR2 FGFR3 KIT PDGFRA RET SRC ABL1""".split()

files = sorted(glob.glob(os.path.join(DATA, 'TCGA_*_HiSeqV2.tsv')))
print('cohort files =', len(files))
rows = []
drv_pct = {g: [] for g in sorted(set(DRIVERS))}
for f in files:
    coh = os.path.basename(f).split('_')[1]
    X = pd.read_csv(f, sep='\t', index_col=0)
    X = X.apply(pd.to_numeric, errors='coerce').dropna(how='all')
    mad = (X.sub(X.median(axis=1), axis=0)).abs().median(axis=1)
    mad = mad.sort_values(ascending=False)
    n = len(mad)
    rank = {g: (np.where(mad.index == g)[0][0] + 1 if g in mad.index else None)
            for g in set(DRIVERS)}
    top = set(mad.index[:N_TOP])
    ndrv_top = len(top & set(DRIVERS))
    pcts = {g: (100.0 * r / n) for g, r in rank.items() if r is not None}
    for g, p in pcts.items():
        drv_pct[g].append(p)
    med_drv_pct = float(np.median(list(pcts.values()))) if pcts else float('nan')
    rows.append(dict(cohort=coh, n_samples=int(X.shape[1]), n_genes=int(n),
                     drivers_in_panel_found=ndrv_top,
                     median_driver_MAD_percentile=round(med_drv_pct, 2)))
    print('%-6s n=%4d | drivers in MAD-top100 = %d | median driver MAD pct = %.1f%%'
          % (coh, X.shape[1], ndrv_top, med_drv_pct))

print()
print('%-10s %8s %8s %10s' % ('gene', 'median%', 'min%', 'best-rank%'))
agg = []
for g, ps in drv_pct.items():
    if not ps:
        continue
    agg.append((g, float(np.median(ps)), float(np.min(ps))))
for g, m, b in sorted(agg, key=lambda t: t[1])[:25]:
    print('%-10s %8.1f %8.1f %10.1f' % (g, m, b, b))

any_top = sum(r['drivers_in_panel_found'] for r in rows)
med_all = float(np.median([r['median_driver_MAD_percentile'] for r in rows]))
print()
print('TOTAL: driver genes entering MAD-top100 across all %d cohorts = %d' % (len(rows), any_top))
print('median across cohorts of (median driver MAD percentile) = %.1f%%' % med_all)

json.dump(dict(n_cohorts=len(rows), top_k=N_TOP, per_cohort=rows,
               driver_percentiles={g: dict(median=float(np.median(ps)), min=float(np.min(ps)),
                                           max=float(np.max(ps)))
                                   for g, ps in drv_pct.items() if ps},
               total_driver_hits_in_top100=any_top,
               median_driver_percentile=med_all),
          open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('wrote', OUT)
