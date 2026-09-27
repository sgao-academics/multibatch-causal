# -*- coding: utf-8 -*-
"""W2-M1: build the second panel (robustness) -- equally variance-extreme, disjoint genes.

For every cohort a MAD is computed; genes are ranked by their cross-cohort median MAD and
100 are drawn (fixed seed) from ranks 201-2000, excluding every gene of panel A.  The panel
tests whether the recurring set is specific to the particular genes chosen.

Output: data/panels/panel_B_100genes.json
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

E = RES  # scratch alias
OUT = os.path.join(PANELS, 'panel_B_100genes.json')
SEED = 20260926
N = 100
LO, HI = 201, 2000

cur = set(json.load(open(os.path.join(PANELS, 'panel_A_100genes.json'), encoding='utf-8'))['panel'])
files = sorted(glob.glob(os.path.join(DATA, 'TCGA_*_HiSeqV2.tsv')))
print('cohorts =', len(files))

acc = None
for f in files:
    X = pd.read_csv(f, sep='\t', index_col=0)
    X = X.apply(pd.to_numeric, errors='coerce').dropna(how='all')
    mad = (X.sub(X.median(axis=1), axis=0)).abs().median(axis=1)
    mad = mad.reindex(acc.index) if acc is not None else mad
    acc = mad if acc is None else acc.add(mad, fill_value=0)
acc = (acc / len(files)).sort_values(ascending=False)
print('genes ranked =', len(acc))

band = acc.index[LO - 1:HI]
band = [g for g in band if g not in cur]
print('candidate band (rank %d-%d, excluding current panel) = %d genes' % (LO, HI, len(band)))

rng = np.random.default_rng(SEED)
panel2 = sorted(rng.choice(band, size=N, replace=False).tolist())
print('panel2 =', panel2)

json.dump(dict(N=N, seed=SEED, rank_band=[LO, HI], panel=panel2,
               exclude=len(cur), ranked_genes=len(acc),
               median_mad_of_panel2=[float(acc[g]) for g in panel2],
               rank_percentile_of_panel2=[float(list(acc.index).index(g) + 1) / len(acc) * 100
                                          for g in panel2]),
          open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('wrote', OUT)
