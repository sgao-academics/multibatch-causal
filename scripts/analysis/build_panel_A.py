# -*- coding: utf-8 -*-
"""W2-A: build the cross-cohort shared gene panel (step 1 of H1).

When each cohort takes its own top-100 genes by MAD the cohorts share only about 4% of
their genes (Jaccard 0.0396), so the same matrix position indexes different genes in
different cohorts and the element-wise median of the stacked matrices collapses to zero.
The fix is one panel shared by every cohort, which makes positions comparable.  The panel
is built from the genes that are highly variable in all 33 cohorts: for each gene take its
within-cohort MAD percentile, average that percentile over the 33 cohorts, and keep the top N.

Output: data/panels/panel_A_100genes.json
"""
import os, json
import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
DATA = os.path.join(_ROOT, 'data')
RES = os.path.join(_ROOT, 'results')
PANELS = os.path.join(DATA, 'panels')
FIG = os.path.join(_ROOT, 'figures')

OUT = os.path.join(PANELS, 'panel_A_100genes.json')
N = 100

files = sorted(f for f in os.listdir(DATA)
               if f.startswith('TCGA_') and f.endswith('_HiSeqV2.tsv'))
cancers = [f.replace('TCGA_', '').replace('_HiSeqV2.tsv', '') for f in files]
print('cancers found =', len(cancers), flush=True)

per, gene_lists, gene_sets = {}, {}, {}
for i, (f, c) in enumerate(zip(files, cancers)):
    df = pd.read_csv(os.path.join(DATA, f), sep='\t', index_col=0).T
    v = df.values.astype(np.float64)
    mad = pd.Series(median_abs_deviation(v, axis=0), index=df.columns)
    mad = mad.replace([np.inf, -np.inf], np.nan).dropna()
    per[c] = mad
    gene_lists[c] = [str(g) for g in mad.sort_values(ascending=False).index[:N]]
    gene_sets[c] = set(gene_lists[c])
    if (i + 1) % 5 == 0 or i == len(cancers) - 1:
        print('  [%2d/%d] %-5s samples=%4d genes=%d' % (i + 1, len(cancers), c, df.shape[0], df.shape[1]),
              flush=True)

union = sorted(set().union(*gene_sets.values()))
inter = sorted(set.intersection(*gene_sets.values()))
print('\nunion of 33 top-%d sets = %d genes' % (N, len(union)))
print('intersection of all 33    = %d genes' % len(inter))

common = sorted(set.intersection(*(set(per[c].index) for c in cancers)))
print('genes present in all 33 cohorts =', len(common))

rank = pd.DataFrame(index=common)
for c in cancers:
    rank[c] = per[c].rank(pct=True).reindex(common)
mean_pct = rank.mean(axis=1).sort_values(ascending=False)
panel = [str(g) for g in mean_pct.index[:N]]

cov = {c: len(set(panel) & gene_sets[c]) for c in cancers}
print('\npanel ∩ each cohort top-%d : min=%d mean=%.1f max=%d  (of %d)'
      % (N, min(cov.values()), float(np.mean(list(cov.values()))), max(cov.values()), N))

json.dump(dict(N=N, cancers=cancers, panel=panel,
               union_size=len(union), inter_size=len(inter), n_common_genes=len(common),
               per_cohort_top=gene_lists,
               panel_overlap_with_cohort_top=cov,
               mean_pct_top60={g: round(float(mean_pct[g]), 4) for g in panel[:60]}),
          open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('wrote', OUT)
print('\npanel[:20] =', panel[:20])
