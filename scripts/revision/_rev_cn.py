# -*- coding: utf-8 -*-
"""N1: does copy number carry the co-expression of the recurring cis pairs?

Copy number is the one genomic layer that can produce correlated expression between two genes
with no regulatory relation between them, and it is spatially organised, so the cis pairs of
the recurring set cannot be cleared by physical distance alone.  For each cis pair this script
residualises each gene's expression rank on that gene's own thresholded copy-number state
within a cohort and recomputes the correlation on the residuals.  If the partial correlation
survives, the edge is not co-amplification.

The five cis edges of Table 4 are the four distinct pairs handled here, since CXCL1/CXCL8 is
recovered in both orientations.

Copy-number source: data/fig8_cna_genes.npz (pan-cancer GISTIC 2 calls for the recurring
genes).  Expression: data/TCGA_<cohort>_HiSeqV2.tsv.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _rev_lib as L  # noqa: E402

CIS = [('CXCL10', 'CXCL11'), ('CXCL9', 'CXCL10'), ('CXCL1', 'IL8'), ('C7', 'PLCXD3')]
ALIAS = {'IL8': 'CXCL8'}
GENE_SET = sorted({g for p in CIS for g in p} | set(ALIAS.values()))
MIN_N = 20


def pat(x):
    return str(x)[:12]


def main():
    z = np.load(os.path.join(L.DATA, 'fig8_cna_genes.npz'), allow_pickle=True)
    samples = [str(x) for x in z['samples']]
    cn = pd.DataFrame({k: z[k] for k in z.files if k != 'samples'}, index=samples)
    cnidx = set(cn.index)
    _fits, cohorts = L.load_fits('_shared_panel_notears.json')

    rows = []
    for c in cohorts:
        path = os.path.join(L.DATA, 'TCGA_%s_HiSeqV2.tsv' % c)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, sep='\t', index_col=0)
        use = [str(x) for x in df.columns if str(x) in cnidx]
        if len(use) < MIN_N:
            continue
        expr = {}
        for g in GENE_SET:
            for cand in (g, ALIAS.get(g, g)):
                if cand in df.index:
                    expr[g] = {pat(k): float(v) for k, v in df.loc[cand].items()}
                    break
        for a, b in CIS:
            if a not in expr or b not in expr:
                continue
            d = pd.DataFrame({'a': [expr[a].get(pat(x), np.nan) for x in use],
                              'b': [expr[b].get(pat(x), np.nan) for x in use],
                              'ca': cn.loc[use, a].values.astype(float),
                              'cb': cn.loc[use, b].values.astype(float)},
                             index=[pat(x) for x in use])
            d = d.groupby(level=0).mean().dropna()
            if len(d) < MIN_N:
                continue
            raw = float(stats.spearmanr(d['a'], d['b']).statistic)
            r1, r2 = stats.rankdata(d['a']), stats.rankdata(d['b'])
            c1, c2 = d['ca'].values.astype(float), d['cb'].values.astype(float)

            def resid(r, x):
                if np.std(x) == 0:
                    return r - r.mean()
                return r - r.mean() - (x - x.mean()) * (np.cov(r, x)[0, 1] / np.var(x))

            part = float(stats.pearsonr(resid(r1, c1), resid(r2, c2)).statistic)
            rows.append(dict(pair='%s-%s' % (a, b), cohort=c, n=int(len(d)),
                             rho_raw=raw, rho_part=part,
                             gain=float((d['ca'] >= 1).mean())))

    R = pd.DataFrame(rows)
    summary = {}
    for key, g in R.groupby('pair'):
        summary[key] = dict(cohorts=int(len(g)),
                            median_raw=float(g['rho_raw'].median()),
                            median_partial=float(g['rho_part'].median()),
                            cohorts_partial_ge_030=int((g['rho_part'] >= 0.30).sum()),
                            cohorts_with_gain=int((g['gain'] > 0).sum()))
    locus_gain = {g: float((cn[g] >= 1).mean()) for g in GENE_SET if g in cn.columns}
    out = dict(n_samples=int(len(samples)), n_cohorts=len(cohorts), min_n=MIN_N,
               pairs=['%s-%s' % p for p in CIS], summary=summary,
               locus_gain_frequency=locus_gain, per_cohort=rows,
               note=('Partial Spearman controls each gene expression rank for its own '
                     'thresholded copy-number state within the cohort; if it survives, the '
                     'co-expression is not co-amplification.'))
    L.save(out, 'cn_partial.json')
    R.to_csv(os.path.join(L.REV, 'cn_partial_per_cohort.csv'), index=False)
    for k, v in sorted(summary.items()):
        print('%-16s cohorts=%2d  raw=%.3f  partial=%.3f  partial>=0.30 in %2d of %d'
              % (k, v['cohorts'], v['median_raw'], v['median_partial'],
                 v['cohorts_partial_ge_030'], v['cohorts']))
    print('locus gain frequency: %s'
          % '  '.join('%s=%.3f' % kv for kv in sorted(locus_gain.items())))


if __name__ == '__main__':
    main()
