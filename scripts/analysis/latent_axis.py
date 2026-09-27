# -*- coding: utf-8 -*-
"""W2-H: dominant axis of the pooled aligned matrix.

Pools the expression matrix across the 33 cohorts and diagonalises its correlation matrix,
then compares the leading eigenvalue with the Marchenko-Pastur upper edge
lambda_+ = (1 + sqrt(d/n))^2, which bounds the largest eigenvalue a null matrix of this
shape would produce.

Output: results/_latent_axis.json
"""
import os, json
import numpy as np
import pandas as pd
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
DATA = os.path.join(_ROOT, 'data')
RES = os.path.join(_ROOT, 'results')
PANELS = os.path.join(DATA, 'panels')
# The TCGA expression matrices are not redistributed with this package: read them from
# ./data/ unless MULTIBATCH_DATA names the folder that holds them, as run_all.py does.
TCGA_DIR = os.environ.get('MULTIBATCH_DATA') or DATA
FIG = os.path.join(_ROOT, 'figures')

meta = json.load(open(os.path.join(PANELS, 'panel_A_100genes.json'), encoding='utf-8'))
panel, cancers = meta['panel'], meta['cancers']
OUT = os.path.join(RES, '_latent_axis.json')
FIND = json.load(open(os.path.join(RES, '_recurrence_analysis.json'), encoding='utf-8'))
edges = FIND['aligned']['shared_edges']
gi = {g: i for i, g in enumerate(panel)}

# ---- pooling: z-scores within each cohort, stacked vertically (removes cohort mean and scale) ----
blocks, per_coh = [], {}
for c in cancers:
    df = pd.read_csv(os.path.join(TCGA_DIR, 'TCGA_%s_HiSeqV2.tsv' % c), sep='\t', index_col=0).T
    X = df.reindex(columns=panel).values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0)
    Z = (X - X.mean(0)) / (X.std(0) + 1e-12)
    blocks.append(Z)
    per_coh[c] = dict(n=int(Z.shape[0]),
                      mean_abs_corr=float(np.abs(np.corrcoef(Z, rowvar=False)[np.triu_indices(len(panel), 1)]).mean()))
P = np.vstack(blocks)
print('pooled matrix =', P.shape)

# ---- A) PCA ----
U, S, Vt = np.linalg.svd(P - P.mean(0), full_matrices=False)
ev = (S ** 2) / (S ** 2).sum()
cum = np.cumsum(ev)
print()
print('PC   var%%   cum%%')
for k in range(10):
    print('%2d  %5.2f  %6.2f' % (k + 1, 100 * ev[k], 100 * cum[k]))
n80 = int(np.searchsorted(cum, 0.80) + 1)
n90 = int(np.searchsorted(cum, 0.90) + 1)
print('PCs for 80%% = %d | for 90%% = %d' % (n80, n90))
print('mean |corr| between pooled genes = %.3f' % np.mean([per_coh[c]['mean_abs_corr'] for c in cancers]))

L = Vt[:5].T                                  # (d, 5) loadings

# ---- B) end-point loadings of the 14 recurring edges ----
print()
print('14 shared edges → PC loadings of endpoints')
rows = []
for e in edges:
    i, j = gi[e['src']], gi[e['dst']]
    rows.append(dict(src=e['src'], dst=e['dst'], n_cohorts=e['n_cohorts'],
                     src_PC1=float(L[i, 0]), dst_PC1=float(L[j, 0]),
                     src_PC2=float(L[i, 1]), dst_PC2=float(L[j, 1]),
                     both_same_sign_PC1=bool(np.sign(L[i, 0]) == np.sign(L[j, 0]))))
    print('  %-10s -> %-10s  PC1: %+.3f / %+.3f   PC2: %+.3f / %+.3f   same-sign PC1 = %s'
          % (e['src'], e['dst'], L[i, 0], L[j, 0], L[i, 1], L[j, 1], rows[-1]['both_same_sign_PC1']))
nsame = sum(r['both_same_sign_PC1'] for r in rows)
print('edges whose endpoints share PC1 sign = %d / %d' % (nsame, len(rows)))

# random control: probability that two arbitrary genes share the PC1 sign
pos = (L[:, 0] > 0).mean()
p_rand = pos ** 2 + (1 - pos) ** 2
print('random baseline P(same sign on PC1) = %.3f' % p_rand)

# ---- C) proxy for the non-malignant component ----
STROMA = ['COL10A1', 'COL11A1', 'COL9A3', 'COL22A1', 'THBS4', 'SMOC1', 'OGN',
          'PCOLCE2', 'SFRP1', 'SFRP2', 'SFRP4', 'WISP2', 'FNDC1', 'CRLF1', 'PI16']
IMMUNE = ['CXCL9', 'CXCL10', 'CXCL11', 'CXCL13', 'CCL19', 'IGJ', 'MGC29506', 'PIGR',
          'CCL18', 'CHIT1', 'CHI3L1', 'MARCO', 'S100A8', 'FCER1A', 'TPSB2']
ERYTH = ['HBB', 'HBA1']
BLOOD = ['HBB', 'HBA1', 'FABP4', 'ADH1B', 'RBP4', 'PTGDS', 'FMO2']
for nm, st in [('stroma', STROMA), ('immune', IMMUNE), ('erythrocyte', ERYTH), ('adipose', BLOOD)]:
    ids = [gi[g] for g in st if g in gi]
    print('  %-12s markers in panel = %d/%d' % (nm, len(ids), len(st)))

ids = [gi[g] for g in (STROMA + IMMUNE + ERYTH + BLOOD) if g in gi]
proxy = P[:, ids].mean(1)
r1 = stats.pearsonr(proxy, U[:, 0] * S[0] / np.sqrt(P.shape[0]))[0]
r2 = stats.pearsonr(proxy, U[:, 1] * S[1] / np.sqrt(P.shape[0]))[0]
print()
print('PRIMARY CHECK: non-malignant-content proxy vs PC1: r = %.3f' % r1)
print('               non-malignant-content proxy vs PC2: r = %.3f' % r2)

# variance explained by the proxy itself
pv = P[:, ids].mean(1)
r_proxy_all = np.corrcoef(pv, P.T)[0, 1:]
print('proxy vs each panel gene: mean r = %.3f | #|r|>0.5 = %d/%d'
      % (float(np.nanmean(np.abs(r_proxy_all))),
         int(np.sum(np.abs(r_proxy_all) > 0.5)), len(panel)))

json.dump(dict(
    pooled_shape=list(P.shape),
    pca_var_explained=[float(x) for x in ev[:15]],
    pca_cum=[float(x) for x in cum[:15]],
    pcs_for_80pct=n80, pcs_for_90pct=n90,
    mean_abs_corr_pooled=float(np.mean([per_coh[c]['mean_abs_corr'] for c in cancers])),
    edges_pc_loadings=rows,
    edges_same_sign_pc1=nsame, n_edges=len(rows),
    random_same_sign_pc1=float(p_rand),
    proxy_vs_pc1=float(r1), proxy_vs_pc2=float(r2),
    proxy_gene_count=len(ids),
    proxy_mean_abs_r=float(np.nanmean(np.abs(r_proxy_all))),
    proxy_genes_above_0p5=int(np.sum(np.abs(r_proxy_all) > 0.5)),
    marker_counts=dict(stroma=len([g for g in STROMA if g in gi]),
                       immune=len([g for g in IMMUNE if g in gi]),
                       erythrocyte=len([g for g in ERYTH if g in gi]),
                       adipose=len([g for g in BLOOD if g in gi])),
), open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print()
print('wrote', OUT)
