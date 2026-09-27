# -*- coding: utf-8 -*-
"""W2-I: attribution of the dominant axis.

Correlates PC1 with stromal, immune and endothelial marker genes drawn deliberately from
outside the 100-gene panel -- so that the panel cannot validate itself -- and records the
sign and magnitude cohort by cohort.

Output: results/_axis_attribution.json
"""
import os, json, glob
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

OUT = os.path.join(RES, '_axis_attribution.json')
meta = json.load(open(os.path.join(PANELS, 'panel_A_100genes.json'), encoding='utf-8'))
panel, cancers = meta['panel'], meta['cancers']
THR = [100, 200, 500, 1000, 2000]

NONMAL = ['PTPRC', 'CD3D', 'CD3E', 'CD2', 'CD19', 'MS4A1', 'CD79A', 'CD79B',
          'COL1A1', 'COL1A2', 'COL3A1', 'DCN', 'LUM', 'VWF', 'PECAM1', 'ACTA2',
          'PDGFRB', 'FAP', 'THY1', 'CD163', 'CSF1R', 'LYZ']
TUMOR = ['EPCAM', 'KRT8', 'KRT18', 'KRT19', 'CDH1', 'MUC1']
DRIVERS = """TP53 KRAS MYC PIK3CA PTEN EGFR BRAF NRAS APC CTNNB1 RB1 CDKN2A SMAD4
ARID1A BRCA1 BRCA2 ERBB2 MET ALK IDH1 IDH2 VHL NF1 AKT1 CCND1 MDM2 GATA3 FOXA1
ESR1 AR TP63 SOX2 NFE2L2 KEAP1 STK11 ATM CHEK2 FAT1 NOTCH1 NOTCH2 CASP8 HRAS
PTPN11 MLH1 MSH2 MSH6 PMS2 BAP1 SETD2 B2M JAK2 STAT3 MTOR TSC1 TSC2 EZH2
KMT2D CREBBP EP300 FGFR1 FGFR2 FGFR3 KIT PDGFRA RET SRC ABL1""".split()

in_panel = set(panel)
NONMAL = [g for g in NONMAL if g not in in_panel]
TUMOR = [g for g in TUMOR if g not in in_panel]
print('external non-malignant markers =', len(NONMAL), '| external tumour-epithelial =', len(TUMOR))
print('  (panel genes excluded to avoid circularity)')

# ---------- A) Marchenko-Pastur reference from the pooled panel matrix ----------
pooled = []
per_coh = {}
for c in cancers:
    df = pd.read_csv(os.path.join(TCGA_DIR, 'TCGA_%s_HiSeqV2.tsv' % c), sep='\t', index_col=0).T
    panelX = df.reindex(columns=panel).values.astype(np.float64)
    panelX = np.nan_to_num(panelX, nan=0.0)
    Zp = (panelX - panelX.mean(0)) / (panelX.std(0) + 1e-12)
    pooled.append(Zp)
P = np.vstack(pooled)
n, p = P.shape
ev = np.linalg.svd(P - P.mean(0), compute_uv=False)
lam = (ev ** 2) / n                      # eigenvalues (variables standardised, so their sum is approximately p)
gamma = p / n
mp_plus = (1 + np.sqrt(gamma)) ** 2
n_sig = int(np.sum(lam > mp_plus))
print()
print('=========== A) Marchenko–Pastur ===========')
print('n=%d p=%d  gamma=%.5f  MP upper edge = %.3f' % (n, p, gamma, mp_plus))
print('top eigenvalues: ' + ', '.join('%.2f' % x for x in lam[:12]))
print('# eigenvalues above MP edge = %d ; lam1 = %.2f = %.1fx the edge'
      % (n_sig, lam[0], lam[0] / mp_plus))

# loadings (used for the per-cohort PC1 score)
_, _, Vt = np.linalg.svd(P - P.mean(0), full_matrices=False)
L1 = Vt[0]

# ---------- B) within each cohort: PC1 score against external markers ----------
rows = []
for c in cancers:
    df = pd.read_csv(os.path.join(TCGA_DIR, 'TCGA_%s_HiSeqV2.tsv' % c), sep='\t', index_col=0).T
    ids = [i for i, j in [(i, True) for i in range(1)]]
    cols = list(dict.fromkeys(NONMAL + TUMOR))
    nm = df.reindex(columns=cols).values.astype(np.float64)
    nm = np.nan_to_num(nm, nan=0.0)
    nm = (nm - nm.mean(0)) / (nm.std(0) + 1e-12)
    kn = [cols.index(g) for g in NONMAL]
    kt = [cols.index(g) for g in TUMOR]
    score_nm = nm[:, kn].mean(1)
    score_tu = nm[:, kt].mean(1)
    pm = df.reindex(columns=panel).values.astype(np.float64)
    pm = np.nan_to_num(pm, nan=0.0)
    pm = (pm - pm.mean(0)) / (pm.std(0) + 1e-12)
    pc1 = pm @ L1
    r_nm = stats.pearsonr(pc1, score_nm)[0]
    r_tu = stats.pearsonr(pc1, score_tu)[0]
    rows.append(dict(cohort=c, n=int(pm.shape[0]), r_PC1_nonmal=float(r_nm),
                     r_PC1_tumor=float(r_tu)))
    print('  %-6s n=%4d  r(PC1, non-malignant)= %+.3f   r(PC1, tumour-epithelial)= %+.3f'
          % (c, pm.shape[0], r_nm, r_tu))

r_nm_all = np.array([x['r_PC1_nonmal'] for x in rows])
r_tu_all = np.array([x['r_PC1_tumor'] for x in rows])
print()
print('MEAN over %d cohorts: r(PC1, non-malignant) = %+.3f  (range %+.3f .. %+.3f)'
      % (len(rows), r_nm_all.mean(), r_nm_all.min(), r_nm_all.max()))
print('MEAN over %d cohorts: r(PC1, tumour-epi)    = %+.3f  (range %+.3f .. %+.3f)'
      % (len(rows), r_tu_all.mean(), r_tu_all.min(), r_tu_all.max()))
print('cohorts with r(PC1, non-malignant) > 0 : %d / %d' % (int((r_nm_all > 0).sum()), len(rows)))
print('cohorts with r(PC1, tumour-epi)    < 0 : %d / %d' % (int((r_tu_all < 0).sum()), len(rows)))

# ---------- C) driver-gene threshold sensitivity curve ----------
print()
print('=========== C) driver genes vs MAD threshold ===========')
cnt = {t: 0 for t in THR}
coh_with = {t: 0 for t in THR}
medpct = []
for c in cancers:
    df = pd.read_csv(os.path.join(TCGA_DIR, 'TCGA_%s_HiSeqV2.tsv' % c), sep='\t', index_col=0)
    X = df.apply(pd.to_numeric, errors='coerce').dropna(how='all')
    mad = (X.sub(X.median(axis=1), axis=0)).abs().median(axis=1).sort_values(ascending=False)
    tot = len(mad)
    present = [g for g in set(DRIVERS) if g in mad.index]
    pct = [float(np.where(mad.index == g)[0][0] + 1) / tot * 100 for g in present]
    medpct.append(float(np.median(pct)))
    for t in THR:
        top = set(mad.index[:t])
        k = len(top & set(DRIVERS))
        cnt[t] += k
        coh_with[t] += (1 if k > 0 else 0)
print('%-8s %12s %14s' % ('top_k', 'driver_hits', 'cohorts_any'))
for t in THR:
    print('%-8d %12d %14d' % (t, cnt[t], coh_with[t]))
print('median driver MAD percentile across cohorts = %.1f%%' % float(np.median(medpct)))
print('selection percentile needed for top-100 = %.3f%%' % (100.0 * 100 / 20530))

json.dump(dict(
    mp=dict(n=int(n), p=int(p), gamma=float(gamma), mp_upper=float(mp_plus),
            top_eigenvalues=[float(x) for x in lam[:20]],
            n_significant=n_sig, lam1_over_edge=float(lam[0] / mp_plus)),
    purity_attribution=dict(external_nonmal_markers=NONMAL,
                            external_tumor_markers=TUMOR,
                            per_cohort=rows,
                            mean_r_nonmal=float(r_nm_all.mean()),
                            min_r_nonmal=float(r_nm_all.min()),
                            max_r_nonmal=float(r_nm_all.max()),
                            mean_r_tumor=float(r_tu_all.mean()),
                            min_r_tumor=float(r_tu_all.min()),
                            max_r_tumor=float(r_tu_all.max()),
                            n_cohorts_positive_nonmal=int((r_nm_all > 0).sum()),
                            n_cohorts_negative_tumor=int((r_tu_all < 0).sum()),
                            n_cohorts=len(rows)),
    driver_threshold_curve=dict(thresholds=THR, driver_hits=cnt, cohorts_any=coh_with,
                                median_driver_percentile=float(np.median(medpct)),
                                top100_percentile_cut=float(100.0 * 100 / 20530)),
), open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('\nwrote', OUT)
