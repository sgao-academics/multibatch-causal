# -*- coding: utf-8 -*-
"""W2-J: cross-platform check of the axis.

Uses the GISTIC 2 thresholded copy-number matrix for breast carcinoma and correlates PC1
with an aneuploidy-burden score, defined as the fraction of genes with an absolute
thresholded value of at least one.  This probes the same axis on a different data type.

Output: results/_crossmodal_check.json
"""
import os, json, gzip
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

GISTIC = os.path.join(TCGA_DIR, 'Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes.gz')
OUT = os.path.join(RES, '_crossmodal_check.json')

meta = json.load(open(os.path.join(PANELS, 'panel_A_100genes.json'), encoding='utf-8'))
panel = meta['panel']
L1 = np.array(json.load(open(os.path.join(RES, '_latent_axis.json'),
                             encoding='utf-8'))['_L1']) if False else None

NONMAL = ['PTPRC', 'CD3D', 'CD3E', 'CD2', 'CD19', 'MS4A1', 'CD79A', 'CD79B',
          'COL1A1', 'COL1A2', 'COL3A1', 'DCN', 'LUM', 'VWF', 'PECAM1', 'ACTA2',
          'PDGFRB', 'FAP', 'THY1', 'CD163', 'CSF1R', 'LYZ']
TUMOR = ['EPCAM', 'KRT8', 'KRT18', 'KRT19', 'CDH1', 'MUC1']
NONMAL = [g for g in NONMAL if g not in set(panel)]
TUMOR = [g for g in TUMOR if g not in set(panel)]

# ---- GISTIC: mean |score| per sample ----
print('reading GISTIC ...')
with gzip.open(GISTIC, 'rt', encoding='utf-8', errors='ignore') as f:
    header = f.readline().rstrip('\n').split('\t')
    samples = header[1:]
    acc = np.zeros(len(samples))
    cnt = 0
    for line in f:
        p = line.rstrip('\n').split('\t')
        if len(p) != len(header):
            continue
        v = np.array(p[1:], dtype=np.float64)
        acc += np.abs(v)
        cnt += 1
print('  genes used = %d | samples = %d' % (cnt, len(samples)))
aneu = pd.Series(acc / cnt, index=samples)

# ---- BRCA expression ----
df = pd.read_csv(os.path.join(TCGA_DIR, 'TCGA_BRCA_HiSeqV2.tsv'), sep='\t', index_col=0).T
print('expression samples =', df.shape[0])

common = sorted(set(aneu.index) & set(df.index))
print('matched samples =', len(common))

# expression side: PC1 score of the panel + marker score (z-scores within this study)
Xp = df.reindex(columns=panel).values.astype(np.float64)
Xp = np.nan_to_num(Xp, nan=0.0)
Zp = (Xp - Xp.mean(0)) / (Xp.std(0) + 1e-12)
U, S, Vt = np.linalg.svd(Zp - Zp.mean(0), full_matrices=False)
pc1 = U[:, 0] * S[0] / np.sqrt(Zp.shape[0])

cols = list(dict.fromkeys(NONMAL + TUMOR))
Xn = df.reindex(columns=cols).values.astype(np.float64)
Xn = np.nan_to_num(Xn, nan=0.0)
Zn = (Xn - Xn.mean(0)) / (Xn.std(0) + 1e-12)
s_nm = Zn[:, [cols.index(g) for g in NONMAL]].mean(1)
s_tu = Zn[:, [cols.index(g) for g in TUMOR]].mean(1)

idx = {s: i for i, s in enumerate(df.index)}
sel = np.array([idx[s] for s in common])
a = aneu.loc[common].values

r_pc1 = stats.pearsonr(pc1[sel], a)
r_nm = stats.pearsonr(s_nm[sel], a)
r_tu = stats.pearsonr(s_tu[sel], a)
print()
print('BRCA n = %d' % len(common))
print('r(PC1 of variance-selected panel , aneuploidy burden) = %+.3f  p = %.3g' % r_pc1)
print('r(non-malignant marker score     , aneuploidy burden) = %+.3f  p = %.3g' % r_nm)
print('r(tumour-epithelial marker score , aneuploidy burden) = %+.3f  p = %.3g' % r_tu)
print()
print('aneuploidy burden: mean=%.3f sd=%.3f' % (a.mean(), a.std()))

json.dump(dict(genes_used=cnt, gistic_samples=len(samples),
               matched_samples=len(common),
               r_PC1_aneuploidy=float(r_pc1[0]), p_PC1_aneuploidy=float(r_pc1[1]),
               r_nonmal_aneuploidy=float(r_nm[0]), p_nonmal_aneuploidy=float(r_nm[1]),
               r_tumor_aneuploidy=float(r_tu[0]), p_tumor_aneuploidy=float(r_tu[1]),
               aneuploidy_mean=float(a.mean()), aneuploidy_sd=float(a.std()),
               note='GISTIC file contains BRCA samples only; single-cohort cross-modality check'),
          open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('\nwrote', OUT)
