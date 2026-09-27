# -*- coding: utf-8 -*-
"""W2-M3: robustness analysis on the second panel.

Whether the recurring edges still fall on the non-malignant axes, whether the overlap with
panel A is at the level of genes or of axes, and whether the n against edges relation
reproduces.

Output: results/_panel_B_robustness.json
"""
import os, json
import numpy as np
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
DATA = os.path.join(_ROOT, 'data')
RES = os.path.join(_ROOT, 'results')
PANELS = os.path.join(DATA, 'panels')
FIG = os.path.join(_ROOT, 'figures')

E = RES  # scratch alias
OUT = os.path.join(RES, '_panel_B_robustness.json')
TAU, Q = 0.3, 0.30

p1 = json.load(open(os.path.join(PANELS, 'panel_A_100genes.json'), encoding='utf-8'))['panel']
p2m = json.load(open(os.path.join(PANELS, 'panel_B_100genes.json'), encoding='utf-8'))
p2 = p2m['panel']
print('panel1 = %d genes | panel2 = %d genes | overlap = %d'
      % (len(p1), len(p2), len(set(p1) & set(p2))))

# axis definitions (used by panel A)
AXIS = {
    'IFN-g/CXCR3 chemokine': ['CXCL9', 'CXCL10', 'CXCL11'],
    'neutrophil/CXCR2 chemokine': ['CXCL1', 'IL8'],
    'erythrocyte content': ['HBA1', 'HBB'],
    'plasma-cell/Ig locus': ['ADAM6', 'IGJ', 'MGC29506'],
    'stromal/complement': ['COL10A1', 'COL11A1', 'C7', 'PLCXD3'],
    'myeloid/cytokine': ['CCL18', 'CHIT1', 'IL6', 'FOSB'],
}
# non-malignant / microenvironment genes in panel B (curated, for classification)
MICROENV = set("""MMP13 MMP1 INHBA LAMA3 COL6A6 COL22A1 COL11A1 HAPLN1 SMOC2 SMOC1 TNN VWA2
WIF1 SFRP1 SFRP2 SFRP4 THBS4 OGN PCOLCE2 FNDC1 PI16 CRLF1 C15orf48
SEMA3A SEMA3C SEMA6D PLAC9 ELFN2
LYVE1 LY6D LAG3 NKG7 LTB SLAMF6 SPIB CCR2 CCL17 CCL4L2 CLEC4E CD1A C4A C4B
SERPINA1 LYZ PTPRC CD3D CD19 MS4A1
MAMDC2 LY6D DNER FBXO2 SCEL TRIM29 KRT13 KRT17
C7 C8orf80 SERPINA3
FABP4 ADH1B RBP4 PTGDS FMO2
XIST
TDO2 PAPPA""".split())
EPI_TUMOR = set("MUC1 EPCAM KRT8 KRT18 KRT19 CDH1 WT1 SFN TGM3 TRIM29 TG ALPL GDA GPT".split())


def load(p):
    d = json.load(open(p, encoding='utf-8'))
    cs = sorted(k for k, v in d.items() if isinstance(v, dict) and 'W' in v)
    return d, cs


def analyse(store, cs, genes, tag):
    M = np.stack([np.abs(np.array(store[c]['W'], dtype=float)) for c in cs], 0)
    med = np.median(M, 0)
    np.fill_diagonal(med, 0)
    pres = (M > TAU).sum(0)
    frac = pres / float(M.shape[0])
    sh = [(i, j, int(pres[i, j])) for i in range(len(genes)) for j in range(len(genes))
          if i != j and frac[i, j] >= Q]
    ns = np.array([store[c]['n'] for c in cs], float)
    es = np.array([np.sum(np.abs(np.array(store[c]['W'], dtype=float)) > TAU) for c in cs], float)
    r_all, p_all = stats.pearsonr(ns, es)
    m = ns >= 200
    r_hi, p_hi = stats.pearsonr(ns[m], es[m]) if m.sum() > 2 else (np.nan, np.nan)
    print()
    print('=' * 24, tag)
    print('cohorts=%d | total edges=%d | median-matrix edges=%d | max|median|=%.4f'
          % (len(cs), int(es.sum()), int(np.sum(med > TAU)), med.max()))
    print('shared edges (>=%d%% of %d) = %d' % (Q * 100, len(cs), len(sh)))
    print('r(n, edges) all = %.3f (p=%.2e) | n>=200 (%d): r=%.3f (p=%.2f)'
          % (r_all, p_all, int(m.sum()), r_hi, p_hi))
    lo = ns < 200
    print('low-n cohorts (n<200, %d) hold %.1f%% of edges' % (int(lo.sum()), 100 * es[lo].sum() / es.sum()))
    print()
    print('shared edge list:')
    for i, j, k in sorted(sh, key=lambda x: -x[2]):
        st = 'MICROENV' if (genes[i] in MICROENV and genes[j] in MICROENV) else (
            'EPI/TUMOUR' if (genes[i] in EPI_TUMOR or genes[j] in EPI_TUMOR) else 'other')
        print('   %-10s -> %-10s  %2d/%d   [%s]' % (genes[i], genes[j], k, len(cs), st))
    deg = {}
    for i, j, k in sh:
        deg[genes[i]] = deg.get(genes[i], 0) + 1
        deg[genes[j]] = deg.get(genes[j], 0) + 1
    print('top hub genes:', sorted(deg.items(), key=lambda t: -t[1])[:10])
    return dict(n_cohorts=len(cs), total_edges=int(es.sum()),
                median_edges=int(np.sum(med > TAU)), max_abs_median=float(med.max()),
                shared_ge30=len(sh),
                shared_edges=[dict(src=genes[i], dst=genes[j], n=k) for i, j, k in
                              sorted(sh, key=lambda x: -x[2])],
                r_all=float(r_all), p_all=float(p_all), r_n200=float(r_hi), p_n200=float(p_hi),
                low_n_edge_share=float(es[lo].sum() / es.sum()))


s1 = json.load(open(os.path.join(RES, '_shared_panel_notears.json'), encoding='utf-8'))
c1 = sorted(k for k, v in s1.items() if isinstance(v, dict) and 'W' in v)
r1 = analyse(s1, c1, p1, 'PANEL 1 (this work, MAD top-100)')

s2 = json.load(open(os.path.join(RES, '_panel2_notears.json'), encoding='utf-8'))
c2 = sorted(k for k, v in s2.items() if isinstance(v, dict) and 'W' in v)
r2 = analyse(s2, c2, p2, 'PANEL 2 (robustness, MAD rank 201-2000)')

e1 = set((e['src'], e['dst']) for e in r1['shared_edges'])
e2 = set((e['src'], e['dst']) for e in r2['shared_edges'])
g1 = set(g for e in e1 for g in e)
g2 = set(g for e in e2 for g in e)
print()
print('edge-set overlap between panels = %d' % len(e1 & e2))
print('gene-set overlap of edge endpoints = %d  (%s)' % (len(g1 & g2), sorted(g1 & g2)))
print('panel gene overlap = %d' % len(set(p1) & set(p2)))

r2['edge_set_overlap'] = len(e1 & e2)
r2['gene_set_overlap_endpoints'] = sorted(g1 & g2)
r1_out = r1
json.dump(dict(panel1=r1_out, panel2=r2), open(OUT, 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
print()
print('wrote', OUT)
