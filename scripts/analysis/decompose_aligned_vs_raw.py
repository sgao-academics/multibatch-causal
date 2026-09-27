# -*- coding: utf-8 -*-
"""W2-C: A/B test of the two-stage decomposition on real data (last step of H1).

(A) aligned panel (genes shared by all cohorts): does the element-wise median retain
    non-zero shared structure?
(B) misaligned design (original per-cohort top-100, where positions index different genes):
    does the median collapse to the zero matrix?

Output: results/_shared_decomposition.json
"""
import os, json
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
DATA = os.path.join(_ROOT, 'data')
RES = os.path.join(_ROOT, 'results')
PANELS = os.path.join(DATA, 'panels')
FIG = os.path.join(_ROOT, 'figures')

SH = os.path.join(RES, '_shared_panel_notears.json')
OLD = os.path.join(RES, '_pipeline_notears.json')
OUT = os.path.join(RES, '_shared_decomposition.json')
TAU = 0.3


def load(path):
    d = json.load(open(path, encoding='utf-8'))
    cs = sorted(k for k, v in d.items() if isinstance(v, dict) and 'W' in v)
    return d, cs


def median_matrix(d, cs):
    M = np.stack([np.abs(np.array(d[c]['W'])) for c in cs], axis=0)
    med = np.median(M, axis=0)
    np.fill_diagonal(med, 0.0)
    return M, med


print('=' * 20, 'A) ALIGNED panel (shared panel)')
sh, cs = load(SH)
M, med = median_matrix(sh, cs)
d = med.shape[0]
n_al = int(np.sum(med > TAU))
print('cohorts = %d | d = %d' % (len(cs), d))
print('median-matrix edges (|med| > %.1f) = %d' % (TAU, n_al))
print('max |median| = %.4f' % float(med.max()))

present = (M > TAU).sum(axis=0)
shared = [(i, j, int(present[i, j])) for i in range(d) for j in range(d)
          if i != j and present[i, j] >= len(cs) * 0.3]
print('edges present in >=30%% of cohorts = %d' % len(shared))

print()
print('=' * 20, 'B) MISALIGNED (original per-cohort panels)')
old, cs2 = load(OLD)
Mo, med_o = median_matrix(old, cs2)
n_old = int(np.sum(med_o > TAU))
print('cohorts = %d | d = %d' % (len(cs2), Mo.shape[1]))
print('median-matrix edges (|med| > %.1f) = %d' % (TAU, n_old))
print('max |median| = %.4f' % float(med_o.max()))

json.dump(dict(
    panel_aligned=dict(n_cohorts=len(cs), d=d, median_edges=n_al,
                       max_abs_median=float(med.max()),
                       shared_ge30pct=len(shared),
                       shared_list=[dict(i=i, j=j, count=k) for i, j, k in shared[:80]]),
    misaligned=dict(n_cohorts=len(cs2), d=Mo.shape[1], median_edges=n_old,
                    max_abs_median=float(med_o.max())),
), open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('\nwrote', OUT)
