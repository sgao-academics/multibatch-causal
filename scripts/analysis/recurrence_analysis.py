# -*- coding: utf-8 -*-
"""W2-D: full analysis on the shared panel (all 33 cohorts fitted).

1) A/B decomposition: aligned against misaligned
2) Recurring edges (present in >= 30% of cohorts), annotated with gene names
3) Edge count against n on the shared panel: the gene set is fixed, so the effect comes
   purely from sample size
4) Cross-reference against the original per-cohort top-100 graph

Output: results/_recurrence_analysis.json
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

SH = os.path.join(RES, '_shared_panel_notears.json')
OLD = os.path.join(RES, '_pipeline_notears.json')
PNL = os.path.join(PANELS, 'panel_A_100genes.json')
OUT = os.path.join(RES, '_recurrence_analysis.json')
TAU = 0.3

panel = json.load(open(PNL, encoding='utf-8'))
genes = panel['panel']
N = panel['N']


def load(path):
    d = json.load(open(path, encoding='utf-8'))
    cs = sorted(k for k, v in d.items() if isinstance(v, dict) and 'W' in v)
    return d, cs


def median_matrix(d, cs, dref=None):
    mats = []
    for c in cs:
        W = np.abs(np.array(d[c]['W'], dtype=float))
        if dref is not None and W.shape[0] != dref:
            continue
        mats.append(W)
    M = np.stack(mats, axis=0)
    med = np.median(M, axis=0)
    np.fill_diagonal(med, 0.0)
    return M, med


report = {}

# ---------- A) ALIGNED ----------
sh, cs = load(SH)
M, med = median_matrix(sh, cs, dref=N)
d = med.shape[0]
n_al = int(np.sum(med > TAU))
print('=' * 22, 'A) ALIGNED (shared panel)')
print('cohorts = %d | d = %d' % (M.shape[0], d))
print('median-matrix edges (|med| > %.1f) = %d' % (TAU, n_al))
print('max |median| = %.4f' % float(med.max()))
report['aligned'] = dict(n_cohorts=int(M.shape[0]), d=int(d),
                         median_edges=n_al, max_abs_median=float(med.max()))

# recurring edges: present in >=30% of cohorts, with consistent direction (|W| > TAU)
present = (M > TAU).sum(axis=0)
frac = present / float(M.shape[0])
shared = [(i, j, int(present[i, j]), float(frac[i, j]))
          for i in range(d) for j in range(d)
          if i != j and frac[i, j] >= 0.30]
print('edges in >=30%% of cohorts = %d' % len(shared))
report['aligned']['shared_ge30pct'] = len(shared)
report['aligned']['shared_edges'] = [
    dict(src=genes[i], dst=genes[j], n_cohorts=k, frac=round(f, 3))
    for i, j, k, f in sorted(shared, key=lambda x: -x[2])]
print()
print('shared edge list (top 40):')
for i, j, k, f in sorted(shared, key=lambda x: -x[2])[:40]:
    print('   %-10s -> %-10s  %2d/%d (%.0f%%)' % (genes[i], genes[j], k, M.shape[0], f * 100))
print()
print('shared DEGREE rank (top 15):')
deg = np.zeros(d)
for i, j, k, f in shared:
    deg[i] += 1
    deg[j] += 1
rank = sorted(range(d), key=lambda x: -deg[x])[:15]
for r in rank:
    print('   %-10s deg=%d' % (genes[r], int(deg[r])))
report['aligned']['shared_degree'] = [
    dict(gene=genes[r], degree=int(deg[r])) for r in rank]

# ---------- B) MISALIGNED ----------
print()
print('=' * 22, 'B) MISALIGNED (original per-cohort top-100)')
old, cs2 = load(OLD)
old_cs = [c for c in cs2 if np.array(old[c]['W']).shape[0] == 100]
M_old, med_o = median_matrix(old, old_cs, dref=100)
n_old = int(np.sum(med_o > TAU))
print('cohorts = %d | d = 100' % M_old.shape[0])
print('median-matrix edges (|med| > %.1f) = %d' % (TAU, n_old))
print('max |median| = %.4f' % float(med_o.max()))
report['misaligned'] = dict(n_cohorts=int(M_old.shape[0]), d=100,
                            median_edges=n_old, max_abs_median=float(med_o.max()))

# ---------- C) n against edges on the shared panel ----------
print()
print('=' * 22, 'C) identifiability ON THE SAME PANEL')
ns, es = [], []
per = {}
for c in cs:
    n = sh[c].get('n', None)
    W = np.abs(np.array(sh[c]['W'], dtype=float))
    e = int(np.sum(W > TAU))
    if n is None:
        continue
    ns.append(n)
    es.append(e)
    per[c] = dict(n=int(n), edges=e)
ns = np.array(ns, dtype=float)
es = np.array(es, dtype=float)
r, p = stats.pearsonr(ns, es)
print('ALL %d cohorts: r(n, edges) = %.3f  p = %.3g' % (len(ns), r, p))
report['panel_identifiability_all'] = dict(n=int(len(ns)), r=float(r), p=float(p))
mask = ns >= 200
if mask.sum() >= 3:
    r2, p2 = stats.pearsonr(ns[mask], es[mask])
    print('n >= 200 (%d cohorts): r = %.3f  p = %.3g' % (int(mask.sum()), r2, p2))
    report['panel_identifiability_n200'] = dict(n=int(mask.sum()), r=float(r2), p=float(p2))
low = ns < 200
print('low-n (<200) cohorts hold %.1f%% of edges (n=%d of %d)' % (
    100.0 * es[low].sum() / es.sum(), int(low.sum()), len(ns)))
report['panel_identifiability_split'] = dict(
    low_n_cohorts=int(low.sum()), low_n_edge_share=float(es[low].sum() / es.sum()),
    hi_n_edge_share=float(es[~low].sum() / es.sum()))
report['per_cohort_panel'] = per

# ---------- D) cross-reference against the original per-cohort graph ----------
print()
print('=' * 22, 'D) consistency vs original pipeline')
detail = os.path.join(RES, '_per_cancer_network.json')
if os.path.exists(detail):
    dd = json.load(open(detail, encoding='utf-8'))
    pairs = []
    for c in cs:
        if c in dd and isinstance(dd[c], dict) and 'n_edges' in dd[c]:
            pairs.append((c, int(dd[c]['n_edges']), per[c]['edges']))
    if pairs:
        a = np.array([x[1] for x in pairs], dtype=float)
        b = np.array([x[2] for x in pairs], dtype=float)
        rr, pp = stats.pearsonr(a, b)
        print('rows=%d | r(orig_edges, panel_edges) = %.3f  p = %.3g' % (len(pairs), rr, pp))
        report['consistency'] = dict(n=len(pairs), r=float(rr), p=float(pp),
                                     pairs=[dict(c=c, orig=int(x), panel=int(y))
                                            for c, x, y in pairs])
        print('  top divergent:')
        for c, x, y in sorted(pairs, key=lambda t: -abs(t[1] - t[2]))[:8]:
            print('    %-6s orig=%4d panel=%4d' % (c, x, y))

json.dump(report, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print()
print('wrote', OUT)
