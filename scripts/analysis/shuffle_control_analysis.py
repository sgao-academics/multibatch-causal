# -*- coding: utf-8 -*-
"""W2-G2: analysis of the shuffled-label control.

Compares the real and shuffled runs on edge count, on the recurring set and on the
sample-size dependence, and reports the empirical boundary: the largest cohort that still
returns edges on shuffled data and the smallest cohort that returns none.

Output: results/_shuffle_control.json
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
OUT = os.path.join(RES, '_shuffle_control.json')
TAU, Q = 0.3, 0.30

panel = json.load(open(os.path.join(PANELS, 'panel_A_100genes.json'), encoding='utf-8'))['panel']
gi = {g: i for i, g in enumerate(panel)}


def load(p):
    d = json.load(open(p, encoding='utf-8'))
    cs = sorted(k for k, v in d.items() if isinstance(v, dict) and 'W' in v)
    return d, cs


neg, ncs = load(os.path.join(RES, '_shared_panel_negctl.json'))
rea, rcs = load(os.path.join(RES, '_shared_panel_notears.json'))
print('negative control cohorts = %d | real cohorts = %d' % (len(ncs), len(rcs)))


def shared_count(store, cs):
    M = np.stack([np.abs(np.array(store[c]['W'], dtype=float)) for c in cs], 0)
    med = np.median(M, 0)
    np.fill_diagonal(med, 0)
    pres = (M > TAU).sum(0)
    frac = pres / float(M.shape[0])
    sh = [(i, j, int(pres[i, j])) for i in range(len(panel)) for j in range(len(panel))
          if i != j and frac[i, j] >= Q]
    return M, med, sh, pres


Mn, medn, shn, presn = shared_count(neg, ncs)
Mr, medr, shr, presr = shared_count(rea, rcs)
print()
print('%-22s %10s %10s' % ('', 'real', 'shuffled'))
print('%-22s %10d %10d' % ('total edges', int(sum(neg[c]['edges'] for c in ncs)),
                           int(sum(rea[c]['edges'] for c in rcs))))
print('%-22s %10d %10d' % ('median-matrix edges', int(np.sum(medr > TAU)), int(np.sum(medn > TAU))))
print('%-22s %10.4f %10.4f' % ('max |median|', medr.max(), medn.max()))
print('%-22s %10d %10d' % ('shared edges (>=30%)', len(shr), len(shn)))
if shn:
    print('  !! shuffled shared edges:', [(panel[i], panel[j], k) for i, j, k in shn])

# n vs edges
def ncorr(store, cs):
    ns = np.array([store[c]['n'] for c in cs], float)
    es = np.array([np.sum(np.abs(np.array(store[c]['W'], dtype=float)) > TAU) for c in cs], float)
    r_all, p_all = stats.pearsonr(ns, es)
    m = ns >= 200
    r_hi, p_hi = stats.pearsonr(ns[m], es[m]) if m.sum() > 2 else (np.nan, np.nan)
    # n < 100 vs n >= 170
    return ns, es, r_all, p_all, r_hi, p_hi, m

nn, ne, rn_all, pn_all, rn_hi, pn_hi, mn = ncorr(neg, ncs)
rn, re, rr_all, pr_all, rr_hi, pr_hi, mr = ncorr(rea, rcs)
print()
print('%-34s %12s %12s' % ('', 'real', 'shuffled'))
print('%-34s %12.3f %12.3f' % ('r(n, edges), all cohorts', rr_all, rn_all))
print('%-34s %12.2e %12.2e' % ('p', pr_all, pn_all))
print('%-34s %12.3f %12.3f' % ('r(n, edges), n>=200', rr_hi, rn_hi))
print('%-34s %12.2f %12.2f' % ('p, n>=200', pr_hi, pn_hi))

# empirical boundary: largest n that still returns a non-zero edge on shuffled data
nz = nn[ne > 0]
zr = nn[ne == 0]
print()
print('shuffled: cohorts with >0 edges = %d | max n among them = %d'
      % (len(nz), nz.max() if len(nz) else -1))
print('shuffled: cohorts with 0 edges  = %d | min n among them = %d'
      % (len(zr), zr.min() if len(zr) else -1))
print('=> empirical boundary  n* ~= %d  (d = 100, so n*/d ~= %.2f)'
      % (int(zr.min()) if len(zr) else -1, (zr.min() / 100.0) if len(zr) else -1))

# paired control
pairs = [(c,) for c in rcs if c in neg]
print()
print('%-6s %5s %8s %10s' % ('cohort', 'n', 'real', 'shuffled'))
for c in sorted(rcs, key=lambda x: rea[x]['n']):
    print('%-6s %5d %8d %10d' % (c, rea[c]['n'], rea[c]['edges'], neg[c]['edges']))

json.dump(dict(
    n_cohorts=len(ncs),
    real=dict(total_edges=int(sum(rea[c]['edges'] for c in rcs)),
              median_edges=int(np.sum(medr > TAU)), max_abs_median=float(medr.max()),
              shared_ge30=len(shr),
              r_all=float(rr_all), p_all=float(pr_all), r_n200=float(rr_hi), p_n200=float(pr_hi)),
    shuffled=dict(total_edges=int(sum(neg[c]['edges'] for c in ncs)),
                  median_edges=int(np.sum(medn > TAU)), max_abs_median=float(medn.max()),
                  shared_ge30=len(shn),
                  shared_edges=[dict(src=panel[i], dst=panel[j], n=n) for i, j, n in shn],
                  r_all=float(rn_all), p_all=float(pn_all), r_n200=float(rn_hi), p_n200=float(pn_hi),
                  cohorts_with_edges=int(len(nz)),
                  max_n_with_edges=int(nz.max()) if len(nz) else None,
                  min_n_without_edges=int(zr.min()) if len(zr) else None,
                  empirical_boundary_n=int(zr.min()) if len(zr) else None,
                  boundary_over_d=float(zr.min() / 100.0) if len(zr) else None),
    per_cohort=[dict(cohort=c, n=int(rea[c]['n']), real=int(rea[c]['edges']),
                     shuffled=int(neg[c]['edges'])) for c in rcs],
), open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('\nwrote', OUT)
