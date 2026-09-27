# -*- coding: utf-8 -*-
"""W2-E: null baseline -- does the number of recurring edges exceed chance?

The |W| matrix of every cohort is subjected to a node-label permutation, which preserves
the edge-count distribution but randomises which gene pairs carry the edges; the number of
edges present in >= 30% of cohorts is then recomputed.  Repeating the whole procedure P
times gives the null distribution.

Output: results/_permutation_null.json
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
OUT = os.path.join(RES, '_permutation_null.json')
TAU = 0.3
Q = 0.30
P = 200
SEED = 42

d = json.load(open(os.path.join(PANELS, 'panel_A_100genes.json'), encoding='utf-8'))['N']
raw = json.load(open(SH, encoding='utf-8'))
cs = sorted(k for k, v in raw.items() if isinstance(v, dict) and 'W' in v)
C = len(cs)
B = np.stack([(np.abs(np.array(raw[c]['W'], dtype=float)) > TAU).astype(np.int8) for c in cs], axis=0)
print('cohorts=%d d=%d' % (C, d))


def count_shared(Bin, q, d):
    pres = Bin.sum(axis=0)
    thr = q * Bin.shape[0]
    off = ~np.eye(d, dtype=bool)
    return int(np.sum((pres >= thr) & off))


obs = count_shared(B, Q, d)
print('observed shared edges (>=%.0f%% of %d cohorts) = %d' % (Q * 100, C, obs))

rng = np.random.default_rng(SEED)
null = np.empty(P, dtype=int)
for t in range(P):
    Bp = np.empty_like(B)
    for k in range(C):
        perm = rng.permutation(d)
        Bp[k] = B[k][np.ix_(perm, perm)]
    null[t] = count_shared(Bp, Q, d)

mu, sd = float(null.mean()), float(null.std())
z = (obs - mu) / sd if sd > 0 else float('inf')
pv = float((np.sum(null >= obs) + 1) / (P + 1))
print('null: mean=%.2f sd=%.2f max=%d' % (mu, sd, int(null.max())))
print('z = %.1f | perm p = %.4f' % (z, pv))

json.dump(dict(observed=obs, n_cohorts=C, d=d, thresh_frac=Q, tau=TAU,
               permutations=P, null_mean=mu, null_sd=sd, null_max=int(null.max()),
               z=z, p_perm=pv,
               null_hist=np.bincount(null, minlength=obs + 2).tolist()[:obs + 3]),
          open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('wrote', OUT)
