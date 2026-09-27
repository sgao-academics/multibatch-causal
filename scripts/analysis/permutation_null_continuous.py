# -*- coding: utf-8 -*-
"""W2-E2: permutation test with a statistic that is non-degenerate under permutation.

The total number of edges summed over all gene pairs is invariant under node permutation
(2239 here), so it cannot serve as a test statistic.  We use instead:
  (1) the collision statistic Q = sum_{i!=j} presence[i,j]^2, which grows with the
      concentration of recurrence onto few pairs;
  (2) the threshold counts N_k = #{ (i,j): presence[i,j] >= k } for k = 2, 3, 5, 10, 15.

Output: results/_permutation_null.json (updates the 'continuous' section)
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
P = 500
SEED = 7
KS = [2, 3, 5, 10, 15]

d = json.load(open(os.path.join(PANELS, 'panel_A_100genes.json'), encoding='utf-8'))['N']
raw = json.load(open(SH, encoding='utf-8'))
cs = sorted(k for k, v in raw.items() if isinstance(v, dict) and 'W' in v)
C = len(cs)
B = np.stack([(np.abs(np.array(raw[c]['W'], dtype=float)) > TAU).astype(np.int64) for c in cs], axis=0)
off = ~np.eye(d, dtype=bool)


def stats_of(Bin):
    pres = Bin.sum(axis=0).astype(np.int64)
    v = pres[off]
    Q = float((v.astype(np.float64) ** 2).sum())
    Nk = {k: int((v >= k).sum()) for k in KS}
    return Q, Nk


obsQ, obsNk = stats_of(B)
print('observed: Q=%.0f | N_k=' % obsQ, obsNk)

rng = np.random.default_rng(SEED)
nQ = np.empty(P)
nNk = {k: np.empty(P, dtype=np.int64) for k in KS}
for t in range(P):
    Bp = np.empty_like(B)
    for k in range(C):
        perm = rng.permutation(d)
        Bp[k] = B[k][np.ix_(perm, perm)]
    q, nk = stats_of(Bp)
    nQ[t] = q
    for k in KS:
        nNk[k][t] = nk[k]

mu, sd = float(nQ.mean()), float(nQ.std(ddof=1))
z = (obsQ - mu) / sd if sd > 0 else float('inf')
pv = float((np.sum(nQ >= obsQ) + 1) / (P + 1))
print('Q: null mean=%.0f sd=%.0f max=%.0f' % (mu, sd, nQ.max()))
print('Q: z=%.1f | p_perm=%.4f' % (z, pv))
print()
print('k | observed | null_mean | null_max | p_perm')
row = {}
for k in KS:
    m = float(nNk[k].mean())
    mx = int(nNk[k].max())
    p2 = float((np.sum(nNk[k] >= obsNk[k]) + 1) / (P + 1))
    row[k] = dict(observed=obsNk[k], null_mean=m, null_max=mx, p_perm=p2)
    print('%2d | %8d | %9.2f | %8d | %.4f' % (k, obsNk[k], m, mx, p2))

rep = json.load(open(OUT, encoding='utf-8'))
rep['continuous'] = dict(
    statistic='Q = sum_{i!=j} presence^2 (node-permutation)',
    observed_Q=obsQ, null_Q_mean=mu, null_Q_sd=sd, null_Q_max=float(nQ.max()),
    z=z, p_perm=pv, permutations=P, seed=SEED,
    recurrence_counts=row,
    note_total_edges_invariant='total edge count is invariant under node permutation (2239); not usable as a test statistic')
json.dump(rep, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('\nupdated', OUT)
