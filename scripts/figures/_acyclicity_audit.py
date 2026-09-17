"""Acyclicity diagnostic for the per-cancer graphs.

NOTEARS imposes acyclicity as a differentiable penalty and returns the least-violating
iterate within its iteration budget, so the reported graphs are near-acyclic rather than
exactly acyclic.  This script records, for every cohort, the achieved |h(W)| and how many
of the thresholded edges lie on a directed cycle, so that the paper can state the
violation instead of asserting that it is zero.

Output: results/_acyclicity.json
"""
import os
import sys
import json
import collections

import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(BASE, 'results')
TAU = 0.3


def reachable(A):
    """Transitive closure by repeated Boolean squaring."""
    R = A.astype(bool).copy()
    n = len(A)
    for k in range(n):
        R |= R[:, [k]] & R[[k], :]
    return R


P = json.load(open(os.path.join(RES, '_pipeline_notears.json'), encoding='utf-8'))

per_cohort = {}
tot_edges = 0
tot_cyclic = 0
n_acyclic = 0
shared = collections.defaultdict(int)
flagged_pairs = set()

# first pass: which directed gene-pairs recur in three or more cohorts
for c, d in P.items():
    g = list(d['genes'])
    W = np.array(d['W'], dtype=float)
    n = len(g)
    for i in range(n):
        for j in range(n):
            if i == j or W[i, j] is None:
                continue
            if abs(float(W[i, j])) > TAU:
                shared[(g[i], g[j])] += 1
recurrent = {p for p, v in shared.items() if v >= 3}

for c, d in sorted(P.items()):
    g = list(d['genes'])
    W = np.array(d['W'], dtype=float)
    A = (np.abs(W) > TAU).astype(int)
    np.fill_diagonal(A, 0)
    R = reachable(A)
    C = np.zeros_like(A)
    for i in range(len(A)):
        for j in range(len(A)):
            if A[i, j] and R[j, i]:
                C[i, j] = 1
    cyc = [(g[i], g[j]) for i in range(len(A)) for j in range(len(A)) if C[i, j]]
    for (a, b) in cyc:
        if (a, b) in recurrent:
            flagged_pairs.add((a, b))
    per_cohort[c] = {'h_abs': float(abs(d['h'])), 'n_edges': int(A.sum()),
                     'n_cyclic_edges': int(C.sum()), 'acyclic': bool(C.sum() == 0),
                     'cyclic_edges': ['%s->%s' % t for t in cyc]}
    tot_edges += int(A.sum())
    tot_cyclic += int(C.sum())
    n_acyclic += int(C.sum() == 0)

hs = np.array([per_cohort[c]['h_abs'] for c in per_cohort])
out = {
    'tau': TAU,
    'n_cohorts': len(per_cohort),
    'h_abs_min': float(hs.min()),
    'h_abs_max': float(hs.max()),
    'h_abs_median': float(np.median(hs)),
    'n_cohorts_below_1e-2': int((hs < 1e-2).sum()),
    'solver_early_stop_tolerance': 1e-7,
    'n_cohorts_acyclic_after_threshold': n_acyclic,
    'n_edges_total': tot_edges,
    'n_edges_on_a_cycle': tot_cyclic,
    'pct_edges_on_a_cycle': round(100.0 * tot_cyclic / tot_edges, 2),
    'n_recurrent_directed_pairs': len(recurrent),
    'n_recurrent_pairs_on_a_cycle_somewhere': len(flagged_pairs),
    'recurrent_pairs_on_a_cycle_somewhere': sorted('%s->%s' % p for p in flagged_pairs),
    'per_cohort': per_cohort,
}
OUT = os.path.join(RES, '_acyclicity.json')
json.dump(out, open(OUT, 'w', encoding='utf-8'), indent=1)

print('cohorts %d | |h| %.3g..%.3g (median %.3g)' % (len(per_cohort), hs.min(), hs.max(), np.median(hs)))
print('acyclic after thresholding: %d/%d' % (n_acyclic, len(per_cohort)))
print('edges on a directed cycle: %d/%d (%.2f%%)' % (tot_cyclic, tot_edges, 100.0 * tot_cyclic / tot_edges))
print('recurrent pairs on a cycle somewhere: %d/%d' % (len(flagged_pairs), len(recurrent)))
print('written: %s' % OUT)
