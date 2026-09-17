"""Baseline comparison statistics behind Figure 3.

Single source of truth for the NOTEARS versus GENIE3 comparison. Two distinct
counts have to be kept apart, and both are written here:

  * the cross-cancer sharing rate, which is reported under both counting
    conventions because the manuscript quotes both wherever the rate appears:
    a recurrence is counted as directed (i, j) or, disregarding orientation,
    as the unordered pair {i, j}. GENIE3 scores a candidate regulator for a
    target gene, so its output carries an orientation too, but that
    orientation is a feature-importance ranking rather than a causal
    direction, which is why both ways of counting are reported;

  * the overlap between the two methods, which is the number of distinct
    gene-pairs the two edge sets have in common. A pair recovered in k
    cohorts contributes one pair to the overlap, not k.

Inputs
------
results/_pipeline_notears.json    per-cohort W matrices, gene lists, n
results/_genie3_lbfgs_ckpt.json   per-cohort GENIE3 top-K edge lists
results/_pipeline_pooled.json     pooled NOTEARS fit

Output
------
results/_baseline_stats.json

Run:  python scripts/figures/gen_baseline_stats.py
"""
import json
import os
import sys
from collections import Counter

import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(BASE, 'results')
TAU = 0.3

CKPT_NOTEARS = os.path.join(RESULTS, '_pipeline_notears.json')
CKPT_GENIE3 = os.path.join(RESULTS, '_genie3_lbfgs_ckpt.json')
CKPT_POOLED = os.path.join(RESULTS, '_pipeline_pooled.json')
CKPT_LEGACY = os.path.join(RESULTS, '_genie3_overlap_lbfgs.json')
OUT = os.path.join(RESULTS, '_baseline_stats.json')


def load(path):
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def note_edge_sets(notears):
    """Directed and undirected pair sets per cancer, with a validation pass.

    Every cohort's edge list is rebuilt from the stored W matrix with the same
    threshold the pipeline uses, and compared against the count stored
    alongside it, so a silent change in either cannot slip through.
    """
    directed, undirected = {}, {}
    mismatch = []
    for name in sorted(notears):
        rec = notears[name]
        W = np.array(rec['W'], dtype=float)
        genes = [str(g).upper() for g in rec['genes']]
        d = {(genes[i], genes[j])
             for i in range(W.shape[0]) for j in range(W.shape[1])
             if i != j and abs(W[i, j]) > TAU}
        directed[name] = d
        undirected[name] = {tuple(sorted(p)) for p in d}
        if len(d) != rec['edges']:
            mismatch.append((name, len(d), rec['edges']))
    if mismatch:
        raise SystemExit('threshold no longer reproduces the stored edge counts: %s' % mismatch)
    return directed, undirected


def g3_edge_sets(genie3):
    directed, undirected = {}, {}
    for name in sorted(genie3):
        d = {(str(e[0]).upper(), str(e[1]).upper()) for e in genie3[name]['edges']}
        directed[name] = d
        undirected[name] = {tuple(sorted(p)) for p in d}
    return directed, undirected


def recur(per_cancer, unordered):
    """Recurrence counter over the union of all cohorts."""
    cnt = Counter()
    for name in sorted(per_cancer):
        for p in per_cancer[name]:
            cnt[tuple(sorted(p)) if unordered else p] += 1
    return cnt


def summarise(cnt):
    n = len(cnt)
    ge3 = sum(1 for v in cnt.values() if v >= 3)
    return {
        'unique_pairs': n,
        'pairs_in_1': sum(1 for v in cnt.values() if v == 1),
        'pairs_in_2': sum(1 for v in cnt.values() if v == 2),
        'pairs_in_ge3': ge3,
        'pct_ge3': round(100.0 * ge3 / max(n, 1), 1),
    }


def main():
    notears = load(CKPT_NOTEARS)
    genie3 = load(CKPT_GENIE3)

    nt_dir, nt_und = note_edge_sets(notears)
    g3_dir, g3_und = g3_edge_sets(genie3)

    nt_edges = sum(len(v) for v in nt_dir.values())
    g3_edges = sum(len(v) for v in g3_dir.values())

    nt_dir_c, nt_und_c = recur(nt_dir, False), recur(nt_und, True)
    g3_dir_c, g3_und_c = recur(g3_dir, False), recur(g3_und, True)

    out = {
        'tau': TAU,
        'n_cohorts': len(nt_dir),
        'notears': {
            'total_edges': nt_edges,
            'directed': summarise(nt_dir_c),
            'undirected': summarise(nt_und_c),
        },
        'genie3': {
            'total_edges': g3_edges,
            'directed': summarise(g3_dir_c),
            'undirected': summarise(g3_und_c),
        },
        'overlap': {},
    }

    # The overlap between two gene-pair sets is reported on distinct pairs: a
    # pair recovered in k cohorts is one pair, not k, and the per-cohort sum is
    # recorded separately because it is a different quantity.
    for label, nt, g3 in (('directed', nt_dir_c, g3_dir_c),
                          ('undirected', nt_und_c, g3_und_c)):
        nt_set, g3_set = set(nt), set(g3)
        shared = nt_set & g3_set
        out['overlap'][label] = {
            'shared_pairs': len(shared),
            'notears_only_pairs': len(nt_set - g3_set),
            'genie3_only_pairs': len(g3_set - nt_set),
            'pct_of_notears': round(100.0 * len(shared) / max(len(nt_set), 1), 1),
            'pct_of_genie3': round(100.0 * len(shared) / max(len(g3_set), 1), 1),
            'per_cohort_sum': sum(len(nt_dir[c] & g3_dir[c]) for c in nt_dir)
                              if label == 'directed' else
                              sum(len(nt_und[c] & g3_und[c]) for c in nt_und),
        }

    pooled = load(CKPT_POOLED)
    pooled_edges = pooled['edges']
    out['pooled'] = {
        'edges': pooled_edges,
        'h': pooled.get('h'),
        'n': pooled.get('n'),
        'fold_vs_per_cancer_total': round(nt_edges / max(pooled_edges, 1), 1),
        'fold_vs_mean_per_cancer': round(nt_edges / len(nt_dir) / max(pooled_edges, 1), 1),
    }

    # Cross-check against the historical aggregate. The pair counts agree once
    # the historical file is read as an undirected count, which is what it is;
    # its "overlap_edges" is the per-cohort sum, not a distinct-pair count, and
    # its notears denominator is an edge count. Both are recorded so that the
    # difference between the two quantities stays visible.
    if os.path.exists(CKPT_LEGACY):
        legacy = load(CKPT_LEGACY)
        out['cross_check_vs_legacy'] = {
            'legacy_genie3_unique_pairs': legacy.get('genie3_unique_pairs'),
            'here_undirected': out['genie3']['undirected']['unique_pairs'],
            'legacy_genie3_shared_gte3': legacy.get('genie3_shared_gte3'),
            'here_undirected_ge3': out['genie3']['undirected']['pairs_in_ge3'],
            'note': ('the legacy file counted GENIE3 disregarding orientation and '
                     'NOTEARS with orientation; its overlap figure was a per-cohort '
                     'sum divided by an edge count, while the distinct-pair overlap '
                     'is reported here'),
        }

    tmp = OUT + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=2)
    os.replace(tmp, OUT)

    ntd, ntu = out['notears']['directed'], out['notears']['undirected']
    g3d, g3u = out['genie3']['directed'], out['genie3']['undirected']
    ovd, ovu = out['overlap']['directed'], out['overlap']['undirected']
    print('cohorts %d  tau %.2f' % (out['n_cohorts'], TAU))
    print('NOTEARS  %d edges | directed %d pairs, %d in >=3 (%.1f%%)'
          % (nt_edges, ntd['unique_pairs'], ntd['pairs_in_ge3'], ntd['pct_ge3']))
    print('                   | undirected %d pairs, %d in >=3 (%.1f%%)'
          % (ntu['unique_pairs'], ntu['pairs_in_ge3'], ntu['pct_ge3']))
    print('GENIE3   %d edges | directed %d pairs, %d in >=3 (%.1f%%)'
          % (g3_edges, g3d['unique_pairs'], g3d['pairs_in_ge3'], g3d['pct_ge3']))
    print('                   | undirected %d pairs, %d in >=3 (%.1f%%)'
          % (g3u['unique_pairs'], g3u['pairs_in_ge3'], g3u['pct_ge3']))
    print('overlap  directed   %d pairs (%.1f%% of NOTEARS, %.1f%% of GENIE3)'
          % (ovd['shared_pairs'], ovd['pct_of_notears'], ovd['pct_of_genie3']))
    print('overlap  undirected %d pairs (%.1f%% of NOTEARS, %.1f%% of GENIE3)'
          % (ovu['shared_pairs'], ovu['pct_of_notears'], ovu['pct_of_genie3']))
    print('pooled   %d edges vs %d per-cancer total (%.1fx)'
          % (pooled_edges, nt_edges, out['pooled']['fold_vs_per_cancer_total']))
    print('wrote %s' % os.path.relpath(OUT, BASE))


if __name__ == '__main__':
    main()
