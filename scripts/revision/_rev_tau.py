# -*- coding: utf-8 -*-
"""N3: the recurrence count as a function of the edge threshold tau.

Nothing is refitted -- the 33 aligned-panel weight matrices are already stored and tau only
thresholds their output -- so this can be recomputed at any time from the cache.  It is
reported so the headline count ('14 pairs recur in >=10 cohorts') can be read against its
threshold dependence rather than taken as a threshold-free quantity.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _rev_lib as L  # noqa: E402

TAUS = (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
MIN_COHORTS = 10
# the two chemokine edges named in the abstract as the headline module
HEADLINE = (('CXCL9', 'CXCL10'), ('CXCL10', 'CXCL11'))


def main():
    fits, cs = L.load_fits('_shared_panel_notears.json')
    g0 = list(fits[cs[0]]['genes'])
    A = np.array([np.asarray(fits[c]['W'], float) for c in cs])  # (n_cohorts, d, d)
    med = np.median(A, axis=0)
    np.fill_diagonal(med, 0.0)
    gi = {g: i for i, g in enumerate(g0)}

    out = dict(min_cohorts=MIN_COHORTS, n_cohorts=len(cs), tau=[],
               recurrence=[], median_edges=[], edge_counts={}, max_abs_median=None,
               note=('tau thresholds the stored aligned-panel matrices; no refitting. '
                     'recurrence = directed pairs recovered in >=10 of the cohorts.'))
    for (a, b) in HEADLINE:
        out['edge_counts']['%s|%s' % (a, b)] = []
    for t in TAUS:
        cnt = (np.abs(A) > t).sum(axis=0)
        out['tau'].append(float(t))
        out['recurrence'].append(int((cnt >= MIN_COHORTS).sum()))
        out['median_edges'].append(int((np.abs(med) > t).sum()))
        for (a, b) in HEADLINE:
            out['edge_counts']['%s|%s' % (a, b)].append(int(cnt[gi[a], gi[b]]))
    out['max_abs_median'] = float(np.abs(med).max())
    L.save(out, 'tau_recurrence.json')

    print('   tau   recurrence   median>tau   ' +
          '   '.join('%s->%s' % h for h in HEADLINE))
    for i, t in enumerate(out['tau']):
        print('%6.2f %12d %13d   ' % (t, out['recurrence'][i], out['median_edges'][i]) +
              '   '.join('%9d' % out['edge_counts']['%s|%s' % h][i] for h in HEADLINE))
    print('validation: max |median| = %.4f (text 0.731); '
          'median edges at tau=0.30 = %d (text 6)'
          % (out['max_abs_median'], out['median_edges'][out['tau'].index(0.30)]))


if __name__ == '__main__':
    main()
