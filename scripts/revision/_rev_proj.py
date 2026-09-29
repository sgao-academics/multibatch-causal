# -*- coding: utf-8 -*-
"""Does the DAG projection converge on the aligned TCGA panel, and at what iteration budget?

The projection operator is the one the synthetic benchmark validates (lam1 = 0.001).  On the
real 100-gene median matrix the first run only moved h from 4.47e-3 to 4.19e-3, which is too
little to claim a DAG-projected backbone, so check whether it is a budget problem.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('REV_DIR', '_rev')
import _rev_lib as L
import _rev_extra as X


def main():
    meta = L.load_panel()
    panel = meta['panel']
    fits, cs = L.load_fits('_shared_panel_notears.json')
    M = np.stack([np.array(fits[c]['W'], dtype=float) for c in cs], axis=0)
    Wbar = np.median(M, axis=0)
    np.fill_diagonal(Wbar, 0.0)
    print('h(median) = %.6e   max|median| = %.4f   edges>tau = %d'
          % (X._h(Wbar), np.abs(Wbar).max(), int((np.abs(Wbar) > L.TAU).sum())))
    print('per-cohort h: min %.2e  median %.2e  max %.2e'
          % tuple(np.percentile([X._h(np.array(fits[c]['W'], float)) for c in cs],
                                [0, 50, 100])))
    for it in (100, 300, 600):
        t0 = time.time()
        W0, h0 = X.project_to_dag(Wbar, lam1=0.001, max_iter=it)
        print('max_iter=%3d -> h(W0)=%.4e  edges>tau=%d  max|W0|=%.4f  %s  (%.0fs)'
              % (it, h0, int((np.abs(W0) > L.TAU).sum()), np.abs(W0).max(),
                 'acyclic' if h0 < 1e-6 else 'still cyclic', time.time() - t0))
        if h0 < 1e-6:
            break


if __name__ == '__main__':
    main()
