# -*- coding: utf-8 -*-
"""Does a refit with BLAS threads pinned to 1 still reproduce the cached W bit-for-bit?

If yes, the parallel refit is numerically identical to the original pipeline and the
task-parallel rewrite is safe.  If no, we must keep the original thread setting.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[v] = '1'
import numpy as np
import _rev_lib as L

panel = L.load_panel()['panel']
fits, cs = L.load_fits('_shared_panel_notears.json')

for c in ['CHOL', 'ACC', 'BLCA']:
    X, _ = L.expr(c, panel)
    W, h = L.notears_lbfgs(X, lam=0.01, max_outer=100)
    W0 = np.array(fits[c]['W'], float)
    print('%-5s n=%4d  max|dW|=%.3e  edges_new=%d edges_cached=%d  same=%s'
          % (c, X.shape[0], np.abs(W - W0).max(),
             int((np.abs(W) > L.TAU).sum()), int(fits[c]['edges']),
             bool(np.array_equal(W, W0))))
