# -*- coding: utf-8 -*-
"""Second speed probe: with BLAS threads capped at 1, how many worker processes scale?"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['REV_DIR'] = '_rev_speed'
# cap BLAS threads *before* any child is spawned so children inherit 1-thread BLAS
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[v] = '1'
import numpy as np
import _rev_lib as L

PANEL = L.load_panel()['panel']
COHORTS = ['ACC', 'BLCA', 'BRCA', 'LUAD', 'COAD', 'KIRC']


def _one(c):
    X, _ = L.expr(c, PANEL)
    W, h = L.notears_lbfgs(X, lam=0.01, max_outer=100)
    return int((np.abs(W) > L.TAU).sum())


def main():
    from concurrent.futures import ProcessPoolExecutor
    X, _ = L.expr('BLCA', PANEL)
    t0 = time.perf_counter(); c0 = time.process_time()
    L.notears_lbfgs(X, lam=0.01, max_outer=100)
    print('ONE FIT (OMP=1)  wall=%.1fs cpu=%.1fs cpu/wall=%.2f'
          % (time.perf_counter() - t0, time.process_time() - c0,
             (time.process_time() - c0) / (time.perf_counter() - t0)))

    base = None
    for nw in (8, 12, 16, 20):
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=nw) as ex:
            got = list(ex.map(_one, COHORTS))
        t_par = time.perf_counter() - t0
        if base is None:
            base = (got, t_par)
        print('PAR%-2d  %d fits  %.0fs  (%.1fs/fit)  match=%s'
              % (nw, len(COHORTS), t_par, t_par / len(COHORTS), got == base[0]))


if __name__ == '__main__':
    main()
