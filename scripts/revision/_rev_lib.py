# -*- coding: utf-8 -*-
"""Shared library for the revision compute batch.

notears_lbfgs is copied verbatim from scripts/analysis/fit_panel_A.py (which in turn
takes it verbatim from run_all.py).  The unit test _rev_unit.py checks that this copy
reproduces the cached W bit-for-bit before any refitting is allowed to run.
"""
import io
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# The repository root is the grandparent of this file (scripts/revision/_rev_lib.py), so the
# batch runs from a checkout anywhere.  MULTIBATCH_ROOT overrides it, which is what the smoke
# run uses to redirect every artefact into a sandbox.
ROOT = os.environ.get('MULTIBATCH_ROOT') or os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, 'data')
RES = os.path.join(ROOT, 'results')
PANELS = os.path.join(DATA, 'panels')
TCGA_DIR = os.environ.get('MULTIBATCH_DATA') or DATA
# everything this batch writes lives here.  REV_DIR lets the smoke test run in its own
# sandbox so it can never leave partial checkpoints in the real run's path.
REV = os.path.join(RES, os.environ.get('REV_DIR', '_rev'))
os.makedirs(REV, exist_ok=True)

TAU = 0.3

# ----------------------------------------------------------------- NOTEARS
def notears_lbfgs(X, lam=0.01, max_outer=100, seed=None):
    """verbatim from scripts/analysis/fit_panel_A.py -- do not edit.

    seed only perturbs the starting point; seed=None reproduces the original fit.
    """
    from scipy.linalg import expm
    from scipy.optimize import minimize
    n, d = X.shape
    cov = X.T @ X / n
    w_vec = np.zeros(d * d)
    if seed is not None:
        rng = np.random.default_rng(seed)
        w_vec = rng.normal(0.0, 0.01, d * d)
    rho, alpha = 0.1, 0.0
    prev_h = 1e10
    best_w, best_h = w_vec.copy(), 1e10
    for i in range(max_outer):
        lag = lambda w: 0.5 * np.trace((np.eye(d) - w.reshape(d, d)).T @ cov @ (np.eye(d) - w.reshape(d, d))) \
            + lam * np.sum(np.abs(w.reshape(d, d))) \
            + 0.5 * rho * (np.trace(expm(w.reshape(d, d) ** 2)) - d) ** 2 \
            + alpha * (np.trace(expm(w.reshape(d, d) ** 2)) - d)

        def lag_grad(w):
            W = w.reshape(d, d)
            hv = np.trace(expm(W * W)) - d
            dh = 2 * W * expm(W * W).T
            g = cov @ (W - np.eye(d)) + lam * np.sign(W)
            return g.flatten() + rho * hv * dh.flatten() + alpha * dh.flatten()

        res = minimize(lag, w_vec, method='L-BFGS-B', jac=lag_grad,
                       options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-14})
        w_vec = res.x
        W = w_vec.reshape(d, d)
        hv = np.trace(expm(W * W)) - d
        alpha += rho * hv
        if abs(hv) < best_h:
            best_h, best_w = abs(hv), w_vec.copy()
        if abs(hv) > abs(prev_h) and abs(hv) > 1e-8:
            rho = min(rho * 2.0, 1e10)
        prev_h = hv
        if abs(hv) < 1e-7:
            break
    return best_w.reshape(d, d), best_h


# ----------------------------------------------------------------- data
_EXPR = {}


def load_panel(name='panel_A_100genes.json'):
    return json.load(io.open(os.path.join(PANELS, name), encoding='utf-8'))


def expr(cancer, genes, zscore=True):
    """samples x len(genes), standardised per column (same as the original pipeline)."""
    key = (cancer, tuple(genes), zscore)
    if key in _EXPR:
        return _EXPR[key]
    path = os.path.join(TCGA_DIR, 'TCGA_%s_HiSeqV2.tsv' % cancer)
    df = pd.read_csv(path, sep='\t', index_col=0).T
    miss = [g for g in genes if g not in df.columns]
    X = df.reindex(columns=list(genes)).values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0)
    if zscore:
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
    _EXPR[key] = (X, miss)
    return X, miss


def all_cancers():
    return sorted(f.replace('TCGA_', '').replace('_HiSeqV2.tsv', '')
                  for f in os.listdir(TCGA_DIR)
                  if f.startswith('TCGA_') and f.endswith('_HiSeqV2.tsv'))


def load_fits(name):
    p = os.path.join(RES, name)
    d = json.load(io.open(p, encoding='utf-8'))
    cs = sorted(k for k, v in d.items() if isinstance(v, dict) and 'W' in v)
    return d, cs


# ----------------------------------------------------------------- stats
def median_matrix(fits, cancers, dref=None):
    mats = []
    for c in cancers:
        W = np.abs(np.array(fits[c]['W'], dtype=float))
        if dref is not None and W.shape[0] != dref:
            continue
        mats.append(W)
    M = np.stack(mats, axis=0)
    med = np.median(M, axis=0)
    np.fill_diagonal(med, 0.0)
    return M, med


def recurrence(M, tau=TAU):
    """M is cohorts x d x d of |W|.  Returns per-directed-pair counts."""
    present = (M > tau).sum(axis=0)
    return present


def recurrence_list(present, genes, min_cohorts=10):
    d = present.shape[0]
    out = []
    for i in range(d):
        for j in range(d):
            if i != j and present[i, j] >= min_cohorts:
                out.append(dict(src=genes[i], dst=genes[j], n=int(present[i, j])))
    return sorted(out, key=lambda r: -r['n'])


def qstat(present):
    return float((present.astype(float) ** 2).sum())


def nk_counts(present, levels=(2, 3, 5, 10, 15)):
    return {int(k): int((present >= k).sum()) for k in levels}


# ----------------------------------------------------------------- io
def save(obj, name):
    """Write atomically: serialise to a temp file, then replace.

    json.dump streams, so a key it cannot serialise used to leave a truncated file behind
    (a tuple key in composition.json did exactly that).  Writing via a temp file means a
    failed save leaves the previous good version untouched.
    """
    p = os.path.join(REV, name)
    tmp = p + '.tmp'
    with io.open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, p)
    return p


def load_rev(name):
    p = os.path.join(REV, name)
    if not os.path.exists(p):
        return None
    return json.load(io.open(p, encoding='utf-8'))


def log(msg):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), msg), flush=True)


def burst(n):
    """cap thread counts so parallel stages do not fight for the machine.

    Measured on this machine: a single d=100 fit takes 57 s with 32-thread BLAS and
    40 s with 1-thread BLAS, and 1-thread BLAS still reproduces the cached W
    bit-for-bit (_rev_unit1.py) -- so pinning BLAS to one thread is both faster and
    numerically identical.  Cohort-level parallelism then does the real work.
    """
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS'):
        os.environ[v] = str(n)


# ----------------------------------------------------------------- parallelism
# 32 logical cores; measured throughput is ~5x at 8 workers for cohort-sized fits.
WORKERS = max(1, int(os.environ.get('REV_WORKERS', '1')))


def pmap(fn, items):
    """Apply fn to each item, yielding results in completion order.

    fn must be a module-level function (picklable) and items must be picklable.
    Falls back to a plain loop when one worker is configured, so the serial path stays
    available as a reference implementation.  Yields results only (not the input item) --
    an earlier version yielded (item, result) and every call site silently unpacked it
    one level wrong.
    """
    items = list(items)
    if WORKERS <= 1 or len(items) <= 1:
        for it in items:
            yield fn(it)
        return
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(fn, it) for it in items]
        for f in as_completed(futs):
            yield f.result()
