# -*- coding: utf-8 -*-
"""W2-M2: refit NOTEARS for all 33 cohorts on the second panel (robustness).

NOTEARS is taken verbatim from the main pipeline; the only variable is the gene panel.
Set SMOKE=1 to fit only two cohorts.

Output: results/_panel2_notears.json
"""
import os, json, time
import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.optimize import minimize

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
DATA = os.path.join(_ROOT, 'data')
RES = os.path.join(_ROOT, 'results')
PANELS = os.path.join(DATA, 'panels')
FIG = os.path.join(_ROOT, 'figures')

PANEL = os.path.join(PANELS, 'panel_B_100genes.json')
CKPT = os.path.join(RES, '_panel2_notears.json')

TAU, LAM = 0.3, 0.01
MAX_OUTER = 100
SMOKE = os.environ.get('SMOKE') == '1'


def notears_lbfgs(X, lam=LAM, max_outer=MAX_OUTER):
    """verbatim from run_all.py"""
    n, d = X.shape
    cov = X.T @ X / n
    w_vec = np.zeros(d * d)
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


meta = json.load(open(PANEL, encoding='utf-8'))
panel = meta['panel']
cancers = json.load(open(os.path.join(PANELS, 'panel_A_100genes.json'),
                         encoding='utf-8'))['cancers']
print('PANEL-2 robustness run | panel =', len(panel), '| cancers =', len(cancers),
      '| SMOKE =', SMOKE, flush=True)
print('panel2 =', panel, flush=True)

store = json.load(open(CKPT, encoding='utf-8')) if os.path.exists(CKPT) else {}
done = [k for k, v in store.items() if isinstance(v, dict) and 'W' in v]
todo = [c for c in cancers if c not in done]
if SMOKE:
    todo = todo[:2]
print('already done = %d | todo = %d' % (len(done), len(todo)), flush=True)

for idx, c in enumerate(todo):
    t0 = time.time()
    df = pd.read_csv(os.path.join(DATA, 'TCGA_%s_HiSeqV2.tsv' % c), sep='\t', index_col=0).T
    missing = [g for g in panel if g not in df.columns]
    if missing:
        print('  !! %s missing %d panel genes' % (c, len(missing)), flush=True)
    X = df.reindex(columns=panel).values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
    W, h = notears_lbfgs(X)
    nz = int(np.sum(np.abs(W) > TAU))
    store[c] = {'W': W.tolist(), 'edges': nz, 'h': float(h), 'genes': panel, 'n': int(X.shape[0])}
    with open(CKPT, 'w', encoding='utf-8') as f:
        json.dump(store, f)
    print('  [%d/%d] %-5s n=%4d edges=%3d h=%.2e  %.0fs'
          % (idx + 1, len(todo), c, X.shape[0], nz, h, time.time() - t0), flush=True)

valid = [k for k, v in store.items() if isinstance(v, dict) and 'W' in v]
print('\nDONE. cohorts =', len(valid), '| total edges =', sum(store[k]['edges'] for k in valid))
