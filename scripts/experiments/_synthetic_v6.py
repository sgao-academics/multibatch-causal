#!/usr/bin/env python3
"""Self-contained V6 synthetic validation (inlines MultiBatchCausalV6).

Reproduces the manuscript numbers for the synthetic benchmark:
    h(W0) = 5.2e-6, shared 10/11, batch-specific 4/4, TOTAL 14/15 (93%).

This is the exact V6 two-stage pipeline (per-batch NOTEARS + median
augmented-Lagrangian DAG projection + batch-specific decomposition),
previously implemented in _archive/multibatch_causal.py. It is inlined here
so the reproduction package is fully self-contained: running run_all.py's
STAGE 5 calls this script and regenerates results/_v6_original_output.json
(and the matching fields of results/synth_ckpt.json).

IMPORTANT batch-label convention (kept consistent with the manuscript, Fig.1
generator scripts/figures/gen_fig1.py and the original _v6_original_output.json):
  * The "fit" order (how data is generated / V6 is run) places the rewired
    batch at array index 2, so that v6['Deltas'][2] is the rewired+new batch
    used by gen_fig1 panel (d).
  * The "display" order written to synth_ckpt['W_trues'] places the rewired
    batch at index 1, so that gen_fig1 panel (b) reads W_trues[1] as the
    "ground truth Batch 2 (rewired + new)".

Dependencies: numpy, scipy. Deterministic with seed 42.
"""
import os, sys, json, time
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

# Paths (script lives at <repo>/scripts/experiments/_synthetic_v6.py)
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(BASE, 'results')
os.makedirs(RESULTS, exist_ok=True)


def h_constraint(W):
    """DAG constraint h(W) = tr(e^{W o W}) - d.  h = 0 iff W is a DAG."""
    return np.trace(expm(W * W)) - W.shape[0]


def notears_single(X, lam=0.01, max_outer=100):
    """NOTEARS via L-BFGS-B. Returns a d x d weight matrix."""
    d = X.shape[1]
    cov = X.T @ X / X.shape[0]
    w_vec = np.zeros(d * d)
    rho, alpha = 0.1, 0.0
    prev_h = 1e10
    for i in range(max_outer):
        def loss(w):
            W = w.reshape(d, d)
            diff = np.eye(d) - W
            return 0.5 * np.trace(diff.T @ cov @ diff) + lam * np.sum(np.abs(W))

        def loss_grad(w):
            W = w.reshape(d, d)
            g = cov @ (W - np.eye(d))
            if lam > 0:
                g += lam * np.sign(W)
            return g.flatten()

        def lagrangian(w):
            W = w.reshape(d, d)
            h_v = np.trace(expm(W * W)) - d
            return loss(w) + 0.5 * rho * h_v ** 2 + alpha * h_v

        def lagrangian_grad(w):
            W = w.reshape(d, d)
            h_v = np.trace(expm(W * W)) - d
            dh = 2 * W * expm(W * W).T
            return loss_grad(w) + rho * h_v * dh.flatten() + alpha * dh.flatten()

        res = minimize(lagrangian, w_vec, method='L-BFGS-B', jac=lagrangian_grad,
                       options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-14})
        w_vec = res.x
        W = w_vec.reshape(d, d)
        h_v = np.trace(expm(W * W)) - d
        alpha += rho * h_v
        if abs(h_v) > abs(prev_h) and abs(h_v) > 1e-10:
            rho = min(rho * 2.0, 1e8)
        prev_h = h_v
        if abs(h_v) < 1e-8:
            break
    return w_vec.reshape(d, d)


def project_to_dag(W_target, lam1=0.001, max_iter=100):
    """Project W_target onto the DAG space:
    min_W ||W - W_target||^2 + lam1 |W|_1  s.t.  h(W) = 0."""
    d = W_target.shape[0]
    W = W_target.copy()
    rho, alpha = 0.05, 0.0
    prev_h = abs(h_constraint(W))
    for i in range(max_iter):
        def lagrangian(w_vec):
            W_cur = w_vec.reshape(d, d)
            diff = W_cur - W_target
            loss = 0.5 * np.sum(diff ** 2) + lam1 * np.sum(np.abs(W_cur))
            h_v = h_constraint(W_cur)
            return loss + 0.5 * rho * h_v ** 2 + alpha * h_v

        def lagrangian_grad(w_vec):
            W_cur = w_vec.reshape(d, d)
            diff = W_cur - W_target
            grad = diff + lam1 * np.sign(W_cur)
            h_v = h_constraint(W_cur)
            dh = 2 * W_cur * expm(W_cur * W_cur).T
            return (grad + rho * h_v * dh + alpha * dh).flatten()

        res = minimize(lagrangian, W.flatten(), method='L-BFGS-B',
                       jac=lagrangian_grad, options={'maxiter': 100, 'ftol': 1e-12})
        W = res.x.reshape(d, d)
        h_v = h_constraint(W)
        alpha += rho * h_v
        if abs(h_v) > 0.5 * prev_h and abs(h_v) > 1e-8:
            rho = min(rho * 2.0, 1e6)
        prev_h = abs(h_v)
        if abs(h_v) < 1e-8:
            break
    return W


def run():
    d, K = 10, 3
    n = 500
    np.random.seed(42)

    # --- Ground truth (shared backbone) ---
    W_shared = np.zeros((d, d))
    for i in range(d - 1):
        W_shared[i, i + 1] = 0.7
    W_shared[2, 5] = 0.5
    W_shared[5, 8] = -0.4

    # --- FIT order: index 1 = edges (2,6),(4,9); index 2 = rewired (2,5)+new (0,7) ---
    # This is the ordering used to generate data and run V6; Deltas[2] ends up as
    # the rewired+new batch, matching gen_fig1 panel (d).
    W_trues_fit = []
    for k in range(K):
        Wk = W_shared.copy()
        if k == 1:
            Wk[2, 6] = 0.8
            Wk[4, 9] = -0.6
        if k == 2:
            Wk[2, 5] = -0.5   # rewired
            Wk[0, 7] = 0.9    # new
        W_trues_fit.append(Wk)

    for k in range(K):
        assert abs(h_constraint(W_trues_fit[k])) < 1e-10, f"B{k} not a DAG"

    # --- Generate data in FIT order ---
    X_list = [np.random.randn(n, d) * 0.02 @ np.linalg.inv(np.eye(d) - W_trues_fit[k])
              for k in range(K)]

    # --- Stage 1: per-batch NOTEARS ---
    W_per_batch = []
    for k in range(K):
        W_per_batch.append(notears_single(X_list[k], lam=0.0))

    # --- Stage 2: median init + DAG projection + batch-specific decomposition ---
    W0_init = np.median(np.stack(W_per_batch), axis=0)
    W0 = project_to_dag(W0_init, lam1=0.001)
    Deltas = []
    for k in range(K):
        D = W_per_batch[k] - W0
        D[np.abs(D) < 0.01] = 0   # soft-threshold (lam2 = 0.01)
        Deltas.append(D)

    # --- Evaluate (tau = 0.2) ---
    tau = 0.2
    shared_true = [(i, i + 1) for i in range(d - 1)] + [(2, 5), (5, 8)]
    spec_edges = {1: [(2, 6), (4, 9)], 2: [(2, 5), (0, 7)]}

    shared_ok = sum(1 for i, j in shared_true if abs(W0[i, j]) > tau)
    spec_ok = 0
    spec_tot = 0
    for k, edges in spec_edges.items():
        for i, j in edges:
            spec_tot += 1
            if abs(Deltas[k][i, j]) > tau:
                spec_ok += 1
    total_ok = shared_ok + spec_ok
    total_gt = len(shared_true) + spec_tot
    h_W0 = h_constraint(W0)
    recovery_pct = 100.0 * total_ok / total_gt

    print(f"\nV6 synthetic validation (deterministic, seed 42)")
    print(f"  W0 shared:   {shared_ok}/{len(shared_true)}")
    print(f"  Delta spec:  {spec_ok}/{spec_tot}")
    print(f"  h(W0) = {h_W0:.2e}")
    print(f"  TOTAL: {total_ok}/{total_gt} ({recovery_pct:.1f}%)")

    # --- Write _v6_original_output.json (manuscript & fig1 source of truth) ---
    v6_out = {
        'W0': W0.tolist(),
        'Deltas': [D.tolist() for D in Deltas],
        'h_W0': float(h_W0),
        'shared_ok': shared_ok, 'spec_ok': spec_ok,
        'total_ok': total_ok, 'total_gt': total_gt,
        'lam1': 0.001, 'lam2': 0.01, 'tau': tau,
    }
    v6_path = os.path.join(RESULTS, '_v6_original_output.json')
    with open(v6_path, 'w') as f:
        json.dump(v6_out, f)
    print(f"Saved: {v6_path}")

    # --- Display order for synth_ckpt['W_trues']: index 0 = spec, index 1 = rewired,
    #     index 2 = shared-only. This is what gen_fig1 panel (b) expects (W_trues[1]). ---
    W_trues_disp = []
    for k in range(K):
        Wk = W_shared.copy()
        if k == 0:
            Wk[2, 6] = 0.8
            Wk[4, 9] = -0.6
        if k == 1:
            Wk[2, 5] = -0.5   # rewired
            Wk[0, 7] = 0.9    # new
        W_trues_disp.append(Wk)

    synth_path = os.path.join(RESULTS, 'synth_ckpt.json')
    if os.path.exists(synth_path):
        synth = json.load(open(synth_path))
    else:
        synth = {}
    synth['d'] = d
    synth['K'] = K
    synth['n_per_batch'] = n
    synth['seed'] = 42
    synth['tau_eval'] = tau
    synth['W0_true'] = W_shared.tolist()
    synth['W_trues'] = [W.tolist() for W in W_trues_disp]
    synth['W0_rec'] = W0.tolist()
    synth['Deltas'] = [D.tolist() for D in Deltas]
    synth['h0_final'] = float(h_W0)
    synth['h_W0'] = float(h_W0)
    synth['shared_ok'] = shared_ok
    synth['spec_ok'] = spec_ok
    synth['total_ok'] = total_ok
    synth['total_gt'] = total_gt
    synth['recovery_pct'] = recovery_pct
    synth['recovery'] = recovery_pct
    synth['shared_edges'] = [list(e) for e in shared_true]
    synth['spec_edges_per_batch'] = {
         '0': [[2, 6], [4, 9]],
         '1': [[2, 5], [0, 7]],
         '2': [],
    }
    with open(synth_path, 'w') as f:
        json.dump(synth, f)
    print(f"Synced metrics into: {synth_path}")


if __name__ == '__main__':
    run()
