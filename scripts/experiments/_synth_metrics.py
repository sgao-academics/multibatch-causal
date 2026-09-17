#!/usr/bin/env python3
r"""Standard structure-recovery metrics for the synthetic two-stage validation.

Produces results/_synth_metrics.json, the table behind Table 2 of the
manuscript (Precision / Recall / F1 / FDR / directed SHD on the d=10, K=3
synthetic benchmark at tau = 0.2).

Input is results/synth_ckpt.json, written by scripts/experiments/_synthetic_v6.py
in the same stage of run_all.py, which carries:

    W0_true : shared-backbone ground-truth adjacency (d x d)
    W_trues : per-batch ground-truth adjacency (list of K, each d x d)
    W_hats  : per-batch Stage-1 recovered adjacency (list of K)
    W0_rec  : Stage-2 DAG-projected recovered shared backbone
    Deltas  : per-batch deviation matrices (Delta_k = W_hat_k - W0_rec)

Three targets are scored, and each is stored for every tau in
{0.1, 0.15, 0.2, 0.25, 0.3}:

  (A) shared backbone : pred = thresh(W0_rec)      vs gt = thresh(W0_true)
  (B) end-to-end      : pred = thresh(W_hats[k])   vs gt = thresh(W_trues[k])
  (C) batch-specific  : pred = thresh(Delta_k)     vs gt = thresh(W_trues[k] - W0_true)

Standard definitions:
  Precision = |pred & gt| / |pred| ;  Recall = |pred & gt| / |gt|
  F1        = 2PR / (P + R)        ;  FDR    = |pred \ gt| / |pred|
  SHD_dir   = |pred XOR gt| + |{edges opposite in the other set}|
  SHD_und   = |pred| + |gt| - 2|pred & gt|

Usage:
    python scripts/experiments/_synth_metrics.py [out_path]
    (default out_path is results/_synth_metrics.json)

Dependencies: numpy. Deterministic -- reads a checkpoint, does no fitting.
"""
import json
import os
import sys

import numpy as np

# Paths (script lives at <repo>/scripts/experiments/_synth_metrics.py)
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(BASE, 'results')
CKPT = os.path.join(RESULTS, 'synth_ckpt.json')
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(RESULTS, '_synth_metrics.json')

TAUS = [0.1, 0.15, 0.2, 0.25, 0.3]
REPORT_TAU = 0.2          # the operating point used in Table 2 of the manuscript


def load():
    with open(CKPT, encoding='utf-8') as f:
        return json.load(f)


def binarize(M, tau):
    """Directed edges (i, j), i != j, with |M[i, j]| > tau."""
    return {(i, j) for i in range(M.shape[0]) for j in range(M.shape[1])
            if i != j and abs(M[i, j]) > tau}


def metrics(pred, gt):
    inter = pred & gt
    P = len(inter) / len(pred) if pred else 0.0
    R = len(inter) / len(gt) if gt else 0.0
    F1 = (2 * P * R / (P + R)) if (P + R) > 0 else 0.0
    FDR = (len(pred - gt) / len(pred)) if pred else 0.0
    reversed_count = sum(1 for (i, j) in pred if (j, i) in gt and (i, j) not in gt)
    SHD_dir = len(pred ^ gt) + reversed_count
    SHD_und = len(pred) + len(gt) - 2 * len(inter)
    return dict(P=P, R=R, F1=F1, FDR=FDR, SHD_dir=SHD_dir, SHD_und=SHD_und,
                n_pred=len(pred), n_gt=len(gt), n_tp=len(inter))


def to_np(M):
    return np.array(M, dtype=float)


def main():
    ck = load()
    d, K = ck['d'], ck['K']
    W0_true = to_np(ck['W0_true'])
    W_trues = [to_np(W) for W in ck['W_trues']]
    W_hats = [to_np(W) for W in ck['W_hats']]
    W0_rec = to_np(ck['W0_rec'])
    Deltas = [to_np(D) for D in ck['Deltas']]

    print(f"d={d}, K={K}, n_per_batch={ck.get('n_per_batch')}, seed={ck.get('seed')}")
    print(f"tau_eval in checkpoint = {ck.get('tau_eval')}")
    print(f"checkpoint self-report: shared_ok={ck['shared_ok']}, spec_ok={ck['spec_ok']}, "
          f"total_ok={ck['total_ok']}/{ck['total_gt']} ({ck['recovery_pct']:.1f}%), "
          f"h(W0)={ck['h_W0']:.2e}")

    # Each target is scored against its own ground truth: the backbone against
    # W0_true, each per-batch target against W_trues[k].

    res = {}
    for tau in TAUS:
        gt_A = binarize(W0_true, 1e-9)
        pred_A = binarize(W0_rec, tau)
        res[f'shared_backbone_tau{tau}'] = metrics(pred_A, gt_A)
        for k in range(K):
            gt_B = binarize(W_trues[k], 1e-9)
            pred_B = binarize(W_hats[k], tau)          # authentic Stage-1 output
            res[f'end2end_batch{k}_tau{tau}'] = metrics(pred_B, gt_B)
            gt_C = binarize(W_trues[k] - W0_true, 1e-9)
            pred_C = binarize(Deltas[k], tau)
            res[f'batch_specific_batch{k}_tau{tau}'] = metrics(pred_C, gt_C)

        m = res[f'shared_backbone_tau{tau}']
        print(f"tau={tau:<5} backbone P={m['P']:.3f} R={m['R']:.3f} F1={m['F1']:.3f} "
              f"FDR={m['FDR']:.3f} SHD_dir={m['SHD_dir']} (pred={m['n_pred']}, gt={m['n_gt']})")

    print()
    print(f"Table 2 of the manuscript reports the tau={REPORT_TAU} operating point:")
    m = res[f'shared_backbone_tau{REPORT_TAU}']
    print(f"  Stage-2 shared backbone W0  P={m['P']:.2f} R={m['R']:.2f} F1={m['F1']:.2f} "
          f"FDR={m['FDR']:.2f} SHD={m['SHD_dir']}")
    for k in range(K):
        m = res[f'end2end_batch{k}_tau{REPORT_TAU}']
        print(f"  End-to-end Batch {k}          P={m['P']:.2f} R={m['R']:.2f} F1={m['F1']:.2f} "
              f"FDR={m['FDR']:.2f} SHD={m['SHD_dir']}")

    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2, sort_keys=True)
    print(f"\nSaved: {OUT}")


if __name__ == '__main__':
    main()
