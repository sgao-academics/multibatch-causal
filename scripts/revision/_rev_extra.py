# -*- coding: utf-8 -*-
"""Extra revision analyses that need no refitting.

  trrust     B3  TRRUST overlap with counts + matched random background + Fisher p
  stage2     C3  apply the two-stage W0 / Delta decomposition to the aligned TCGA panel
  survival   A-M10 / B4  per-SD continuous Cox, PH check, BH q-values, floor sensitivity

These are deliberately separate from _rev_night.py: they are cheap, they read only
cached results, and keeping them out of the refitting batch means a failure here cannot
disturb the runs that are already checkpointed.

usage:  python _rev_extra.py trrust,stage2,survival
"""
import io
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _rev_lib as L

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                                '..', 'Desktop', 'MultiBatch', 'scripts', 'figures'))
sys.path.insert(0, os.path.join(L.ROOT, 'scripts', 'figures'))
import _gene_symbols  # noqa: E402  (shared display-layer symbol table)

VAL = os.path.join(L.DATA, 'validation')
TAU = L.TAU


def log(m):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), m), flush=True)


def load_json(p):
    return json.load(io.open(p, encoding='utf-8'))


# ================================================================= trrust (B3)
def stage_trrust(n_random=200000, seed=11):
    """TRRUST is a TF->target resource; a dispersion-selected panel barely contains TFs,
    so a near-null overlap is the expected result and has to be reported as such rather
    than described as 'the same picture holds'."""
    ch = load_json(os.path.join(L.RES, '_string_channels.json'))
    # causal_pairs is {open("GENE_A|GENE_B"): [cohort, ...]} -- the same set the STRING
    # analysis uses, so the two resources are compared on an identical pair universe
    cp = ch['causal_pairs']
    pairs = []
    for k in cp:
        if '|' not in k:
            continue
        a, b = k.split('|', 1)
        if a and b and a != b:
            pairs.append((a, b))
    genes = sorted({g for p in pairs for g in p})
    log('causal pairs = %d over %d genes' % (len(pairs), len(genes)))

    tp = os.path.join(VAL, 'trrust_human.tsv')
    trr = set()
    n_rows = 0
    with io.open(tp, encoding='utf-8', errors='replace') as f:
        for line in f:
            f_ = line.rstrip('\n').split('\t')
            if len(f_) < 2:
                continue
            n_rows += 1
            trr.add(tuple(sorted((f_[0].strip(), f_[1].strip()))))
    log('TRRUST rows = %d, unique undirected TF-target pairs = %d' % (n_rows, len(trr)))

    hits = sorted(set(pairs) & trr)
    n_hits = len(hits)

    # matched background: random pairs drawn from the same gene pool
    rng = np.random.default_rng(seed)
    gi = np.array(genes)
    seen = set()
    n_base = 0
    draws = 0
    while draws < n_random:
        a, b = gi[rng.integers(0, len(gi), 2)]
        if a == b:
            continue
        draws += 1
        seen.add(tuple(sorted((a, b))))
    n_base = len(seen & trr)
    pool = len(seen)
    exp_rate = n_base / pool if pool else float('nan')
    obs_rate = n_hits / len(pairs) if pairs else float('nan')

    from scipy.stats import fisher_exact
    npairs_all = len(genes) * (len(genes) - 1) // 2
    a = n_hits
    b = len(pairs) - n_hits
    c = len(trr) - n_hits
    d = npairs_all - len(trr) - b
    orr, p = fisher_exact([[a, b], [c, d]], alternative='greater')

    tf_in_panel = sorted({t for t, _ in trr if t in set(genes)})
    out = dict(
        n_causal_pairs=len(pairs), n_genes=len(genes),
        trrust_rows=n_rows, trrust_pairs=len(trr),
        hits=n_hits, hit_pairs=['%s-%s' % h for h in hits],
        random_pairs_drawn=pool, random_hits=n_base,
        observed_rate_pct=100 * obs_rate, expected_rate_pct=100 * exp_rate,
        fold=(obs_rate / exp_rate if exp_rate else None),
        fisher_or=float(orr), fisher_p=float(p),
        transcription_factors_in_panel=len(tf_in_panel),
        transcription_factors_in_panel_list=tf_in_panel[:50],
        note=('Near-null by expectation: the panel is cut on expression dispersion, and '
              'transcription factors are expressed at stable mid-range levels, so the '
              'panel contains almost none of the TFs TRRUST is built around.'))
    L.save(out, 'trrust_overlap.json')
    log('TRRUST hits = %d/%d (%.2f%%) vs random %.3f%%  fold %.2f  Fisher p = %.3g'
        % (n_hits, len(pairs), 100 * obs_rate, 100 * exp_rate,
           (obs_rate / exp_rate) if exp_rate else float('nan'), p))
    log('TFs present in the panel: %d' % len(tf_in_panel))


# ================================================================= stage2 (C3)
def _h(W):
    from scipy.linalg import expm
    return float(np.trace(expm(W * W)) - W.shape[0])


def project_to_dag_al(W_target, lam1=0.001, max_iter=100, rho0=0.05, gamma=2.0):
    """Same objective and same inner solver, but with the textbook augmented-Lagrangian
    schedule: rho escalates while the constraint violation is still above tolerance.

    project_to_dag() below, which is the operator the synthetic benchmark uses, escalates
    rho only when |h| *increases*.  On the d=100 aligned median that rule never fires,
    because h decreases monotonically from the first iterate, and the projection stalls at
    h ~ 4.2e-3 regardless of the iteration budget.  Reporting both makes the difference
    between "the projection is a small correction" and "the projection did not converge"
    an explicit, checkable statement.
    """
    from scipy.linalg import expm
    from scipy.optimize import minimize
    d = W_target.shape[0]
    w = W_target.flatten().copy()
    rho, alpha = rho0, 0.0
    tol = 1e-8
    for _i in range(max_iter):
        def lag(wv):
            W = wv.reshape(d, d)
            return (0.5 * np.sum((W - W_target) ** 2) + lam1 * np.sum(np.abs(W))
                    + 0.5 * rho * (np.trace(expm(W * W)) - d) ** 2
                    + alpha * (np.trace(expm(W * W)) - d))

        def grad(wv):
            W = wv.reshape(d, d)
            hv = np.trace(expm(W * W)) - d
            dh = 2 * W * expm(W * W).T
            return ((W - W_target) + lam1 * np.sign(W)).flatten() \
                + rho * hv * dh.flatten() + alpha * dh.flatten()

        res = minimize(lag, w, method='L-BFGS-B', jac=grad,
                       options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-14})
        w = res.x
        hv = _h(w.reshape(d, d))
        if abs(hv) < tol:
            break
        alpha += rho * hv
        rho = min(rho * gamma, 1e8)
    return w.reshape(d, d), abs(hv)


def project_to_dag(W_target, lam1=0.001, max_iter=100, rho0=0.05, gamma=2.0):
    """Copied verbatim from scripts/experiments/_synthetic_v6.py so that the projection
    applied to TCGA is the same operator that the synthetic benchmark validates."""
    from scipy.linalg import expm
    from scipy.optimize import minimize
    d = W_target.shape[0]
    w = W_target.flatten().copy()
    rho, alpha = rho0, 0.0
    prev_h = abs(_h(w.reshape(d, d)))
    for _i in range(max_iter):
        def lag(wv):
            W = wv.reshape(d, d)
            return (0.5 * np.sum((W - W_target) ** 2) + lam1 * np.sum(np.abs(W))
                    + 0.5 * rho * (np.trace(expm(W * W)) - d) ** 2
                    + alpha * (np.trace(expm(W * W)) - d))

        def grad(wv):
            W = wv.reshape(d, d)
            hv = np.trace(expm(W * W)) - d
            dh = 2 * W * expm(W * W).T
            return ((W - W_target) + lam1 * np.sign(W)).flatten() \
                + rho * hv * dh.flatten() + alpha * dh.flatten()

        res = minimize(lag, w, method='L-BFGS-B', jac=grad,
                       options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-14})
        w = res.x
        hv = _h(w.reshape(d, d))
        alpha += rho * hv
        if abs(hv) > abs(prev_h) and abs(hv) > 1e-10:
            rho = min(rho * gamma, 1e8)
        prev_h = hv
        if abs(hv) < 1e-8:
            break
    return w.reshape(d, d), abs(hv)


def stage_stage2():
    """Apply the two-stage decomposition to the real TCGA aligned panel."""
    meta = L.load_panel()
    panel, cancers = meta['panel'], meta['cancers']
    fits, cs = L.load_fits('_shared_panel_notears.json')
    Ws = {c: np.array(fits[c]['W'], dtype=float) for c in cs}
    d = len(panel)

    M = np.stack([Ws[c] for c in cs], axis=0)
    Wbar = np.median(M, axis=0)
    np.fill_diagonal(Wbar, 0.0)
    h_bar = _h(Wbar)

    t0 = time.time()
    W0, h0 = project_to_dag(Wbar, lam1=0.001)
    log('projection (conservative schedule) in %.0fs -> h = %.4e' % (time.time() - t0, h0))

    t0 = time.time()
    W0al, h0al = project_to_dag_al(Wbar, lam1=0.001, max_iter=200)
    log('projection (standard AL schedule) in %.0fs -> h = %.4e' % (time.time() - t0, h0al))

    # how acyclic is the shared median compared with the individual cohort fits?
    coh_h = {c: _h(np.array(fits[c]['W'], dtype=float)) for c in cs}
    n_below = int(sum(1 for c in cs if coh_h[c] > h_bar))

    rec = L.recurrence_list(L.recurrence(np.abs(M), TAU), panel, 10)
    rec_pairs = [(panel.index(r['src']), panel.index(r['dst'])) for r in rec]

    same_positions_al = bool(np.array_equal(np.abs(W0al) > TAU, np.abs(W0) > TAU))
    out = dict(
        d=d, n_cohorts=len(cs), tau=TAU, lam_proj=0.001,
        h_median_before=float(h_bar), h_shared_after=float(h0),
        h_shared_after_standard_AL=float(h0al),
        edges_projected_standard_AL=int((np.abs(W0al) > TAU).sum()),
        projected_edge_positions_identical=same_positions_al,
        h_per_cohort={"min": float(min(coh_h.values())),
                      "median": float(np.median(list(coh_h.values()))),
                      "max": float(max(coh_h.values()))},
        n_cohorts_with_larger_h_than_shared_median=n_below,
        median_edges_unprojected=int((np.abs(Wbar) > TAU).sum()),
        shared_edges_projected=int((np.abs(W0) > TAU).sum()),
        recurring_14=int(len(rec_pairs)),
        recurring_among_shared_projected=int(sum(
            1 for (i, j) in rec_pairs if abs(W0[i, j]) > TAU)),
        recurring_among_median_unprojected=int(sum(
            1 for (i, j) in rec_pairs if abs(Wbar[i, j]) > TAU)),
        max_abs_median=float(np.abs(Wbar).max()),
        max_abs_shared=float(np.abs(W0).max()),
    )

    # cohort deviations
    dev = {}
    shared_pos = np.abs(W0) > TAU
    for c in cs:
        D = Ws[c] - W0
        D[np.abs(D) < 0.01] = 0.0
        np.fill_diagonal(D, 0.0)
        raw = Ws[c]
        n_edges = int((np.abs(raw) > TAU).sum())
        n_shared = int(((np.abs(raw) > TAU) & shared_pos).sum())
        dev[c] = dict(n=int(fits[c]['n']),
                      edges_shared_backbone=int(shared_pos.sum()),
                      edges_cohort=n_edges,
                      cohort_edges_also_in_backbone=n_shared,
                      cohort_edges_cohort_specific=n_edges - n_shared,
                      frac_cohort_specific=((n_edges - n_shared) / n_edges
                                            if n_edges else None),
                      delta_nonzero_at_001=int((D != 0).sum()),
                      delta_fro=float(np.linalg.norm(D)),
                      delta_max=float(np.abs(D).max()),
                      delta_above_tau=int((np.abs(D) > TAU).sum()))
    out['per_cohort'] = dev
    fro = np.array([dev[c]['delta_fro'] for c in cs])
    cs_frac = np.array([dev[c]['frac_cohort_specific'] for c in cs
                        if dev[c]['frac_cohort_specific'] is not None])
    out['delta_summary'] = dict(
        fro_mean=float(fro.mean()), fro_min=float(fro.min()), fro_max=float(fro.max()),
        cohort_specific_edges_mean=float(np.mean(
            [dev[c]['cohort_edges_cohort_specific'] for c in cs])),
        cohort_specific_frac_mean=float(cs_frac.mean()),
        cohort_specific_frac_min=float(cs_frac.min()),
        cohort_specific_frac_max=float(cs_frac.max()),
        note=('Delta_k at the 0.01 soft-threshold is dense (~5.8k entries), because NOTEARS '
              'returns many small coefficients below the edge threshold; the interpretable '
              'quantity is the share of a cohort\'s tau-level edges that the shared backbone '
              'does not contain, which is reported as cohort_specific_frac.'))
    L.save(out, 'two_stage_tcga.json')
    log('stage2 delta: cohort-specific share of tau edges = %.2f [%.2f, %.2f]'
        % (out['delta_summary']['cohort_specific_frac_mean'],
           out['delta_summary']['cohort_specific_frac_min'],
           out['delta_summary']['cohort_specific_frac_max']))
    log('stage2: h(median)=%.3e ; h(W0) conservative=%.3e  standard-AL=%.3e ; '
        'edges %d(median) -> %d(projected) -> %d(std-AL), positions identical=%s ; '
        'recurring in shared = %d/%d ; shared median is more acyclic than %d/33 cohorts'
        % (out['h_median_before'], out['h_shared_after'],
           out['h_shared_after_standard_AL'], out['median_edges_unprojected'],
           out['shared_edges_projected'], out['edges_projected_standard_AL'],
           out['projected_edge_positions_identical'],
           out['recurring_among_shared_projected'], out['recurring_14'],
           out['n_cohorts_with_larger_h_than_shared_median']))


# ================================================================= survival (A-M10/B4)
def _bh(pvals):
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n, dtype=float)
    prev = 1.0
    for rank, idx in enumerate(order[::-1]):
        r = n - rank
        val = min(prev, p[idx] * n / r)
        q[idx] = val
        prev = val
    return q


def stage_survival():
    import pandas as pd
    from lifelines import CoxPHFitter
    from lifelines.statistics import proportional_hazard_test

    km = load_json(os.path.join(L.RES, '_km_pancan.json'))
    nomin = []
    for c in sorted(km):
        hub = km[c].get('hub')
        g = km[c]['genes'].get(hub) if hub else None
        if g and g['logrank_p'] < 0.05:
            nomin.append(dict(cohort=c, gene=hub, n_low=g['n_low'], n_high=g['n_high'],
                              logrank_p=float(g['logrank_p']), hr_split=float(g['hr']),
                              hr_lo=float(g['hr_lo']), hr_hi=float(g['hr_hi'])))
    log('nominated hubs (log-rank p<0.05): %d' % len(nomin))

    order = sorted(L.all_cancers())
    rows = []
    for n in nomin:
        c, g = n['cohort'], n['gene']
        osf = os.path.join(VAL, 'pancan_os', '%s_os.json' % c)
        if not os.path.exists(osf):
            continue
        os_rows = load_json(osf)
        surv = {r['patient']: r for r in os_rows}
        # read the gene's row directly from the index so the sample ids stay aligned with
        # the survival table (L.expr returns a bare matrix and drops them)
        path = os.path.join(L.TCGA_DIR, 'TCGA_%s_HiSeqV2.tsv' % c)
        df = pd.read_csv(path, sep='\t', index_col=0)
        # the source matrix carries some of these loci under legacy spellings (SHOC1 is
        # stored as C9orf84), so resolve through the shared symbol table instead of a
        # literal lookup -- a straight df.loc[g] silently dropped one of the six hubs
        lab = _gene_symbols.labels(df.index)
        key = lab.get(_gene_symbols.canonical(g).upper())
        if key is None:
            log('  %-5s %-9s not found in the expression matrix -- skipped' % (c, g))
            continue
        series = df.loc[key]
        # Expression columns are sample barcodes (TCGA-OR-A5LC-01); the survival tables are
        # keyed on the patient id.  Every sample of a patient is averaged and no sample-type
        # filter is applied: this is the convention the KM scan and the cohort tables use, so
        # the continuous model rests on the same patient set as the split it is compared with.
        # An earlier draft kept only '-01' primary tumours, which for SKCM -- a matrix that is
        # mostly metastatic -- dropped 474 samples to 104 and 426 patients to 76, so that row
        # was fitted on a different subset from the KM curve printed beside it.
        agg = {}
        for k, v in series.items():
            k = str(k)
            kp = k.split('-')
            agg.setdefault('-'.join(kp[:3]) if len(kp) >= 3 else k, []).append(float(v))
        vals = {p: float(np.mean(vv)) for p, vv in agg.items()}
        pats = [p for p in vals if p in surv]
        if len(pats) < 30:
            continue
        e = np.array([vals[p] for p in pats], float)
        t = np.array([float(surv[p]['os_months']) for p in pats], float)
        ev = np.array([1 if str(surv[p]['os_status']).startswith('1') else 0 for p in pats], int)
        keep = np.isfinite(e) & np.isfinite(t) & (t > 0)
        e, t, ev = e[keep], t[keep], ev[keep]
        sd = e.std() if e.std() > 0 else 1.0
        cph = CoxPHFitter()
        dfo = pd.DataFrame({'E': (e - e.mean()) / sd, 'T': t, 'event': ev})
        cph.fit(dfo, duration_col='T', event_col='event')
        hr = float(np.exp(cph.params_['E']))
        ci = np.exp(cph.confidence_intervals_.loc['E'].values)
        ph = proportional_hazard_test(cph, dfo, time_transform='rank')
        rows.append(dict(cohort=c, gene=g, n=int(len(e)), events=int(ev.sum()),
                         hr_per_sd=hr, ci_lo=float(ci[0]), ci_hi=float(ci[1]),
                         p_per_sd=float(cph.summary.loc['E', 'p']),
                         ph_p=float(ph.summary.loc['E', 'p']),
                         logrank_p_split=float(n['logrank_p']), hr_split=float(n['hr_split'])))
        log('  %-5s %-9s n=%4d HR/SD=%.2f [%.2f,%.2f] p=%.3g  PH p=%.2f'
            % (c, g, len(e), hr, ci[0], ci[1], rows[-1]['p_per_sd'], rows[-1]['ph_p']))

    if rows:
        q = _bh([r['p_per_sd'] for r in rows])
        for r, qv in zip(rows, q):
            r['q_bh'] = float(qv)
            r['significant_fdr'] = bool(qv < 0.05)

    ks = np.array([r['n'] for r in rows], float) if rows else np.array([])
    thr = {}
    for cut in (172, 180, 200):
        sel = [r for r in rows if r['n'] >= cut]
        thr['n_ge_%d' % cut] = dict(n_hubs=len(sel),
                                    genes=sorted(r['gene'] for r in sel))

    out = dict(rows=rows, floor_sensitivity=thr,
               n_nominated=len(rows),
               n_fdr_significant=int(sum(1 for r in rows if r.get('significant_fdr'))),
               n_ph_violations=int(sum(1 for r in rows if r['ph_p'] < 0.05)),
               note=('Per-SD continuous Cox replaces the median split; the split-based '
                     'log-rank p and hazard ratio are given alongside for continuity. '
                     'q is Benjamini-Hochberg across the nominated hubs.'))
    L.save(out, 'survival_perSD.json')
    log('survival: %d hubs, FDR-significant %d, PH violations %d'
        % (out['n_nominated'], out['n_fdr_significant'], out['n_ph_violations']))


STAGES = dict(trrust=stage_trrust, stage2=stage_stage2, survival=stage_survival)

if __name__ == '__main__':
    want = [s.strip() for s in (sys.argv[1] if len(sys.argv) > 1 else 'all').split(',')]
    if want == ['all']:
        want = list(STAGES)
    for s in want:
        if s not in STAGES:
            print('!! unknown stage', s)
            continue
        t0 = time.time()
        print('=' * 70)
        print('stage', s, time.strftime('%H:%M:%S'), flush=True)
        try:
            STAGES[s]()
            print('-- %s ok (%.1fs)' % (s, time.time() - t0), flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print('!! %s FAILED: %s' % (s, e), flush=True)
