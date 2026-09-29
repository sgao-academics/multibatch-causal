# -*- coding: utf-8 -*-
"""Overnight revision compute batch for MultiBatch (Doubao review round).

Stage order is P0-first, so a partial night still delivers the P0 evidence.

  prep        gene stats (mean expression + mean MAD percentile over 33 cohorts)  ~10 min
  desk        NON-REFIT: tau grid from cached W, h-stratification, jackknife,
              per-cohort Marchenko-Pastur, release lists                          ~3 min
  composition P0  composition-adjusted refit (reviewer B1)                        ~25 min
  randompanel P0  8 dispersion/expression-matched random panels (reviewer A-M2)   ~3 h
  bootstrap   P0  6 sample-bootstraps per cohort -> orientation retention (A-M1)  ~2.2 h
  lamscan     P1  refit at lambda = 0.005 / 0.02 -> tau x lambda grid (A-M3)      ~45 min
  lowmad      P1  low-dispersion matched panel (A-M4)                             ~25 min
  dscan       P1  aligned panels of width 50/75/125/150 (A-M6)                    ~1.5 h
  rotation    P1  covariance-preserving rotation null (A-M2b)                     ~1.2 h
  multiinit   P1  6 random inits on 20 cohorts (A-M1b)                            ~1.3 h

usage:  python _rev_night.py all
        python _rev_night.py desk,composition
"""
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _rev_lib as L

L.burst(1)          # 1-thread BLAS is faster per fit *and* reproduces the cached W exactly
                    # (see _rev_unit.py / _rev_unit1.py); cohorts are parallelised instead

MANIFEST = os.path.join(L.REV, '_manifest.json')
ALL_STAGES = ['prep', 'desk', 'composition', 'randompanel', 'bootstrap',
              'lamscan', 'lowmad', 'dscan', 'rotation', 'multiinit']

# REV_LIMIT=2 runs every stage on the first two cohorts -- the smoke test that must
# pass before the real run is allowed to start.
LIMIT = int(os.environ.get('REV_LIMIT', '0')) or None


def cap(seq):
    return list(seq)[:LIMIT] if LIMIT else list(seq)


def manifest():
    return L.load_rev('_manifest.json') or {}


def mark(stage, ok, note=''):
    m = manifest()
    m[stage] = dict(ok=bool(ok), at=time.strftime('%Y-%m-%d %H:%M:%S'), note=note)
    L.save(m, '_manifest.json')


def save_npz(name, **arrays):
    np.savez_compressed(os.path.join(L.REV, name), **arrays)


def _fit_one(job):
    """One cohort, one refit.  Module-level so it can be shipped to a worker process.

    job = (cohort, genes, lam, max_outer, Qflat, d); Qflat is a flattened rotation for
    the covariance-preserving null (A-M2b), else None.
    """
    c, genes, lam, max_outer, Qflat, d = job
    X, _miss = L.expr(c, genes)
    if Qflat is not None:
        X = X @ np.asarray(Qflat, dtype=np.float64).reshape(d, d)
    W, h = L.notears_lbfgs(X, lam=lam, max_outer=max_outer)
    return c, W, dict(n=int(X.shape[0]), edges=int((np.abs(W) > L.TAU).sum()),
                      h=float(h), max_abs=float(np.abs(W).max()))


def fit_pass(panel_genes, lam=0.01, max_outer=100, tag='', transform=None,
             ckpt_name=None, cohorts=None, subset_fn=None, Q=None):
    """Fit every cohort.  Returns dict cohort -> dict(W, edges, h, n).

    Resumable: ckpt_name names a json listing the cohorts already finished; W is kept in
    a companion npz so a resumed run does not refit.  Cohorts are fitted concurrently
    across processes (REV_WORKERS); results are still checkpointed as they land.
    """
    cancers = cohorts or L.all_cancers()
    if subset_fn is not None:
        cancers = [c for c in cancers if subset_fn(c)]
    cancers = cap(cancers)

    done, Ws = [], {}
    ck = os.path.join(L.REV, (ckpt_name or tag) + '.ckpt.json')
    npz = os.path.join(L.REV, (ckpt_name or tag) + '_W.npz')
    if os.path.exists(ck):
        done = json_load(ck)
    if done and os.path.exists(npz):
        z = np.load(npz, allow_pickle=True)
        for c in done:
            if c in z:
                Ws[c] = z[c]

    # per-cohort summary is persisted too, so a resumed run still returns a complete store
    store = L.load_rev((ckpt_name or tag) + '.json') or {}
    todo = [c for c in cancers if c not in done]
    genes = tuple(panel_genes)
    d = len(genes)
    Qflat = None if Q is None else np.asarray(Q, dtype=np.float64).flatten().tolist()
    L.log('%s: %d/%d cohorts done, %d to fit (lam=%s, workers=%d)'
          % (tag, len(done), len(cancers), len(todo), lam, L.WORKERS))
    t_all = time.time()
    n_done = 0
    jobs = [(c, genes, lam, max_outer, Qflat, d) for c in todo]
    for c, W, summary in L.pmap(_fit_one, jobs):
        n_done += 1
        Ws[c] = W
        store[c] = summary
        with open(ck, 'w', encoding='utf-8') as f:
            import json
            json.dump(sorted(Ws), f)
        save_npz((ckpt_name or tag) + '_W.npz', **{k: v.astype(np.float32) for k, v in Ws.items()})
        L.save(store, (ckpt_name or tag) + '.json')
        L.log('  %s [%d/%d] %-5s n=%4d edges=%3d h=%.2e  (%.0fs elapsed)'
              % (tag, n_done, len(todo), c, summary['n'], summary['edges'],
                 summary['h'], time.time() - t_all))
    return {c: store[c] for c in sorted(store)}, Ws


def json_load(p):
    import io
    import json
    return json.load(io.open(p, encoding='utf-8'))


# =============================================================== prep
def stage_prep():
    """per-gene mean expression and mean within-cohort MAD percentile over all 33 cohorts."""
    out = L.load_rev('_gene_stats.json')
    if out and out.get('ok'):
        L.log('prep: cached (%d genes)' % len(out['genes']))
        return
    import pandas as pd
    from scipy.stats import median_abs_deviation
    files = sorted(f for f in os.listdir(L.TCGA_DIR)
                   if f.startswith('TCGA_') and f.endswith('_HiSeqV2.tsv'))
    per_mad, per_mean = {}, {}
    for i, f in enumerate(files):
        c = f.replace('TCGA_', '').replace('_HiSeqV2.tsv', '')
        df = pd.read_csv(os.path.join(L.TCGA_DIR, f), sep='\t', index_col=0).T
        v = df.values.astype(np.float64)
        per_mad[c] = pd.Series(median_abs_deviation(v, axis=0), index=df.columns)
        per_mean[c] = pd.Series(v.mean(axis=0), index=df.columns)
        L.log('  prep [%2d/%d] %-5s %d x %d' % (i + 1, len(files), c, df.shape[0], df.shape[1]))
    common = sorted(set.intersection(*(set(per_mad[c].index) for c in per_mad)))
    rank = pd.DataFrame(index=common)
    for c in per_mad:
        rank[c] = per_mad[c].rank(pct=True).reindex(common)
    mad_pct = rank.mean(axis=1)
    mrank = pd.DataFrame(index=common)
    for c in per_mean:
        mrank[c] = per_mean[c].rank(pct=True).reindex(common)
    expr_pct = mrank.mean(axis=1)
    L.save(dict(ok=True, genes=list(common),
                mad_pct=[round(float(x), 6) for x in mad_pct.reindex(common).values],
                expr_pct=[round(float(x), 6) for x in expr_pct.reindex(common).values]),
           '_gene_stats.json')
    L.log('prep: %d common genes cached' % len(common))


# =============================================================== desk (no refits)
def stage_desk():
    out = {}
    meta = L.load_panel()
    panel, cancers = meta['panel'], meta['cancers']
    fits, cs = L.load_fits('_shared_panel_notears.json')
    M, med = L.median_matrix(fits, cs, dref=len(panel))
    present = L.recurrence(M, L.TAU)

    # ---- tau grid off the cached W (lambda = 0.01)
    tau_grid = {}
    for tau in (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45):
        pr = L.recurrence(M, tau)
        med_edges = int((med > tau).sum())
        rec = L.recurrence_list(pr, panel, 10)
        tau_grid['%.2f' % tau] = dict(
            median_edges=med_edges, recurring_ge10=len(rec),
            q=L.qstat(pr), nk=L.nk_counts(pr),
            pairs=['%s->%s' % (r['src'], r['dst']) for r in rec])
        L.log('  tau=%.2f  median edges=%d  recurring>=10 = %d  Q=%.0f'
              % (tau, med_edges, len(rec), L.qstat(pr)))
    out['tau_grid'] = tau_grid

    # ---- jackknife: recurrence stability when each cohort is dropped
    jack = []
    base = set('%s->%s' % (r['src'], r['dst'])
               for r in L.recurrence_list(present, panel, 10))
    for drop in cs:
        keep = [c for c in cs if c != drop]
        Mk, _ = L.median_matrix(fits, keep, dref=len(panel))
        pr = L.recurrence(Mk, L.TAU)
        s = set('%s->%s' % (r['src'], r['dst'])
                for r in L.recurrence_list(pr, panel, 10))
        jack.append(dict(drop=drop, n10=len(s), lost=sorted(base - s), gained=sorted(s - base)))
    out['jackknife'] = dict(base_n10=len(base), rows=jack,
                            worst_lost=max((len(r['lost']) for r in jack), default=0))
    L.log('  jackknife: base recurring=%d, max lost on any single drop=%d'
          % (len(base), out['jackknife']['worst_lost']))

    # ---- h-stratification of the 14 recurring edges
    acy = json_load(os.path.join(L.RES, '_acyclicity.json'))
    h_by = {c: acy['per_cohort'][c]['h_abs'] for c in acy['per_cohort']}
    cyc = set(acy['recurrent_pairs_on_a_cycle_somewhere'])
    rows = []
    for r in L.recurrence_list(present, panel, 10):
        key = '%s->%s' % (r['src'], r['dst'])
        i, j = panel.index(r['src']), panel.index(r['dst'])
        sup = [dict(c=b, h=round(h_by.get(b, float('nan')), 5),
                    on_cycle=bool(acy['per_cohort'].get(b, {}).get('acyclic') is False))
               for b in np.array(cs)[M[:, i, j] > L.TAU]]
        rows.append(dict(pair=key, n=r['n'],
                         n_h_lt_1e2=sum(1 for s in sup if s['h'] < 1e-2),
                         n_h_gt_1e1=sum(1 for s in sup if s['h'] > 1e-1),
                         on_cycle_in=key in cyc, support=sup))
    out['h_stratification'] = rows
    out['cycle_note'] = dict(
        sex_chrom_pairs_on_a_cycle=len(cyc),
        recurring_14_on_a_cycle=sorted(
            '%s->%s' % (r['src'], r['dst']) for r in L.recurrence_list(present, panel, 10)
            if '%s->%s' % (r['src'], r['dst']) in cyc))
    L.log('  h-strat done; 14 edges on a cycle somewhere = %d'
          % len(out['cycle_note']['recurring_14_on_a_cycle']))

    # ---- per-cohort Marchenko-Pastur
    mp = []
    for c in cs:
        X, _ = L.expr(c, panel)
        n, d = X.shape
        ev = np.linalg.eigvalsh(np.corrcoef(X, rowvar=False))
        lam_plus = (1 + np.sqrt(float(d) / n)) ** 2
        mp.append(dict(cohort=c, n=int(n), lam1=float(ev[-1]), lam_plus=float(lam_plus),
                       ratio=float(ev[-1] / lam_plus), n_lt_d=bool(n < d),
                       n_eig_above=float((ev > lam_plus).sum())))
    out['mp_per_cohort'] = mp
    L.log('  per-cohort MP done')

    # ---- release lists (reviewer A-M9 / B9)
    aa = json_load(os.path.join(L.RES, '_axis_attribution.json'))
    ds = json_load(os.path.join(L.RES, '_driver_screen.json'))
    pn = json_load(os.path.join(L.RES, '_permutation_null.json'))
    out['release_lists'] = dict(
        external_nonmal_markers=aa['purity_attribution']['external_nonmal_markers'],
        external_tumor_markers=aa['purity_attribution']['external_tumor_markers'],
        canonical_drivers=sorted(ds['driver_percentiles']),
        n_canonical_drivers=len(ds['driver_percentiles']),
        permutation_counts=dict(discrete=pn.get('permutations'),
                                continuous=pn['continuous']['permutations'],
                                seed=pn['continuous']['seed']),
        panel2_note='seed and sampling rule to be read from scripts/analysis/build_panel_B.py')
    L.save(out, 'desk.json')
    L.log('desk: done')


# =============================================================== composition (B1)
def _read_cols(cancer, cols):
    """samples x cols for the requested genes.

    The Xena TSV has genes as *rows* and samples as columns, so there is no way to ask
    pandas for a gene subset without transposing first; the big frame is dropped as soon
    as the small one is taken, which is what keeps several concurrent workers affordable.
    """
    import pandas as pd
    path = os.path.join(L.TCGA_DIR, 'TCGA_%s_HiSeqV2.tsv' % cancer)
    df = pd.read_csv(path, sep='\t', index_col=0).T
    out = df.reindex(columns=list(cols))
    del df
    return out


def _composition_one(job):
    """Residualise the aligned panel on an external non-malignant marker score, refit.

    job = (cohort, markers, panel_genes, pairs); returns (per-cohort summary, W).
    """
    from scipy import stats as st
    c, markers, panel, pairs = job
    df = _read_cols(c, list(markers) + list(panel))

    def zs(a):
        a = np.nan_to_num(np.asarray(a, dtype=np.float64), nan=0.0)
        return (a - a.mean(axis=0)) / (a.std(axis=0) + 1e-12)

    Xm = zs(df.reindex(columns=list(markers)).values)
    # composition score: mean of the standardised marker panel (one value per sample)
    C = Xm.mean(axis=1)
    Cz = (C - C.mean()) / (C.std() + 1e-12)

    X = zs(df.reindex(columns=list(panel)).values)
    # residualise every panel gene on the composition score
    Xr = X - np.outer(Cz, (Cz @ X) / (Cz @ Cz))

    pr = []
    for i2, j2 in pairs:
        a, b = X[:, i2], X[:, j2]
        ar, br = Xr[:, i2], Xr[:, j2]
        pr.append(dict(i=int(i2), j=int(j2),
                       raw=float(st.pearsonr(a, b)[0]),
                       partial=float(st.pearsonr(ar, br)[0]),
                       raw_rho=float(st.spearmanr(a, b)[0]),
                       partial_rho=float(st.spearmanr(ar, br)[0])))
    # only the residualised fit is new; the raw fit is the cached one the paper reports
    W, h = L.notears_lbfgs(Xr, lam=0.01, max_outer=100)
    summary = dict(n=int(X.shape[0]), edges_res=int((np.abs(W) > L.TAU).sum()),
                   h=float(h), pairs=pr,
                   composition_r=float(np.corrcoef(Cz, X[:, pairs[0][0]])[0, 1])
                   if pairs else None)
    return c, (summary, W)


def stage_composition():
    meta = L.load_panel()
    panel, cancers = meta['panel'], meta['cancers']
    aa = json_load(os.path.join(L.RES, '_axis_attribution.json'))
    markers = aa['purity_attribution']['external_nonmal_markers']
    fits, cs = L.load_fits('_shared_panel_notears.json')
    # the pair list is the paper's 14 recurring pairs, taken from the FULL 33-cohort fit;
    # only the refitting loop is capped by REV_LIMIT.
    M, _ = L.median_matrix(fits, cs, dref=len(panel))
    present = L.recurrence(M, L.TAU)
    pairs = [(panel.index(r['src']), panel.index(r['dst']))
             for r in L.recurrence_list(present, panel, 10)]

    ck = os.path.join(L.REV, 'composition.ckpt.json')
    done = json_load(ck) if os.path.exists(ck) else []
    rows = L.load_rev('composition.json') or {}
    per, Wr = rows.get('per_cohort', {}), {}
    npz = os.path.join(L.REV, 'composition_W.npz')
    if os.path.exists(npz):
        z = np.load(npz, allow_pickle=True)
        for c in z:
            Wr[c] = z[c]

    todo = [x for x in cap(cs) if x not in done]
    jobs = [(c, tuple(markers), tuple(panel), tuple(pairs)) for c in todo]
    t_all = time.time()
    for i, (c, (per_c, W)) in enumerate(L.pmap(_composition_one, jobs)):
        Wr[c] = W
        per[c] = per_c
        rows['per_cohort'] = per
        L.save(rows, 'composition.json')
        save_npz('composition_W.npz', **{k: v.astype(np.float32) for k, v in Wr.items()})
        with open(ck, 'w', encoding='utf-8') as f:
            import json
            json.dump(sorted(per), f)
        L.log('  composition [%d/%d] %-5s edges_res=%d h=%.2e  (%.0fs elapsed)'
              % (i + 1, len(todo), c, per_c['edges_res'], per_c['h'], time.time() - t_all))

    # ---- summarise
    if len(Wr) >= len(cap(cs)):
        cc = sorted(Wr)
        Mr = np.stack([np.abs(Wr[c]) for c in cc], axis=0)
        med_r = np.median(Mr, axis=0)
        np.fill_diagonal(med_r, 0.0)
        pr_r = L.recurrence(Mr, L.TAU)
        summary = dict(
            median_edges_residual=int((med_r > L.TAU).sum()),
            max_abs_median_residual=float(med_r.max()),
            recurring_ge10_residual=len(L.recurrence_list(pr_r, panel, 10)),
            surviving_pairs=['%s->%s' % (r['src'], r['dst'])
                             for r in L.recurrence_list(pr_r, panel, 10)],
            pair_table=[])
        for (i2, j2) in pairs:
            raws = [per[c]['pairs'][k]['raw'] for c in cc for k in range(len(pairs))
                    if per[c]['pairs'][k]['i'] == i2 and per[c]['pairs'][k]['j'] == j2]
            pars = [per[c]['pairs'][k]['partial'] for c in cc for k in range(len(pairs))
                    if per[c]['pairs'][k]['i'] == i2 and per[c]['pairs'][k]['j'] == j2]
            summary['pair_table'].append(dict(
                pair='%s->%s' % (panel[i2], panel[j2]),
                mean_raw=float(np.mean(raws)), mean_partial=float(np.mean(pars)),
                n_raw_sig=int(sum(1 for x in raws if abs(x) > L.TAU)),
                n_partial_sig=int(sum(1 for x in pars if abs(x) > L.TAU)),
                n_cohorts_res_edge=int((pr_r[i2, j2] > 0) + (pr_r[j2, i2] > 0))))
        rows['summary'] = summary
        L.save(rows, 'composition.json')
        L.log('  composition SUMMARY: residual median edges=%d (was 6); recurring>=10 = %d (was 14)'
              % (summary['median_edges_residual'], summary['recurring_ge10_residual']))


# =============================================================== randompanel (A-M2)
def _deciles(stats, keys=('expr_pct', 'mad_pct')):
    """Decile bucket of each gene on the requested statistics, 1..11."""
    bins = np.linspace(0, 1, 11)
    out = []
    for k in keys:
        v = np.array(stats[k])
        out.append(dict(zip(stats['genes'], np.clip(np.digitize(v, bins), 1, 11).tolist())))
    return out


def _matched_sample(stats, panel, rng, key_of, restrict=None):
    """One stand-in gene per panel gene, matched on key_of, drawn *without replacement*.

    The first version of this function sampled with replacement, which quietly duplicated
    genes inside a panel.  A repeated column is the same variable entered twice, so the
    estimator puts a large weight on that pair in every cohort, and the pair then appears
    as a 'recurring' edge (GENE -> GENE) that reflects the panel rather than the data --
    13 to 21 of the 100 genes were duplicated in the dispersion-matched panels and the
    self-pairs they produced were the top recurring pairs of several of them.  Every draw
    here is unique; a key whose pool is used up falls through to the nearest key that
    still has a free gene.
    """
    genes = list(stats['genes'])
    real = set(panel)
    keys = {g: key_of(g) for g in genes}
    free = {}
    for g in genes:
        if g not in real and (restrict is None or g in restrict):
            free.setdefault(keys[g], []).append(g)
    all_keys = sorted(free)
    pick = []
    for g in panel:
        if g not in keys:
            return None
        k0 = keys[g]
        order = sorted(all_keys, key=lambda k: (sum(abs(a - b) for a, b in zip(k, k0)), k))
        for k in order:
            if free[k]:
                pick.append(free[k].pop(int(rng.integers(len(free[k])))))
                break
        else:
            return None
    if len(set(pick)) != len(pick):
        return None
    return pick


def _matched_panel(stats, panel, rng):
    """100 genes matched to the real panel by mean-expression decile (A-M2a)."""
    (eidx,) = _deciles(stats, ('expr_pct',))
    return _matched_sample(stats, panel, rng, lambda g: (eidx[g],))


def _matched_panel_dm(stats, panel, rng):
    """100 genes matched on BOTH mean-expression and dispersion decile (A-M2b).

    The reviewer's null has to keep the selection regime, not only the expression level:
    matching on expression alone yields panels sitting at the ~70th MAD percentile, which
    is a weaker selection than the real panel's top-100 cut and is not the comparison the
    claim needs.  Here each real panel gene is replaced by a random gene in the same
    expression decile *and* the same MAD decile.
    """
    eidx, midx = _deciles(stats, ('expr_pct', 'mad_pct'))
    return _matched_sample(stats, panel, rng, lambda g: (eidx[g], midx[g]))


def _panel_stats(stats, genes, store, Wpath):
    """Everything the null comparison needs from one fitted random panel."""
    cs = sorted(store)
    z = np.load(Wpath, allow_pickle=True)
    Mr = np.stack([np.abs(z[c]).astype(float) for c in cs], axis=0)
    med = np.median(Mr, axis=0)
    np.fill_diagonal(med, 0.0)
    pr = L.recurrence(Mr, L.TAU)
    rec = L.recurrence_list(pr, genes, 10)
    gidx = {g: k for k, g in enumerate(stats['genes'])}
    madv = np.array(stats['mad_pct'])
    expv = np.array(stats['expr_pct'])
    return dict(median_edges=int((med > L.TAU).sum()), max_abs_median=float(med.max()),
                recurring_ge10=len(rec),
                pairs=['%s->%s(%d)' % (r['src'], r['dst'], r['n']) for r in rec],
                n_distinct_pairs=len(set(tuple(sorted((r['src'], r['dst']))) for r in rec)),
                q=L.qstat(pr), nk=L.nk_counts(pr),
                total_edges=int(sum(store[c]['edges'] for c in cs)),
                n_unique_genes=len(set(genes)),
                median_mad_pct=float(np.median([madv[gidx[g]] for g in genes])),
                median_expr_pct=float(np.median([expv[gidx[g]] for g in genes])))


def stage_randompanel_dm(n_panels=12, seed0=2001):
    """Second null family: dispersion- AND expression-matched panels (reviewer A-M2b)."""
    meta = L.load_panel()
    panel = meta['panel']
    if LIMIT:
        n_panels = 1
    stats = L.load_rev('_gene_stats.json')
    if not stats:
        L.log('randompanel_dm: need prep first -- skipping')
        return
    rows = L.load_rev('randompanel_dm.json') or {}
    done = rows.get('rows', [])
    for p in range(len(done), n_panels):
        rng = np.random.default_rng(seed0 + p)
        genes = _matched_panel_dm(stats, panel, rng)
        if genes is None:
            L.log('randompanel_dm: could not build a unique panel %d -- skipping' % p)
            continue
        tag = 'dp%d' % p
        t0 = time.time()
        store, _ = fit_pass(genes, tag=tag, ckpt_name=tag)
        row = _panel_stats(stats, genes, store, os.path.join(L.REV, tag + '_W.npz'))
        row.update(panel=p, seed=seed0 + p, genes=list(genes))
        done.append(row)
        rows['rows'] = done
        L.save(rows, 'randompanel_dm.json')
        L.log('  randompanel_dm %d/%d: median_edges=%d max|med|=%.3f recurring>=10=%d '
              'unique=%d (%.0fs)'
              % (p + 1, n_panels, row['median_edges'], row['max_abs_median'],
                 row['recurring_ge10'], row['n_unique_genes'], time.time() - t0))
    L.save(rows, 'randompanel_dm.json')


def stage_randompanel(n_panels=12, seed0=1001):
    """First null family: panels matched on mean-expression decile only (A-M2a)."""
    meta = L.load_panel()
    panel = meta['panel']
    if LIMIT:
        n_panels = 1
    stats = L.load_rev('_gene_stats.json')
    if not stats:
        L.log('randompanel: need prep first -- skipping')
        return
    rows = L.load_rev('randompanel.json') or {}
    done = rows.get('rows', [])
    for p in range(len(done), n_panels):
        rng = np.random.default_rng(seed0 + p)
        genes = _matched_panel(stats, panel, rng)
        if genes is None:
            L.log('randompanel: could not build a unique panel %d -- skipping' % p)
            continue
        tag = 'rp%d' % p
        t0 = time.time()
        store, _ = fit_pass(genes, tag=tag, ckpt_name=tag)
        row = _panel_stats(stats, genes, store, os.path.join(L.REV, tag + '_W.npz'))
        row.update(panel=p, seed=seed0 + p, genes=list(genes))
        done.append(row)
        rows['rows'] = done
        L.save(rows, 'randompanel.json')
        L.log('  randompanel %d/%d: median_edges=%d recurring>=10=%d (%.0fs)'
              % (p + 1, n_panels, row['median_edges'], row['recurring_ge10'],
                 time.time() - t0))
    L.save(rows, 'randompanel.json')


# =============================================================== bootstrap (A-M1)
def _boot_one(job):
    """Sample-bootstrap one cohort and refit B times.

    job = (cohort, genes, B, seed) -> (cohort, n, [W_1..W_B]).
    Seeding is keyed on the cohort's position in the globally sorted cohort list, so the
    resamples do not depend on the order in which workers happen to finish.
    """
    c, genes, B, seed = job
    X, _ = L.expr(c, genes)
    n = int(X.shape[0])
    rng = np.random.default_rng(seed)
    out = []
    for _k in range(B):
        idx = rng.integers(0, n, n)
        W, _h = L.notears_lbfgs(X[idx, :], lam=0.01, max_outer=100)
        out.append(W)
    return c, n, out


def stage_bootstrap(B=6, seed=7):
    if LIMIT:
        B = 2
    meta = L.load_panel()
    panel, cancers = meta['panel'], meta['cancers']
    fits, cs_all = L.load_fits('_shared_panel_notears.json')
    M, _ = L.median_matrix(fits, cs_all, dref=len(panel))
    present = L.recurrence(M, L.TAU)
    pairs = [(r['src'], r['dst'], panel.index(r['src']), panel.index(r['dst']))
             for r in L.recurrence_list(present, panel, 10)]
    keyed = [(p[0], p[1]) for p in pairs]
    posn = {c: k for k, c in enumerate(L.all_cancers())}
    cs = cap(cs_all)
    rows = L.load_rev('bootstrap.json') or dict(pairs=keyed, B=B, per_cohort={})
    per = rows['per_cohort']
    todo = [c for c in cs if c not in per]
    genes = tuple(panel)
    jobs = [(c, genes, B, seed + 1000 * posn[c]) for c in todo]
    t_all = time.time()
    for i, (c, n, Wlist) in enumerate(L.pmap(_boot_one, jobs)):
        # edge src->dst carries W[i2, j2] under this project's convention (verified in
        # _rev_conv.py against the published per-pair support counts)
        draws = {'b%d' % k: [float(W[i2, j2]) for _, _, i2, j2 in pairs]
                 for k, W in enumerate(Wlist)}
        per[c] = dict(n=int(n), draws=draws)
        rows['per_cohort'] = per
        L.save(rows, 'bootstrap.json')
        with open(os.path.join(L.REV, 'bootstrap.ckpt.json'), 'w', encoding='utf-8') as f:
            import json
            json.dump(sorted(per), f)
        L.log('  bootstrap [%d/%d] %-5s n=%4d %d resamples  (%.0fs elapsed)'
              % (i + 1, len(todo), c, n, B, time.time() - t_all))

    if len(per) >= (1 if LIMIT else 33):
        summary = []
        for src, dst, i2, j2 in pairs:
            kk = keyed.index((src, dst))
            # cohorts whose original fit actually put an edge at this position
            sup = [c for c in sorted(per)
                   if abs(float(np.array(fits[c]['W'])[i2][j2])) > L.TAU]
            n_skel, n_ori, tot = 0, 0, 0
            for c in sup:
                orig_sign = np.sign(float(np.array(fits[c]['W'])[i2][j2]))
                for k in range(B):
                    v = per[c]['draws']['b%d' % k][kk]
                    tot += 1
                    if abs(v) > L.TAU:
                        n_skel += 1
                        if np.sign(v) == orig_sign:
                            n_ori += 1
            summary.append(dict(pair='%s->%s' % (src, dst), n_supporting_refit=len(sup),
                                n_draws=tot,
                                skeleton_retention=(n_skel / tot if tot else None),
                                orientation_retention=(n_ori / tot if tot else None)))
        rows['summary'] = summary
        L.save(rows, 'bootstrap.json')
        vals = [s['orientation_retention'] for s in summary
                if s['orientation_retention'] is not None]
        sks = [s['skeleton_retention'] for s in summary
               if s['skeleton_retention'] is not None]
        L.log('  bootstrap SUMMARY: mean orientation retention = %s ; mean skeleton = %s'
              % ('%.3f' % float(np.mean(vals)) if vals else 'n/a',
                 '%.3f' % float(np.mean(sks)) if sks else 'n/a'))


# =============================================================== lamscan (A-M3)
def stage_lamscan():
    meta = L.load_panel()
    panel, cancers = meta['panel'], meta['cancers']
    out = L.load_rev('lamscan.json') or {}
    for lam in ((0.005,) if LIMIT else (0.005, 0.02)):
        key = '%.3f' % lam
        if key in out:
            continue
        store, _ = fit_pass(panel, lam=lam, tag='lam%s' % key, ckpt_name='lam%s' % key)
        Ws = np.load(os.path.join(L.REV, 'lam%s_W.npz' % key), allow_pickle=True)
        grid = {}
        for tau in (0.20, 0.25, 0.30, 0.35, 0.40):
            Mr = np.stack([np.abs(Ws[c]).astype(float) for c in sorted(store)], axis=0)
            med = np.median(Mr, axis=0)
            np.fill_diagonal(med, 0.0)
            pr = L.recurrence(Mr, tau)
            rec = L.recurrence_list(pr, panel, 10)
            grid['%.2f' % tau] = dict(median_edges=int((med > tau).sum()),
                                      recurring_ge10=len(rec),
                                      pairs=['%s->%s' % (r['src'], r['dst']) for r in rec])
        out[key] = grid
        L.save(out, 'lamscan.json')
        L.log('  lamscan lam=%s done: %s' % (key, {k: v['recurring_ge10'] for k, v in grid.items()}))


# =============================================================== lowmad (A-M4)
def stage_lowmad(seed=4242):
    stats = L.load_rev('_gene_stats.json')
    if not stats:
        L.log('lowmad: need prep')
        return
    if L.load_rev('lowmad.json'):
        return
    genes_all = np.array(stats['genes'])
    mad = np.array(stats['mad_pct'])
    real = L.load_panel()['panel']
    # matched on mean-expression decile, drawn from the *low* dispersion half, without
    # replacement (the first version duplicated genes here too -- see _matched_sample)
    low = set(genes_all[mad <= 0.5])
    (eidx,) = _deciles(stats, ('expr_pct',))
    rng = np.random.default_rng(seed)
    pick = _matched_sample(stats, real, rng, lambda g: (eidx[g],), restrict=low)
    store, _ = fit_pass(pick, tag='lowmad', ckpt_name='lowmad')
    Ws = np.load(os.path.join(L.REV, 'lowmad_W.npz'), allow_pickle=True)
    cs = sorted(store)
    Mr = np.stack([np.abs(Ws[c]).astype(float) for c in cs], axis=0)
    med = np.median(Mr, axis=0)
    np.fill_diagonal(med, 0.0)
    pr = L.recurrence(Mr, L.TAU)
    out = dict(genes=pick, median_mad_pct=float(np.median([mad[list(genes_all).index(g)] for g in pick])),
               median_edges=int((med > L.TAU).sum()), max_abs_median=float(med.max()),
               recurring_ge10=len(L.recurrence_list(pr, pick, 10)),
               pairs=['%s->%s' % (r['src'], r['dst']) for r in L.recurrence_list(pr, pick, 10)],
               total_edges=int(sum(store[c]['edges'] for c in cs)))
    L.save(out, 'lowmad.json')
    L.log('  lowmad: median edges=%d recurring>=10=%d' % (out['median_edges'], out['recurring_ge10']))


# =============================================================== dscan (A-M6)
def stage_dscan(widths=(50, 100, 150, 200)):
    stats = L.load_rev('_gene_stats.json')
    if not stats:
        L.log('dscan: need prep')
        return
    genes_all = np.array(stats['genes'])
    mad = np.array(stats['mad_pct'])
    order = genes_all[np.argsort(-mad)]
    if LIMIT:
        widths = (50, 100)
    out = L.load_rev('dscan.json') or {}
    for d in widths:
        key = 'd%d' % d
        if key in out:
            continue
        if d == 100:
            fits, cs = L.load_fits('_shared_panel_notears.json')
            Ws = {c: np.array(fits[c]['W']) for c in cs}
            store = {c: dict(n=fits[c]['n'], edges=fits[c]['edges']) for c in cs}
        else:
            store, _ = fit_pass(list(order[:d]), tag=key, ckpt_name=key)
            z = np.load(os.path.join(L.REV, key + '_W.npz'), allow_pickle=True)
            Ws = {c: z[c].astype(float) for c in sorted(store)}
        ns = np.array([store[c]['n'] for c in sorted(store)], float)
        es = np.array([store[c]['edges'] for c in sorted(store)], float)
        o = np.argsort(ns)
        ns, es = ns[o], es[o]
        r, p = __import__('scipy.stats', fromlist=['x']).pearsonr(ns, es)
        # empirical onset: largest n still yielding >=10 edges vs smallest n yielding <10
        hi = [n for n, e in zip(ns, es) if e >= 10]
        lo = [n for n, e in zip(ns, es) if e < 10]
        out[key] = dict(d=d, n_cohorts=len(ns), total_edges=int(es.sum()),
                        pearson_r=float(r), pearson_p=float(p),
                        max_n_with_edges=int(max(hi)) if hi else None,
                        min_n_without_edges=int(min(lo)) if lo else None,
                        ratio_over_d=float(max(hi) / d) if hi else None,
                        per_cohort={c: dict(n=int(store[c]['n']), edges=int(store[c]['edges']))
                                    for c in sorted(store)})
        L.save(out, 'dscan.json')
        L.log('  dscan %s: total=%d r=%.3f  n*~%s (=%.2f d)'
              % (key, out[key]['total_edges'], r, out[key]['max_n_with_edges'],
                 out[key]['ratio_over_d'] or float('nan')))


# =============================================================== dscan_null (A-M6)
def _shuffle_one(job):
    """One cohort in the noise regime at panel width d: shuffle each gene, then refit.

    job = (cohort, genes, seed) -> (cohort, n, edges, h).  Same control as
    scripts/analysis/shuffle_control_fit.py, generalised to an arbitrary panel width, which
    is what lets the sample floor be located at more than one d.
    """
    c, genes, seed = job
    X, _ = L.expr(c, genes)
    rng = np.random.default_rng(seed)
    for j in range(X.shape[1]):
        X[:, j] = rng.permutation(X[:, j])
    W, h = L.notears_lbfgs(X, lam=0.01, max_outer=100)
    return c, int(X.shape[0]), int((np.abs(W) > L.TAU).sum()), float(h)


def stage_dscan_null(widths=(50, 100, 150, 200), seed0=71):
    """Locate the noise boundary at each panel width (reviewer M6).

    The published boundary (n = 156 still generating edges, n = 172 not) is a single point at
    d = 100; the reviewer's objection is that n* ~ 1.7d is then calibrated on one width.  This
    repeats the shuffle control at four widths on the same top-d-by-dispersion panels used by
    the real-data width scan, so the ratio n*/d can be read off instead of assumed.
    """
    stats = L.load_rev('_gene_stats.json')
    if not stats:
        L.log('dscan_null: need prep')
        return
    order = np.array(stats['genes'])[np.argsort(-np.array(stats['mad_pct']))]
    if LIMIT:
        widths = (50, 100)
    out = L.load_rev('dscan_null.json') or {}
    posn = {c: k for k, c in enumerate(L.all_cancers())}
    for d in widths:
        key = 'd%d' % d
        if key in out:
            continue
        genes = tuple(order[:d])
        jobs = [(c, genes, seed0 + 1000 * posn[c] + d) for c in L.all_cancers()]
        per = {}
        t0 = time.time()
        for _job, (c, n, e, h) in L.pmap(_shuffle_one, jobs):
            per[c] = dict(n=n, edges=e, h=h)
            L.log('  dscan_null d=%3d %-5s n=%4d edges=%3d h=%.2e  (%.0fs elapsed)'
                  % (d, c, n, e, h, time.time() - t0))
        nz = sorted(v['n'] for v in per.values() if v['edges'] > 0)
        zr = sorted(v['n'] for v in per.values() if v['edges'] == 0)
        ns = np.array([v['n'] for v in per.values()], float)
        es = np.array([v['edges'] for v in per.values()], float)
        r, p = __import__('scipy.stats', fromlist=['x']).pearsonr(ns, es)
        out[key] = dict(d=d, n_cohorts=len(per), total_edges=int(es.sum()),
                        pearson_r=float(r), pearson_p=float(p),
                        n_with_edges=len(nz), n_without_edges=len(zr),
                        max_n_with_edges=int(nz[-1]) if nz else None,
                        min_n_without_edges=int(zr[0]) if zr else None,
                        boundary_over_d=(float(zr[0]) / d) if zr else None,
                        per_cohort=per)
        L.save(out, 'dscan_null.json')
        L.log('  dscan_null d=%3d: r=%.3f ; max n with edges=%s ; min n without=%s ; '
              'n*/d=%s' % (d, r, out[key]['max_n_with_edges'],
                           out[key]['min_n_without_edges'], out[key]['boundary_over_d']))
    L.save(out, 'dscan_null.json')


# =============================================================== rotation (A-M2b)
def stage_rotation(reps=3, seed=909):
    meta = L.load_panel()
    panel, cancers = meta['panel'], meta['cancers']
    if LIMIT:
        reps = 1
    d = len(panel)
    out = L.load_rev('rotation.json') or dict(rows=[])
    for rep in range(len(out['rows']), reps):
        rng = np.random.default_rng(seed + rep)
        Qm, _ = np.linalg.qr(rng.normal(size=(d, d)))
        tag = 'rot%d' % rep

        # the rotation is passed to the workers as a flat array (picklable), not a closure
        store, _ = fit_pass(panel, tag=tag, ckpt_name=tag, Q=Qm)
        z = np.load(os.path.join(L.REV, tag + '_W.npz'), allow_pickle=True)
        cs = sorted(store)
        Mr = np.stack([np.abs(z[c]).astype(float) for c in cs], axis=0)
        med = np.median(Mr, axis=0)
        np.fill_diagonal(med, 0.0)
        pr = L.recurrence(Mr, L.TAU)
        out['rows'].append(dict(rep=rep, seed=seed + rep,
                                median_edges=int((med > L.TAU).sum()),
                                recurring_ge10=len(L.recurrence_list(pr, panel, 10)),
                                q=L.qstat(pr), nk=L.nk_counts(pr),
                                total_edges=int(sum(store[c]['edges'] for c in cs))))
        L.save(out, 'rotation.json')
        L.log('  rotation %d/%d: median_edges=%d recurring>=10=%d'
              % (rep + 1, reps, out['rows'][-1]['median_edges'],
                 out['rows'][-1]['recurring_ge10']))


# =============================================================== multiinit (A-M1b)
def _init_one(job):
    """Refit one cohort from several random starting points (A-M1b).

    job = (cohort, genes, inits, seed) -> (cohort, [W_1..W_inits]).
    """
    c, genes, inits, seed = job
    X, _ = L.expr(c, genes)
    out = []
    for k in range(inits):
        W, _h = L.notears_lbfgs(X, lam=0.01, max_outer=100, seed=seed + k)
        out.append(W)
    return c, out


def stage_multiinit(inits=6, n_cohorts=20, seed=555):
    meta = L.load_panel()
    panel, cancers = meta['panel'], meta['cancers']
    fits, cs_all = L.load_fits('_shared_panel_notears.json')
    M, _ = L.median_matrix(fits, cs_all, dref=len(panel))
    present = L.recurrence(M, L.TAU)
    pairs = [(r['src'], r['dst'], panel.index(r['src']), panel.index(r['dst']))
             for r in L.recurrence_list(present, panel, 10)]
    keyed = [(p[0], p[1]) for p in pairs]
    key = sorted(cs_all, key=lambda c: -fits[c]['edges'])[:(1 if LIMIT else n_cohorts)]
    n_cohorts = len(key)
    if LIMIT:
        inits = 2
    posn = {c: k for k, c in enumerate(cs_all)}
    out = L.load_rev('multiinit.json') or dict(pairs=keyed, inits=inits, per_cohort={})
    per = out['per_cohort']
    genes = tuple(panel)
    todo = [c for c in key if c not in per]
    jobs = [(c, genes, inits, seed + 100 * posn[c]) for c in todo]
    t_all = time.time()
    for i, (c, Wlist) in enumerate(L.pmap(_init_one, jobs)):
        per[c] = {'i%d' % k: [float(W[i2, j2]) for _, _, i2, j2 in pairs]
                  for k, W in enumerate(Wlist)}
        out['per_cohort'] = per
        L.save(out, 'multiinit.json')
        L.log('  multiinit [%d/%d] %-5s x%d done  (%.0fs elapsed)'
              % (i + 1, len(todo), c, inits, time.time() - t_all))
    if len(per) >= n_cohorts:
        summ = []
        for src, dst, i2, j2 in pairs:
            kk = keyed.index((src, dst))
            ag, tot = 0, 0
            for c in sorted(per):
                sgn = np.sign(float(np.array(fits[c]['W'])[i2][j2]))
                if abs(sgn) < 1e-12:
                    continue
                for k in range(inits):
                    v = per[c]['i%d' % k][kk]
                    tot += 1
                    if abs(v) > L.TAU and np.sign(v) == sgn:
                        ag += 1
            summ.append(dict(pair='%s->%s' % (src, dst), n_draws=tot,
                             init_agreement=(ag / tot if tot else None)))
        out['summary'] = summ
        L.save(out, 'multiinit.json')
        vals = [s['init_agreement'] for s in summ if s['init_agreement'] is not None]
        L.log('  multiinit SUMMARY: mean init agreement = %s'
              % ('%.3f' % float(np.mean(vals)) if vals else 'n/a'))


STAGES = dict(prep=stage_prep, desk=stage_desk, composition=stage_composition,
              randompanel=stage_randompanel, randompanel_dm=stage_randompanel_dm,
              dscan_null=stage_dscan_null,
              bootstrap=stage_bootstrap,
              lamscan=stage_lamscan, lowmad=stage_lowmad, dscan=stage_dscan,
              rotation=stage_rotation, multiinit=stage_multiinit)


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else 'all'
    want = ALL_STAGES if arg == 'all' else [s.strip() for s in arg.split(',') if s.strip()]
    print('=' * 70)
    print('revision night batch  stages =', want)
    print('start', time.strftime('%Y-%m-%d %H:%M:%S'))
    print('=' * 70, flush=True)
    for s in want:
        if s not in STAGES:
            print('!! unknown stage', s)
            continue
        t0 = time.time()
        try:
            STAGES[s]()
            mark(s, True, '%.1f s' % (time.time() - t0))
        except Exception as e:
            mark(s, False, '%s: %s' % (type(e).__name__, e))
            print('!! stage %s FAILED: %s' % (s, e))
            traceback.print_exc()
    print('=' * 70)
    print('END', time.strftime('%Y-%m-%d %H:%M:%S'))
    import io
    import json
    print(json.dumps(manifest(), indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
