# -*- coding: utf-8 -*-
"""Verify that the earlier positive results survive the n >= 200 restriction.

The earlier version reported three positive results:
  P1 tissue enrichment (surfactant metabolism in LUAD, 4.6-fold)
  P2 prognostic signal in aggregate (BRCA, 98 genes, KS p = 1.2e-5; 13 significant against
     4.9 expected)
  P3 the cohort's own hub is prognostic (6 of 33 against 1.65 expected)
Each is recomputed over all 33 cohorts and over the 18 cohorts with n >= 200.

Output: results/_numbers_verification.json
"""
import os, json
import numpy as np
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
DATA = os.path.join(_ROOT, 'data')
RES = os.path.join(_ROOT, 'results')
PANELS = os.path.join(DATA, 'panels')
FIG = os.path.join(_ROOT, 'figures')

E = RES  # scratch alias
OUT = os.path.join(RES, '_numbers_verification.json')


def load(n):
    return json.load(open(os.path.join(RES, n), encoding='utf-8'))


pc = load('_per_cancer_network.json')
n_by_c = {c: (v.get('n') if isinstance(v, dict) else None) for c, v in pc.items()}
big = sorted([c for c, n in n_by_c.items() if n and n >= 200])
small = sorted([c for c, n in n_by_c.items() if n and n < 200])
print('n>=200 cohorts (%d): %s' % (len(big), ', '.join(big)))
print('n<200  cohorts (%d): %s' % (len(small), ', '.join(small)))
print()

res = dict(n_ge200=big, n_lt200=small,
           n_by_cohort={c: n_by_c[c] for c in sorted(n_by_c)})

# ---------- P1 tissue enrichment ----------
ep = load('_enrichment_percancer.json')
print('=' * 20, 'P1 tissue enrichment')
print('keys of per_cancer entry:', end=' ')
key0 = next(iter(ep['per_cancer']))
print(key0, '->', json.dumps(ep['per_cancer'][key0], ensure_ascii=False)[:500])
print()

# ---------- P2 BRCA KS ----------
print('=' * 20, 'P2 BRCA network genes vs survival (n=1218, unaffected by n filter)')
br = load('_km_scan_brca.json')
pv = np.array([x['logrank_p'] for x in br], float)
ks = stats.kstest(pv, 'uniform')
nsig = int((pv < 0.05).sum())
print('genes = %d | KS D = %.4f  p = %.3g | sig(p<0.05) = %d (expected %.1f = %.1fx)'
      % (len(pv), ks.statistic, ks.pvalue, nsig, 0.05 * len(pv), nsig / (0.05 * len(pv))))
res['P2_brca'] = dict(n_genes=len(pv), ks_D=float(ks.statistic), ks_p=float(ks.pvalue),
                      n_sig=nsig, expected=0.05 * len(pv), fold=nsig / (0.05 * len(pv)))

# ---------- P3 hub prognostic ----------
print()
print('=' * 20, 'P3 cohort hub prognostic')
kp = load('_km_pancan.json')


def hubp(c):
    v = kp.get(c)
    if not isinstance(v, dict):
        return None
    hub = v.get('hub')
    g = v.get('genes', {}).get(hub)
    return g.get('logrank_p') if isinstance(g, dict) else None


rows = []
for c, n in sorted(n_by_c.items(), key=lambda t: -(t[1] or 0)):
    if not isinstance(kp.get(c), dict):
        continue
    p = hubp(c)
    if p is None:
        continue
    rows.append(dict(cohort=c, n=int(n), hub=kp[c]['hub'], p=float(p), sig=bool(p < 0.05)))


def report(tag, sel):
    s = [r for r in rows if r['cohort'] in sel]
    k = sum(r['sig'] for r in s)
    exp = 0.05 * len(s)
    print('%-16s cohorts=%2d | hub sig(p<0.05) = %d | expected %.2f | fold %.2fx'
          % (tag, len(s), k, exp, k / exp if exp else float('nan')))
    return dict(n_cohorts=len(s), n_sig=k, expected=exp, fold=(k / exp if exp else None))


r_all = report('ALL 33', set(n_by_c))
r_big = report('n>=200', set(big))
r_small = report('n<200', set(small))
print()
print('per-cohort detail (sorted by n):')
print('%-6s %6s %-12s %10s %s' % ('coh', 'n', 'hub', 'logrank_p', 'sig'))
for r in rows[::-1]:
    print('%-6s %6d %-12s %10.3g %s' % (r['cohort'], r['n'], r['hub'], r['p'], 'Y' if r['sig'] else ''))
res['P3_hub'] = dict(all33=r_all, ge200=r_big, lt200=r_small, per_cohort=rows)

# ---------- side note: the earlier sharing statistic under n >= 200 ----------
print()
print('=' * 20, 'P0 recomputation of the earlier cross-cancer sharing statistic (misaligned panel) under n >= 200')
cps = load('_cancer_pair_sharing.json')
print('keys:', list(cps.keys()))
for k in cps:
    if not isinstance(cps[k], (list, dict)):
        print('  %-24s = %s' % (k, cps[k]))
    elif isinstance(cps[k], list) and len(cps[k]) < 12:
        print('  %-24s = %s' % (k, str(cps[k])[:160]))
    else:
        print('  %-24s = <%s len=%d>' % (k, type(cps[k]).__name__, len(cps[k])))
res['old_sharing_keys'] = {k: (v if isinstance(v, (int, float, str)) else type(v).__name__)
                           for k, v in cps.items()}

json.dump(res, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print()
print('wrote', OUT)
