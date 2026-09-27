# -*- coding: utf-8 -*-
"""Identifiability diagnostic.

Reports, for every cohort, the sample size against the number of fitted variables
(d = 100) and the edge count; aggregates the n >= 200 main-analysis panel and the
edge-concentration summary.  Consumes only checkpoints already shipped in ./results.

Output: results/_identifiability.json (+ a table on stdout)
"""
import json, os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
RES = os.path.join(_ROOT, 'results')
OUT = os.path.join(RES, '_identifiability.json')
DD = 100  # number of fitted variables per cohort

net = json.load(open(os.path.join(RES, '_per_cancer_network.json'), encoding='utf-8'))
ctx = json.load(open(os.path.join(RES, '_context_robustness.json'), encoding='utf-8'))

rows = [(c, int(v['n']), int(v['n_edges'])) for c, v in net.items()]
rows.sort(key=lambda r: r[1])
tot_e = sum(r[2] for r in rows)

print('cohorts = %d | total edges = %d | d = %d' % (len(rows), tot_e, DD))
print()
print('%-6s %7s %8s  %s' % ('cohort', 'n', 'n_edges', 'flag'))
for c, n, e in rows:
    fl = []
    if n < DD:
        fl.append('RANK-DEFICIENT (n<d)')
    if n < 200:
        fl.append('n<200')
    print('%-6s %7d %8d  %s' % (c, n, e, ' '.join(fl)))

lt_d = [r for r in rows if r[1] < DD]
lt200 = [r for r in rows if r[1] < 200]
ge200 = [r for r in rows if r[1] >= 200]


def agg(name, sub):
    e = sum(r[2] for r in sub)
    return dict(name=name, n_cohorts=len(sub), edges=e, pct_edges=round(100.0 * e / tot_e, 1))


print()
for nm, sub in [('n<d', lt_d), ('n<200', lt200), ('n>=200', ge200)]:
    a = agg(nm, sub)
    print('%-7s : %2d cohorts, %5d edges = %5.1f%% of all'
          % (a['name'], a['n_cohorts'], a['edges'], a['pct_edges']))

rep = dict(d=DD, n_cohorts=len(rows), total_edges=tot_e,
           per_cohort=[dict(cohort=c, n=n, n_edges=e) for c, n, e in rows],
           groups=[agg('n<d', lt_d), agg('n<200', lt200), agg('n>=200', ge200)],
           ctx_edges_vs_n_by_cohort_size=ctx.get('edges_vs_n_by_cohort_size'),
           ctx_edge_concentration=ctx.get('edge_concentration'),
           ctx_sharing_by_cohort_size=ctx.get('sharing_by_cohort_size'),
           ctx_gene_set_overlap=ctx.get('gene_set_overlap'))
with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(rep, f, ensure_ascii=False, indent=1)
print('\nwrote', OUT)

for k in ['edges_vs_n_by_cohort_size', 'edge_concentration', 'sharing_by_cohort_size', 'gene_set_overlap']:
    print('\n--- ctx.%s ---' % k)
    print(json.dumps(ctx.get(k), ensure_ascii=False, indent=1)[:1200])
