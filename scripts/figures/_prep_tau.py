"""Pass 2: tissue-specificity index (tau, Yanai et al. 2005) for every gene in the per-cancer
networks, so that hub genes can be compared against non-hub network genes (same gene universe,
same selection rule).  Data source: TCGA HiSeqV2 log2(TPM+1) matrices, already on disk.
Output: results/_tau_specificity.json
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _gene_symbols import canonical, index

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(BASE, 'results')
# TCGA matrices: ./data/ inside the package, or the folder named by MULTIBATCH_DATA
_LOCAL = os.path.join(BASE, 'data')
TCGA_DIR = os.environ.get('MULTIBATCH_DATA') or _LOCAL

def log(*a):
    print(*a, flush=True)

pc = json.load(open(os.path.join(RESULTS, '_per_cancer_network.json'), encoding='utf-8'))
cancers = sorted(pc.keys())
universe = set()
for c in cancers:
    universe |= {canonical(g) for g in pc[c]['genes']}
log('gene universe = %d   cancers = %d' % (len(universe), len(cancers)))

# Xena row names differ from the network tables in case ("C11orf86" against "C11ORF86") and, for
# the renamed loci, in symbol itself ("C9orf84" against the table's C9ORF84). Without this index
# those rows are dropped silently and a gene's tau is computed from whichever cohorts survive --
# which is what left two hub genes out of the statistic. See _gene_symbols.py.
u_idx = index(universe)

med = {c: {} for c in cancers}                 # med[cancer][gene] = median expression
hub_raw = {}                                   # hub gene raw vectors per cancer
hub_of = {c: canonical(pc[c]['hub']) for c in cancers}
hub_genes = set(hub_of.values())

for c in cancers:
    p = os.path.join(TCGA_DIR, 'TCGA_%s_HiSeqV2.tsv' % c)
    if not os.path.exists(p):
        log('  MISSING %s' % p)
        continue
    row = {}
    with open(p, encoding='utf-8', errors='replace') as f:
        f.readline()
        for line in f:
            parts = line.rstrip('\n').split('\t')
            g = u_idx.get(canonical(parts[0]).upper())
            if g is None:
                continue
            vals = []
            for v in parts[1:]:
                try:
                    vals.append(float(v))
                except ValueError:
                    pass
            if not vals:
                continue
            row[g] = float(np.median(vals))
            if g in hub_genes:
                hub_raw.setdefault(g, {})[c] = vals
    med[c] = row
    log('  %-5s genes matched = %d' % (c, len(row)))

# tau = sum(1 - x/max(x)) / (n-1); 0 = ubiquitous, 1 = tissue specific
def tau_of(g):
    xs = []
    for c in cancers:
        v = med.get(c, {}).get(g)
        if v is not None:
            xs.append(v)
    if len(xs) < 15:
        return None
    xs = np.asarray(xs, dtype=float)
    mx = xs.max()
    if mx <= 0:
        return None
    return float(np.sum(1.0 - xs / mx) / (len(xs) - 1))

hub_tau, nonhub_tau = [], []
gene_tau = {}
# Sorted, not raw set order: iterating a set of strings depends on the per-process hash
# seed, which reordered hub_tau/nonhub_tau from run to run.  Every consumer is
# order-invariant (mean, median, Mann-Whitney, violin/box plots), so nothing downstream
# moves -- but the file now reproduces byte for byte.
for g in sorted(universe):
    t = tau_of(g)
    if t is None:
        continue
    gene_tau[g] = t
    (hub_tau if g in hub_genes else nonhub_tau).append(t)

hub_tau = np.asarray(hub_tau); nonhub_tau = np.asarray(nonhub_tau)
log('\ntau computed for %d genes' % len(gene_tau))
log('  hub genes      n=%d  mean tau=%.3f  median=%.3f' % (len(hub_tau), hub_tau.mean(), np.median(hub_tau)))
log('  non-hub genes  n=%d  mean tau=%.3f  median=%.3f' % (len(nonhub_tau), nonhub_tau.mean(), np.median(nonhub_tau)))

# tau = 1 means tissue-specific and tau = 0 ubiquitous, so the claim the manuscript
# makes ("hubs are significantly less tissue-restricted") is the *less* tail.  Every
# tail is computed and recorded under its own name: the single unlabelled `mw_p` that
# used to be written here held the opposite tail (0.977), which reads as the claim
# being false even though the two-sided value it implies (0.047) is the one Figure 6
# quotes and re-derives for itself.
try:
    from scipy.stats import mannwhitneyu
    u, p_less = mannwhitneyu(hub_tau, nonhub_tau, alternative='less')
    _, p_greater = mannwhitneyu(hub_tau, nonhub_tau, alternative='greater')
    _, p_two = mannwhitneyu(hub_tau, nonhub_tau, alternative='two-sided')
    log('  Mann-Whitney U=%.0f  two-sided p=%.4g  |  hubs less tissue-restricted, one-sided p=%.4g  |  other tail %.4g'
        % (u, p_two, p_less, p_greater))
except Exception as e:
    u = p_less = p_greater = p_two = float('nan')
    log('  MWU failed %s' % e)

# rank test vs uniform null (1..33) for hub genes, from pass 1
p1 = json.load(open(os.path.join(RESULTS, '_hub_expression.json'), encoding='utf-8'))
ranks = [v['rank'] for v in p1['stats'].values()]
rng = np.random.default_rng(0)
null = rng.integers(1, 34, size=(20000, len(ranks))).mean(axis=1)
obs = float(np.mean(ranks))
p_perm = float(np.mean(null <= obs))
log('  hub rank: observed mean=%.2f vs null=17.0  permutation p=%.4g (n=%d)' % (obs, p_perm, len(ranks)))

json.dump({
    'hub_tau': hub_tau.tolist(),
    'nonhub_tau': nonhub_tau.tolist(),
    'gene_tau': gene_tau,
    'hub_genes': sorted(hub_genes),
    'hub_rank_mean': obs,
    'hub_rank_perm_p': p_perm,
    'mw_u': None if u != u else float(u),
    'mw_p_hub_less_tissue_specific': None if p_less != p_less else float(p_less),
    'mw_p_hub_more_tissue_specific': None if p_greater != p_greater else float(p_greater),
    'mw_p_two_sided': None if p_two != p_two else float(p_two),
    'cancers': cancers,
    'hub_raw': hub_raw,
    'hub_of_cancer': hub_of,
}, open(os.path.join(RESULTS, '_tau_specificity.json'), 'w', encoding='utf-8'), ensure_ascii=False)
log('wrote _tau_specificity.json')
log('DONE')
