"""Robustness of the headline sharing result, computed from the stored per-cancer graphs.

Four numbers in the paper depend on a choice that was previously left implicit, and a reviewer
cannot be expected to guess which choice was made:

  1. A "gene-pair" can be counted with or without its orientation.  Counting it directed means that
     two cohorts which infer X->Y and Y->X are recorded as two pairs seen once each, which lowers
     the measured sharing; counting it undirected treats them as one pair seen twice.  The
     manuscript quotes the directed count as the primary figure, so both are written out here.
  2. The mean per-cancer reuse rate averages over cohorts, so a cohort of 45 samples contributes as
     much as one of 1,218.  The edge-weighted value, which answers "of the edges, what fraction is
     found elsewhere", is the one a reader usually has in mind, and is also written out here.
  3. Edge counts fall steeply with sample size across all 33 cohorts (Spearman r = -0.84).  Whether
     that is biology or an artefact of fitting d = 100 variables to fewer than 100 samples can be
     settled by dropping the small cohorts, so the correlation is reported for several cut-offs.
  4. The design rests on the premise that the top-100 gene sets differ between cohorts.  The mean
     number of genes shared by two cohorts' sets quantifies that premise directly.

Reads: results/_pipeline_notears.json, results/_per_cancer_network.json
Writes: results/_context_robustness.json
"""
import os, sys, json, collections, itertools
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
from scipy.stats import spearmanr

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(BASE, "results")
TAU = 0.3

D = json.load(open(os.path.join(RES, "_pipeline_notears.json"), encoding="utf-8"))
PC = json.load(open(os.path.join(RES, "_per_cancer_network.json"), encoding="utf-8"))
cancers = sorted(D.keys())
n_map = {c: int(PC[c]["n"]) for c in cancers}

directed = {c: set() for c in cancers}
undirected = {c: set() for c in cancers}
genes_by_c = {}
n_directed = 0
for c in cancers:
    W = np.array(D[c]["W"], dtype=float)
    g = list(D[c]["genes"])
    genes_by_c[c] = g
    d = len(g)
    for i in range(d):
        for j in range(d):
            if i == j or W[i, j] is None:
                continue
            if abs(float(W[i, j])) > TAU:
                n_directed += 1
                directed[c].add((g[i], g[j]))
                undirected[c].add("|".join(sorted([g[i], g[j]])))

d_cnt = collections.Counter()
u_cnt = collections.Counter()
for c in cancers:
    for p in directed[c]:
        d_cnt[p] += 1
    for p in undirected[c]:
        u_cnt[p] += 1


def summarise(cnt, per_cancer):
    uniq = len(cnt)
    g2 = sum(1 for v in cnt.values() if v >= 2)
    g3 = sum(1 for v in cnt.values() if v >= 3)
    reuse = [100 * sum(1 for p in per_cancer[c] if cnt[p] >= 2) / max(len(per_cancer[c]), 1)
             for c in cancers]
    shared_edges = sum(sum(1 for p in per_cancer[c] if cnt[p] >= 2) for c in cancers)
    total_edges = sum(len(per_cancer[c]) for c in cancers)
    return {
        "unique_pairs": uniq,
        "shared_ge2": g2, "shared_ge3": g3,
        "pct_ge2": 100 * g2 / uniq, "pct_ge3": 100 * g3 / uniq,
        "max_N": max(cnt.values()),
        "reuse_cohort_mean_pct": float(np.mean(reuse)),
        "reuse_edge_weighted_pct": 100 * shared_edges / total_edges,
        "edges_unique_pct": 100 * (total_edges - shared_edges) / total_edges,
        "reuse_vs_n_spearman_r": float(spearmanr([n_map[c] for c in cancers], reuse)[0]),
        "reuse_vs_n_spearman_p": float(spearmanr([n_map[c] for c in cancers], reuse)[1]),
    }


out = {
    "tau": TAU,
    "n_directed_edges": n_directed,
    "directed": summarise(d_cnt, directed),
    "undirected": summarise(u_cnt, undirected),
}

# ---- edge concentration: how much of the edge set comes from the smaller cohorts ----
conc = []
for thr in (100, 200, 300, 500, 1000):
    small = [c for c in cancers if n_map[c] < thr]
    e = sum(len(directed[c]) for c in small)
    out_tot = sum(len(directed[c]) for c in cancers)
    n_tot = sum(n_map.values())
    conc.append({"n_below": thr, "cohorts": len(small),
                 "pct_samples": 100 * sum(n_map[c] for c in small) / n_tot,
                 "edges": e, "pct_edges": 100 * e / out_tot})
out["edge_concentration"] = conc

# ---- sharing recomputed on the larger cohorts only ----
restricted = []
for thr in (0, 200, 300, 500):
    keep = [c for c in cancers if n_map[c] >= thr]
    cnt = collections.Counter()
    for c in keep:
        for p in undirected[c]:
            cnt[p] += 1
    u = len(cnt)
    restricted.append({
        "n_at_least": thr, "cohorts": len(keep), "unique_pairs": u,
        "shared_ge2": sum(1 for v in cnt.values() if v >= 2),
        "shared_ge3": sum(1 for v in cnt.values() if v >= 3),
        "pct_ge2": 100 * sum(1 for v in cnt.values() if v >= 2) / u,
        "pct_ge3": 100 * sum(1 for v in cnt.values() if v >= 3) / u,
    })
out["sharing_by_cohort_size"] = restricted

# ---- edge count vs sample size, with and without the small cohorts ----
efits = []
for thr in (0, 200, 300, 500):
    keep = [c for c in cancers if n_map[c] >= thr]
    r, p = spearmanr([n_map[c] for c in keep], [len(directed[c]) for c in keep])
    efits.append({"n_at_least": thr, "cohorts": len(keep), "r": float(r), "p": float(p),
                  "edges_mean": float(np.mean([len(directed[c]) for c in keep]))})
out["edges_vs_n_by_cohort_size"] = efits

# ---- premise check: overlap of the per-cohort top-100 gene sets ----
jac, shared_n = [], []
for a, b in itertools.combinations(cancers, 2):
    A, B = set(genes_by_c[a]), set(genes_by_c[b])
    jac.append(len(A & B) / len(A | B))
    shared_n.append(len(A & B))
allg = set().union(*[set(v) for v in genes_by_c.values()])
out["gene_set_overlap"] = {
    "pairwise_shared_genes_mean": float(np.mean(shared_n)),
    "pairwise_shared_genes_max": int(np.max(shared_n)),
    "jaccard_mean": float(np.mean(jac)), "jaccard_median": float(np.median(jac)),
    "jaccard_max": float(np.max(jac)), "jaccard_min": float(np.min(jac)),
    "union_of_network_genes": len(allg),
}
# ---- loci the source matrix carries without a gene symbol ----
out["unannotated_loci"] = {
    "genes": sorted({x for v in genes_by_c.values() for x in v if "?" in x or "|" in x}),
    "cohorts": sorted({c for c in cancers for x in genes_by_c[c] if "?" in x}),
}

json.dump(out, open(os.path.join(RES, "_context_robustness.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

d, u = out["directed"], out["undirected"]
print("directed  : %d pairs | >=2 %d (%.2f%%) | >=3 %d (%.2f%%) | cohort-mean reuse %.1f%% | edge-weighted %.1f%%"
      % (d["unique_pairs"], d["shared_ge2"], d["pct_ge2"], d["shared_ge3"], d["pct_ge3"],
         d["reuse_cohort_mean_pct"], d["reuse_edge_weighted_pct"]))
print("undirected: %d pairs | >=2 %d (%.2f%%) | >=3 %d (%.2f%%) | cohort-mean reuse %.1f%% | edge-weighted %.1f%%"
      % (u["unique_pairs"], u["shared_ge2"], u["pct_ge2"], u["shared_ge3"], u["pct_ge3"],
         u["reuse_cohort_mean_pct"], u["reuse_edge_weighted_pct"]))
print()
for row in conc:
    print("  n < %-4d: %2d cohorts, %4.1f%% of samples, %5d edges (%4.1f%%)"
          % (row["n_below"], row["cohorts"], row["pct_samples"], row["edges"], row["pct_edges"]))
print()
for row in restricted:
    print("  n >= %-4d: %2d cohorts | pairs %5d | >=2 %4d (%5.2f%%) | >=3 %3d (%5.2f%%)"
          % (row["n_at_least"], row["cohorts"], row["unique_pairs"],
             row["shared_ge2"], row["pct_ge2"], row["shared_ge3"], row["pct_ge3"]))
print()
for row in efits:
    print("  edges vs n, n >= %-4d (%2d cohorts): r = %+.3f (p = %.4f)"
          % (row["n_at_least"], row["cohorts"], row["r"], row["p"]))
print()
jo = out["gene_set_overlap"]
print("gene-set overlap: mean %.1f of 100 genes shared (Jaccard mean %.3f, median %.3f, max %.3f); union %d"
      % (jo["pairwise_shared_genes_mean"], jo["jaccard_mean"], jo["jaccard_median"],
         jo["jaccard_max"], jo["union_of_network_genes"]))
print("unannotated loci: %s" % out["unannotated_loci"]["genes"])
print("\nwrote results/_context_robustness.json")
