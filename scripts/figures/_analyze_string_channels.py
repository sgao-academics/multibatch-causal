# -*- coding: utf-8 -*-
"""Is the STRING overlap real evidence, or just co-expression talking to co-expression?

Two questions, in order:
  1. Enrichment  -- how often does a random pair of network genes reach score>=0.7 vs our pairs?
  2. Evidence mix -- which STRING channel carries the signal: experiments / curated database,
     or co-expression (which would make the check circular, since the causal search is run on
     expression data)?
"""
import os, sys, json

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    from scipy.stats import fisher_exact
except Exception:
    fisher_exact = None

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results")
D = json.load(open(os.path.join(RES, "_string_channels.json"), encoding="utf-8"))

SE = D["string_edges"]                 # every STRING edge inside the 1775 network genes
CP = D["causal_pairs"]                 # our deduplicated causal pairs -> cancers
NG = D["n_genes"]

CH = [("escore", "experiments"), ("dscore", "curated database"), ("tscore", "text mining"),
      ("ascore", "co-expression"), ("nscore", "neighbourhood"), ("fscore", "gene fusion"),
      ("pscore", "phylogenetic profile")]
EV = 0.041                             # STRING's own "channel has evidence" floor

hits = {k: SE[k] for k in CP if k in SE}
print("network genes %d | STRING edges inside them %d | our causal pairs %d | overlapping %d"
      % (NG, len(SE), len(CP), len(hits)))

# ---------- 1. enrichment against the random-pair expectation ----------
npairs = NG * (NG - 1) / 2.0
exp_rate = len(SE) / npairs
print("\nrandom pair of network genes reaching STRING score>=0.7: %.4f%% (%d of %.0f)"
      % (100 * exp_rate, len(SE), npairs))
print("our causal pairs reaching it:                              %.2f%% (%d of %d)"
      % (100 * len(hits) / len(CP), len(hits), len(CP)))
print("fold enrichment: %.1fx" % ((len(hits) / len(CP)) / exp_rate))

if fisher_exact:
    tbl = [[len(hits), len(CP) - len(hits)]]
    tot = len(SE)
    tbl.append([tot - len(hits), int(npairs - len(CP) - (tot - len(hits)))])
    # explicit 2x2: our pairs vs rest-of-genome pairs, hit vs miss
    a = len(hits)
    b = len(CP) - len(hits)
    c = len(SE) - len(hits)
    d = int(npairs) - len(SE) - b
    orr, p = fisher_exact([[a, b], [c, d]], alternative="greater")
    print("Fisher exact (one-sided, greater): OR = %.2f, p = %.3g" % (orr, p))
else:
    orr, p = float("nan"), float("nan")
    print("scipy unavailable, skipped Fisher")

# ---------- 2. which channel carries each edge ----------
for tag, name in CH:
    print("\n-- %s (%s): scores for our %d overlaps vs the %d background edges"
          % (tag, name, len(hits), len(SE)))

print()
rows = []
for tag, name in CH:
    hv = [hits[k][tag] for k in hits]
    bv = [SE[k][tag] for k in SE]
    n_hi = sum(1 for v in hv if v > EV)
    n_bg = sum(1 for v in bv if v > EV)
    rows.append((tag, name, sum(hv) / len(hv), sum(bv) / len(bv),
                 n_hi, 100.0 * n_hi / len(hv), n_bg, 100.0 * n_bg / len(bv)))

print("%-8s %-22s %8s %8s %14s %14s" % ("channel", "name", "mean_our", "mean_bg",
                                        "ours>%.3f" % EV, "bg>%.3f" % EV))
for tag, name, mo, mb, n_hi, po, n_bg, pb in rows:
    print("%-8s %-22s %8.4f %8.4f %6d (%5.1f%%) %6d (%5.1f%%)"
          % (tag, name, mo, mb, n_hi, po, n_bg, pb))

# ---------- 3. the decisive split ----------
def classify(d):
    e, dd = d["escore"] > EV, d["dscore"] > EV
    if e or dd:
        return "experimental or curated"
    if d["ascore"] > EV:
        return "co-expression only"
    if d["tscore"] > EV or d["nscore"] > EV or d["fscore"] > EV or d["pscore"] > EV:
        return "other channels only"
    return "below channel floor"


print("\n-- evidence class --")
cats = ["experimental or curated", "co-expression only", "other channels only",
        "below channel floor"]
for lab, pop in (("our overlaps", hits), ("STRING background", SE)):
    cnt = {c: 0 for c in cats}
    for k in pop:
        cnt[classify(pop[k])] += 1
    tot = len(pop)
    print("%-20s" % lab, " | ".join("%s %d (%.1f%%)" % (c, cnt[c], 100.0 * cnt[c] / tot)
                                    for c in cats))

# ---------- 4. per-cancer breakdown of the overlap ----------
by_cancer = {}
for k, cs in CP.items():
    if k in SE:
        for c in cs:
            by_cancer[c] = by_cancer.get(c, 0) + 1
print("\noverlapping pairs by cancer (top 12):")
for c, n in sorted(by_cancer.items(), key=lambda x: -x[1])[:12]:
    print("   %-5s %3d" % (c, n))

# strongest overlaps: highest STRING score
top = sorted(hits.items(), key=lambda kv: -kv[1]["score"])[:15]
print("\nstrongest overlapping pairs:")
for k, v in top:
    print("   %-22s score %.3f | exp %.3f db %.3f text %.3f coexp %.3f | cancers %s"
          % (k, v["score"], v["escore"], v["dscore"], v["tscore"], v["ascore"],
             ",".join(sorted(CP[k]))[:46]))

out = {
    "n_genes": NG, "n_string_edges": len(SE), "n_causal_pairs": len(CP),
    "n_overlap": len(hits),
    "expected_rate_pct": 100 * exp_rate,
    "observed_rate_pct": 100.0 * len(hits) / len(CP),
    "fold_enrichment": (len(hits) / len(CP)) / exp_rate,
    "fisher_or": float(orr), "fisher_p": float(p),
    "channels": [{"tag": t, "name": n, "mean_our": mo, "mean_bg": mb,
                  "n_our_above": nh, "pct_our_above": po,
                  "n_bg_above": nb, "pct_bg_above": pb}
                 for t, n, mo, mb, nh, po, nb, pb in rows],
    "evidence_class": {},
    "overlap_by_cancer": by_cancer,
    "overlaps": {k: {"score": v["score"], "escore": v["escore"], "dscore": v["dscore"],
                     "tscore": v["tscore"], "ascore": v["ascore"], "nscore": v["nscore"],
                     "cancers": sorted(CP[k])} for k, v in hits.items()},
}
for lab, pop in (("our_overlaps", hits), ("string_background", SE)):
    cnt = {c: 0 for c in cats}
    for k in pop:
        cnt[classify(pop[k])] += 1
    out["evidence_class"][lab] = cnt

json.dump(out, open(os.path.join(RES, "_string_channel_analysis.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("\nwrote results\\_string_channel_analysis.json")
