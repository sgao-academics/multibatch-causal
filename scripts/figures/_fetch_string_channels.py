# -*- coding: utf-8 -*-
"""Pull STRING v12 per-channel scores for every gene pair in the pan-cancer causal network.

The point is falsification, not decoration: if the pairs that STRING "validates" are backed
only by the co-expression channel, then calling them validated is circular, because the causal
search itself runs on expression data.
"""
import os, sys, json, time, urllib.parse, urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results")
BASE = "https://string-db.org/api"
SCORE = 700

COLS = ["stringId_A", "stringId_B", "preferredName_A", "preferredName_B", "ncbiTaxonId",
        "score", "nscore", "fscore", "pscore", "ascore", "escore", "dscore", "tscore"]


def get(path, params, tries=3):
    """POST the parameters: the full identifier list blows past the URL length limit on GET."""
    body = urllib.parse.urlencode(params).encode("utf-8")
    url = BASE + path
    last = None
    for t in range(tries):
        try:
            req = urllib.request.Request(url, data=body,
                                         headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=300) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:
            last = e
            print("   retry %d after %s" % (t + 1, e), flush=True)
            time.sleep(5)
    raise last


# W is a dense weight matrix (every off-diagonal entry is non-zero); reported edges come from
# thresholding it. Mirror exactly what scripts/figures/_prep_fig_extra.py does -- |W| > 0.3 on the
# directed matrix -- so the counts reconcile with _per_cancer_network.json (2445 directed edges).
TAUCUT = 0.3
_ee = json.load(open(os.path.join(RES, "_edge_extract.json"), encoding="utf-8"))

# ---------- 1. our causal edges ----------
D = json.load(open(os.path.join(RES, "_pipeline_notears.json"), encoding="utf-8"))
our_edges = {}          # "A|B" sorted undirected key -> set(cancers)
cancer_edges = {}
tot = 0
for c, v in D.items():
    g = v["genes"]
    W = v["W"]
    n = len(g)
    pairs = []
    nd = 0
    for i in range(n):
        row = W[i]
        for j in range(n):
            if i == j:
                continue
            w = row[j]
            if w is not None and abs(float(w)) > TAUCUT:
                nd += 1
                a, b = g[i], g[j]
                key = "|".join(sorted([a, b]))
                pairs.append(key)
                our_edges.setdefault(key, set()).add(c)
    cancer_edges[c] = pairs
    tot += nd
    print("%-5s genes %4d directed edges %4d" % (c, n, nd))
print("directed edges summed over cancers: %d (registry %s / per_cancer_network 2445)"
      % (tot, _ee.get("total_edges")))

net_genes = sorted({x for v in D.values() for x in v["genes"]})
print("\ntotal cancers %d | unique genes %d | unique causal pairs %d"
      % (len(D), len(net_genes), len(our_edges)))

# ---------- 2. STRING, batched (reuse a previous pull so reruns stay offline) ----------
edges = {}
CACHE = os.path.join(RES, "_string_channels.json")
# Batching silently drops every pair that straddles two batches, so the whole gene list must go
# in one request (STRING accepts up to 2000 identifiers). Tag the cache with the strategy used.
B = 2000
CACHED_B = None
if os.path.exists(CACHE):
    try:
        _old = json.load(open(CACHE, encoding="utf-8"))
        CACHED_B = _old.get("batch")
        if (_old.get("n_genes") == len(net_genes) and _old.get("score_threshold") == SCORE
                and CACHED_B == B):
            edges = _old["string_edges"]
            print("reusing %d STRING edges from cache" % len(edges))
        else:
            print("cache is from a batched pull (batch=%s, need %d) -- refetching"
                  % (CACHED_B, B))
    except Exception as e:
        print("cache unreadable (%s), pulling fresh" % e)

batches = list(range(0, len(net_genes), B)) if not edges else []
for i in batches:
    chunk = net_genes[i:i + B]
    txt = get("/tsv/network", {"identifiers": "\r".join(chunk), "species": 9606,
                               "required_score": SCORE, "caller_identity": "multibatch"})
    lines = [l for l in txt.strip().split("\n") if l.strip()]
    if not lines:
        print("batch %d/%d -> empty" % (i // B + 1, (len(net_genes) + B - 1) // B), flush=True)
        continue
    hdr = lines[0].split("\t")
    n_new = 0
    for ln in lines[1:]:
        p = ln.split("\t")
        d = dict(zip(hdr, p))
        a, b = d.get("preferredName_A"), d.get("preferredName_B")
        if not a or not b:
            continue
        key = "|".join(sorted([a, b]))
        if key in edges:
            continue
        edges[key] = {k: float(d.get(k, 0) or 0) for k in
                      ("score", "nscore", "fscore", "pscore", "ascore", "escore",
                       "dscore", "tscore")}
        edges[key]["a"], edges[key]["b"] = a, b
        n_new += 1
    print("batch %d/%d (%d genes) -> %d new edges, total %d"
          % (i // B + 1, (len(net_genes) + B - 1) // B, len(chunk), n_new, len(edges)), flush=True)
    time.sleep(1.5)

# ---------- 3. how many of our pairs are in STRING at all ----------
hit = sum(1 for k in our_edges if k in edges)
print("\nSTRING high-confidence edges among the %d network genes: %d" % (len(net_genes), len(edges)))
print("our causal pairs: %d | present in STRING at score>=%.0f: %d (%.1f%%)"
      % (len(our_edges), SCORE / 1000.0, hit, 100.0 * hit / max(len(our_edges), 1)))

out = {
    "score_threshold": SCORE,
    "batch": B,
    "n_genes": len(net_genes),
    "n_string_edges": len(edges),
    "n_causal_pairs": len(our_edges),
    "n_causal_pairs_in_string": hit,
    "string_edges": {k: v for k, v in edges.items()},
    "causal_pairs": {k: sorted(v) for k, v in our_edges.items()},
    "n_directed_edges": tot,
    "causal_edges_by_cancer": cancer_edges,
}
json.dump(out, open(os.path.join(RES, "_string_channels.json"), "w", encoding="utf-8"))
print("\nwrote results\\_string_channels.json")
