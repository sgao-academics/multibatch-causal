# -*- coding: utf-8 -*-
"""Task 2B, analysis.

Compare the somatic alteration burden of each cohort's hub gene against canonical
driver genes measured in the same cohort, and against the other hub genes.
"""
import json, os, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
d = json.load(open(os.path.join(ROOT, "results", "_hub_variant_landscape.json"), encoding="utf-8"))
hubs, coh = d["hubs"], d["cohorts"]

HUB = sorted(coh)
print("cohorts: %d" % len(HUB))
print()

hdr = ("%-6s %-9s %5s %6s | %6s %6s %6s | %6s %6s %6s" %
       ("coh", "hub", "nseq", "ncna", "hMut", "hAmp", "hDel", "TP53", "PIK3C", "KRAS"))
print(hdr)
print("-" * len(hdr))

rows = []
for c in HUB:
    x = coh[c]
    h, ns, nc = x["hub"], x["n_sequenced"], x["n_cna"]
    f = lambda k, g, n: 100.0 * x[k].get(g, 0) / max(1, n)
    r = dict(cohort=c, hub=h, nseq=ns, ncna=nc,
             hub_mut=f("mut_nonsyn", h, ns), hub_amp=f("amp", h, nc),
             hub_del=f("homdel", h, nc),
             tp53=f("mut_nonsyn", "TP53", ns), pik3ca=f("mut_nonsyn", "PIK3CA", ns),
             kras=f("mut_nonsyn", "KRAS", ns),
             gstm1_mut=f("mut_nonsyn", "GSTM1", ns), gstm1_amp=f("amp", "GSTM1", nc),
             gstm1_del=f("homdel", "GSTM1", nc))
    r["hub_any"] = 100.0 * len(set() | set()) if False else None
    rows.append(r)
    print("%-6s %-9s %5d %6d | %6.1f %6.1f %6.1f | %6.1f %6.1f %6.1f" % (
        c, h, ns, nc, r["hub_mut"], r["hub_amp"], r["hub_del"], r["tp53"], r["pik3ca"], r["kras"]))

# ---- summary statistics ----
import statistics as st
hm = [r["hub_mut"] for r in rows]
tp = [r["tp53"] for r in rows]
pk = [r["pik3ca"] for r in rows]
kr = [r["kras"] for r in rows]
drv = [max(r["tp53"], r["pik3ca"], r["kras"]) for r in rows]

print()
print("median hub mutation rate      : %.2f%%   (mean %.2f%%, max %.2f%%)" % (st.median(hm), st.mean(hm), max(hm)))
print("median TP53 mutation rate     : %.2f%%" % st.median(tp))
print("median PIK3CA mutation rate   : %.2f%%" % st.median(pk))
print("median KRAS mutation rate     : %.2f%%" % st.median(kr))
print("median best-driver rate       : %.2f%%" % st.median(drv))
print()
print("cohorts where hub mut > 5%%   : %d/%d -> %s" % (
    sum(1 for v in hm if v > 5), len(hm), [r["cohort"] + "(" + r["hub"] + ",%.1f%%)" % r["hub_mut"] for r in rows if r["hub_mut"] > 5]))
print("cohorts where hub any-amp>5%% : %d/%d -> %s" % (
    sum(1 for r in rows if r["hub_amp"] > 5), len(rows),
    [r["cohort"] + "(" + r["hub"] + ",%.1f%%)" % r["hub_amp"] for r in rows if r["hub_amp"] > 5]))
print("cohorts where hub del>5%%     : %d/%d" % (sum(1 for r in rows if r["hub_del"] > 5), len(rows)))
print()
print("GSTM1 across cohorts: median mut %.2f%%, median amp %.2f%%, median del %.2f%%" % (
    st.median([r["gstm1_mut"] for r in rows]), st.median([r["gstm1_amp"] for r in rows]),
    st.median([r["gstm1_del"] for r in rows])))
print("GSTM1 cohorts with del > 10%%: %s" % [r["cohort"] + ",%.0f%%" % r["gstm1_del"] for r in rows if r["gstm1_del"] > 10])

# paired test
try:
    from scipy.stats import wilcoxon, mannwhitneyu
    for name, arr in (("TP53", tp), ("PIK3CA", pk), ("KRAS", kr), ("best-driver", drv)):
        w = wilcoxon(hm, arr)
        print("Wilcoxon hub vs %-12s : stat=%.1f  p=%.3g" % (name, w.statistic, w.pvalue))
except Exception as e:
    print("scipy unavailable:", e)

json.dump({"rows": rows,
           "summary": {"median_hub_mut": st.median(hm), "mean_hub_mut": st.mean(hm),
                       "max_hub_mut": max(hm), "median_tp53": st.median(tp),
                       "median_pik3ca": st.median(pk), "median_kras": st.median(kr),
                       "median_best_driver": st.median(drv)}},
          open(os.path.join(ROOT, "results", "_fig_variant_landscape.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print()
print("saved -> results/_fig_variant_landscape.json")
