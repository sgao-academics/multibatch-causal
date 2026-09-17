# -*- coding: utf-8 -*-
"""Task 2B, data acquisition.

For every one of the 33 TCGA cohorts, pull the somatic-mutation and discrete GISTIC
copy-number status of (a) that cohort's own hub gene and (b) every other per-cohort hub
gene, plus GSTM1 and three canonical drivers as reference, from the TCGA PanCancer Atlas
studies on cBioPortal.

Writes results/_hub_variant_landscape.json.  Resumable: completed cohorts are skipped.
"""
import json, os, socket, sys, time, urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
socket.setdefaulttimeout(120)

BASE = "https://www.cbioportal.org/api"
HDR = {"User-Agent": "Mozilla/5.0", "Content-Type": "application/json", "Accept": "application/json"}
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results", "_hub_variant_landscape.json")

S2STUDY = {
    "ACC": "acc_tcga_pan_can_atlas_2018", "BLCA": "blca_tcga_pan_can_atlas_2018",
    "BRCA": "brca_tcga_pan_can_atlas_2018", "CESC": "cesc_tcga_pan_can_atlas_2018",
    "CHOL": "chol_tcga_pan_can_atlas_2018", "COAD": "coadread_tcga_pan_can_atlas_2018",
    "READ": "coadread_tcga_pan_can_atlas_2018", "DLBC": "dlbc_tcga_pan_can_atlas_2018",
    "ESCA": "esca_tcga_pan_can_atlas_2018", "GBM": "gbm_tcga_pan_can_atlas_2018",
    "HNSC": "hnsc_tcga_pan_can_atlas_2018", "KICH": "kich_tcga_pan_can_atlas_2018",
    "KIRC": "kirc_tcga_pan_can_atlas_2018", "KIRP": "kirp_tcga_pan_can_atlas_2018",
    "LAML": "laml_tcga_pan_can_atlas_2018", "LGG": "lgg_tcga_pan_can_atlas_2018",
    "LIHC": "lihc_tcga_pan_can_atlas_2018", "LUAD": "luad_tcga_pan_can_atlas_2018",
    "LUSC": "lusc_tcga_pan_can_atlas_2018", "MESO": "meso_tcga_pan_can_atlas_2018",
    "OV": "ov_tcga_pan_can_atlas_2018", "PAAD": "paad_tcga_pan_can_atlas_2018",
    "PCPG": "pcpg_tcga_pan_can_atlas_2018", "PRAD": "prad_tcga_pan_can_atlas_2018",
    "SARC": "sarc_tcga_pan_can_atlas_2018", "SKCM": "skcm_tcga_pan_can_atlas_2018",
    "STAD": "stad_tcga_pan_can_atlas_2018", "TGCT": "tgct_tcga_pan_can_atlas_2018",
    "THCA": "thca_tcga_pan_can_atlas_2018", "THYM": "thym_tcga_pan_can_atlas_2018",
    "UCEC": "ucec_tcga_pan_can_atlas_2018", "UCS": "ucs_tcga_pan_can_atlas_2018",
    "UVM": "uvm_tcga_pan_can_atlas_2018",
}

REFERENCE = ["TP53", "PIK3CA", "KRAS"]
EXTRA = {"SHOC1": 158401, "MZB1": 51237}   # HGNC symbols for the two legacy aliases


def api(path, method="GET", payload=None, tries=3):
    data = json.dumps(payload).encode() if payload is not None else None
    last = None
    for a in range(tries):
        try:
            req = urllib.request.Request(BASE + path, headers=HDR, method=method, data=data)
            with urllib.request.urlopen(req) as r:
                return json.loads(r.read().decode("utf-8", "replace"))
        except Exception as e:
            last = e
            time.sleep(1.5 * (a + 1))
    raise last


# ---------------- gene ids ----------------
gid = json.load(open(os.path.join(ROOT, "results", "_hub_gene_ids.json"), encoding="utf-8"))
hubs = gid["hubs"]
mapping = dict(gid["mapping"])
mapping.update(EXTRA)
all_syms = sorted(set(mapping) | set(REFERENCE))
e2s = {mapping[s]: s for s in all_syms if s in mapping}
print("genes to query: %d" % len(e2s))

# ---------------- existing progress ----------------
data = {"hubs": hubs, "genes": e2s, "cohorts": {}}
if os.path.exists(OUT):
    try:
        data = json.load(open(OUT, encoding="utf-8"))
        print("resuming: %d cohorts already done" % len(data.get("cohorts", {})))
    except Exception:
        pass
data["hubs"], data["genes"] = hubs, {str(k): v for k, v in e2s.items()}
done = data.setdefault("cohorts", {})

entrez = sorted(e2s.keys())
todo = [c for c in S2STUDY if c not in done]
print("cohorts remaining: %d" % len(todo))

only = sys.argv[1:] if len(sys.argv) > 1 else None
if only:
    todo = [c for c in todo if c in only]
    print("restricted to: %s" % todo)

for ci, cohort in enumerate(todo, 1):
    study = S2STUDY[cohort]
    try:
        # denominators
        seq = api("/sample-lists/%s_sequenced" % study)
        cna = api("/sample-lists/%s_cna" % study)
        n_seq = len(seq.get("sampleIds", []))
        n_cna = len(cna.get("sampleIds", []))

        # mutations
        muts = api("/molecular-profiles/%s_mutations/mutations/fetch?projection=SUMMARY" % study,
                   "POST", {"entrezGeneIds": entrez, "sampleListId": "%s_sequenced" % study})
        mut_by_gene = {}
        for m in muts:
            g = e2s.get(m.get("entrezGeneId"))
            if g:
                mut_by_gene.setdefault(g, set()).add(m.get("sampleId"))
        # nonsynonymous only (exclude Silent / Intron / 3'UTR / 5'UTR / IGR / RNA)
        SILENT = {"Silent", "Intron", "3'UTR", "5'UTR", "5'Flank", "3'Flank", "IGR",
                  "RNA", "lincRNA", "Splice_Region", "Targeted_Region"}
        mut_ns = {}
        for m in muts:
            g = e2s.get(m.get("entrezGeneId"))
            if g and m.get("mutationType") not in SILENT:
                mut_ns.setdefault(g, set()).add(m.get("sampleId"))

        # CNA
        cna_rows = api("/molecular-profiles/%s_gistic/molecular-data/fetch?projection=SUMMARY" % study,
                       "POST", {"entrezGeneIds": entrez, "sampleListId": "%s_cna" % study})
        amp, homdel, n_cna_gene = {}, {}, {}
        for r in cna_rows:
            g = e2s.get(r.get("entrezGeneId"))
            if not g:
                continue
            v = str(r.get("value"))
            n_cna_gene[g] = n_cna_gene.get(g, 0) + 1
            if v == "2":
                amp.setdefault(g, set()).add(r.get("sampleId"))
            elif v == "-2":
                homdel.setdefault(g, set()).add(r.get("sampleId"))

        done[cohort] = {
            "study": study,
            "n_sequenced": n_seq,
            "n_cna": n_cna,
            "hub": hubs.get(cohort),
            "mut_all": {g: len(s) for g, s in mut_by_gene.items()},
            "mut_nonsyn": {g: len(s) for g, s in mut_ns.items()},
            "amp": {g: len(s) for g, s in amp.items()},
            "homdel": {g: len(s) for g, s in homdel.items()},
            "n_cna_gene": n_cna_gene,
        }
        h = hubs.get(cohort)
        print("[%2d/%2d] %-5s %-38s nseq=%-5d ncna=%-5d  %s mut=%.1f%% amp=%.1f%% homdel=%.1f%%" % (
            ci, len(todo), cohort, study, n_seq, n_cna, h,
            100.0 * done[cohort]["mut_nonsyn"].get(h, 0) / max(1, n_seq),
            100.0 * done[cohort]["amp"].get(h, 0) / max(1, n_cna),
            100.0 * done[cohort]["homdel"].get(h, 0) / max(1, n_cna)))
        json.dump(data, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        time.sleep(0.35)
    except Exception as e:
        print("[%2d/%2d] %-5s FAILED: %s" % (ci, len(todo), cohort, str(e)[:140]))
        time.sleep(2)

print()
print("saved -> %s  (%d/%d cohorts)" % (OUT, len(done), len(S2STUDY)))
