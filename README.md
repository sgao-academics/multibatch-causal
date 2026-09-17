# Edge-level causal graph comparison across 33 TCGA cohorts reveals tissue-specific regulatory structure and prognostic hub genes

**Shuaidong Gao** — Chongqing Institute of Foreign Studies

Replication package for the manuscript submitted to *Functional & Integrative Genomics* (Springer).

## Quick Start

```bash
# 1. Download TCGA HiSeqV2 RSEM data from https://xenabrowser.net/
#    Place TCGA_XXX_HiSeqV2.tsv files (33 cancer types) in ./data/
#    (set MULTIBATCH_DATA to read them from somewhere else)
#    Optional, for the DepMap panel of Figure 9:
#      OmicsExpressionProteinCodingGenesTPMLogp1.csv and CRISPRGeneEffect.csv in ./data/depmap/
#    Required for the survival panel of Figure 6:
#      validation/pancan_os/*.json in ./data/  (33 cohorts, shipped with this package;
#      regenerate from cBioPortal with scripts/figures/_fetch_pancan_os.py)
#    Optional, for the BRCA p-values quoted in the survival section:
#      brca_survival.json in ./data/validation/
#    Only for regenerating Supplementary Figure S4 from source -- the derived tables ship with
#    the package, so the figure rebuilds without them; fetching them again needs
#      TIL_abundance.zip (TISIDB) and HM450_gencode.tsv.gz in ./data/immune/
#    (set MULTIBATCH_IMMUNE to read them from somewhere else)

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline (~25 min on a laptop CPU for stages 1-5)
python run_all.py

# 4. Figures, supplementary figures and tables are generated into ./figures/ and ./supplementary/
#    Set SKIP_FIGURES=1 to reproduce only the analysis stages.

# 5. Compile the supplementary figures.  The manuscript source is not part of
#    this package, which releases the replication material only.
pdflatex supplementary_figures.tex  # -> supplementary_figures.pdf (submitted as ESM_5)
```

`run_all.py` reads the TCGA matrices from `./data/`, or from the folder named by the
`MULTIBATCH_DATA` environment variable. If neither exists it stops with the download
instructions instead of a traceback.

## What This Package Contains

| Directory | Contents |
|:----------|:---------|
| `run_all.py` | One-command reproduction: 5 analysis stages plus the figure/table stage, all checkpointed |
| `scripts/figures/` | Data derivation (`_prep_fig1_landscape.py`, `_prep_fig_extra.py`, `_prep_tau.py`), the pan-cancer survival scan (`_km_pancan_scan.py`) and one generator per figure (`gen_fig1_landscape.py`, `gen_fig1.py` … `gen_fig8.py`, `gen_fig_bio.py`, `gen_km_pancan.py`, `gen_km_panel.py`) |
| `scripts/figures/_make_supplementary.py` | Builds Tables S1–S4 from the result files |
| `scripts/figures/_gene_symbols.py` | Gene-symbol table imported by the figure scripts. Two hub genes are carried by the source data under symbols HGNC has since replaced (`C9orf84` → `SHOC1`, `MGC29506` → `MZB1`), and the Xena matrices spell the unnamed-reading-frame loci in mixed case; a plain string comparison silently drops those rows, so every lookup goes through this module |
| `supplementary_figures.tex` | LaTeX source of the supplementary figures (S1–S4); compiled with `pdflatex` it reproduces `supplementary/ESM_5.pdf` |
| `supplementary/` | The five Online Resources as submitted: `ESM_1`–`ESM_4.xlsx` (Tables S1–S4) and `ESM_5.pdf` (Figures S1–S4), rebuilt by the last two steps of `run_all.py` |
| `scripts/experiments/` | Self-contained V6 synthetic validation (`_synthetic_v6.py`); it writes `synth_ckpt.json` and `_v6_original_output.json` |
| `results/` | Pre-computed checkpoints and derived data tables. The 46 MB expression cache `_fig1_landscape.npz` is deliberately not shipped; `_prep_fig1_landscape.py` rebuilds it from the TCGA files |
| `figures/` | Pre-built figures, vector PDF |
| `refs.bib` | The 91 references of the manuscript in BibTeX format |
| `sn-jnl.cls` | Springer Nature LaTeX class file |
| `requirements.txt` | Python dependencies |

## Pipeline Stages

| Stage | Description | Checkpoint |
|:------|:------------|:-----------|
| 1 | Per-cancer NOTEARS (33 cancers, L-BFGS-B) | `_pipeline_notears.json` |
| 2 | Cross-cancer gene-pair analysis | `_pipeline_genepair.json` |
| 3 | GENIE3 baseline | `_genie3_lbfgs_ckpt.json` |
| 4 | Pooled NOTEARS | `_pipeline_pooled.json` |
| 5 | Synthetic validation (self-contained V6 two-stage pipeline) | `synth_ckpt.json`, `_v6_original_output.json` |
| 6b | Figures 1–9, supplementary Figures S1–S4 and Tables S1–S4, plus the acyclicity diagnostic of Section 2.7 and the baseline table behind Figure 3 | `figures/`, `supplementary/`, `_acyclicity.json`, `_baseline_stats.json` |

All stages are idempotent. Re-running resumes from the last checkpoint.

## Figure Map

| Figure | Generator | Content |
|:-------|:----------|:--------|
| 1 | `gen_fig1_landscape.py` | Pan-cancer expression landscape: GSTM1 (most recurrent network gene) across 33 cohorts, tumor versus adjacent normal for each cohort's own hub gene (18 cohorts), and cross-cancer sharing of the inferred gene-pairs |
| 2 | `gen_fig2.py` | Pan-cancer edge analysis: per-cancer edge counts, reuse rates, sharing distribution, edges vs sample size, 33 × 33 pairwise sharing map, composition of the shared set |
| 3 | `gen_baseline_stats.py` → `gen_fig3.py` | Baseline comparisons: the cross-cancer sharing rate of both estimators under each counting convention, the overlap between their edge sets, and pooled against per-cancer NOTEARS |
| 4 | `gen_fig_ppi.py` | External support of the inferred edges: largest supported components, STRING evidence channels, supported pairs per cancer |
| 5 | `gen_km_pancan.py` | Pan-cancer hub survival: all 33 cohorts scanned on their own hub gene, showing the six that reach significance, with hazard ratios and 95% confidence intervals |
| 6 | `gen_fig7.py` | Tissue specificity of the per-cancer hub genes (τ index, own-cancer rank, seven expression profiles) |
| 7 | `gen_fig_variant_landscape.py` | Somatic alteration burden of the hub genes against the canonical drivers |
| 8 | `gen_fig_bio.py` | MSigDB C2 pathway enrichment for LUAD, BRCA, CHOL (null case) and the pan-cancer network |
| 9 | `gen_fig8.py` | DepMap cross-platform expression concordance and its relation to CRISPR co-dependency |

The supplementary figures are shipped as a file of their own (`supplementary_figures.tex`,
compiled to `Supplementary_Figures.pdf`):

| Figure | Generator | Content |
|:-------|:----------|:--------|
| S1 | `gen_fig4.py` | Parameter sensitivity to λ₁ and τ |
| S2 | `gen_fig1.py` | Two-stage decomposition on a synthetic ground-truth DAG |
| S3 | `gen_fig_corr.py` | Co-expression structure of the STRING-supported network |
| S4 | `gen_fig_immune.py` | Immune-microenvironment association of each cohort's own hub gene, with a matched control against the other genes of the same network |

Several generators predate the current numbering, so their filenames carry a number that no longer
matches the figure number; the tables above are therefore keyed on the generator, not the filename.

Figure 1 needs `_prep_fig1_landscape.py` to be run first: it reads the 33 per-cancer HiSeqV2
matrices, extracts the 1,775 genes that occur in any inferred network, and writes
`results/_fig1_landscape.npz` (46 MB, not shipped — regenerate it from the TCGA files instead).
Figure 7 needs `_prep_fig_variant_landscape.py`; Figure 4 and Figure S3 both need
`_analyze_string_channels.py`, and Figure S3 additionally needs `_prep_fig_corr.py`. Figure S4 is
drawn from `results/_immune_corr_all.json`, which `_analyze_immune_all.py` regenerates from the
intermediate tables that ship in this package; those come from `_prep_immune_expr_cnv.py` (TCGA
expression and GISTIC copy number) and `_fetch_meth_samples.py` (PANCAN methylation), while the
matched control in `_analyze_immune_control.py` reads the TCGA expression tables and a TISIDB TIL
download placed in `./data/immune/`.

Development-version PNG copies of the time-course snapshots are not included; the PDFs are the
versions referenced by the manuscript.

## Verification Notes

Several generators recompute a reported quantity from raw data and abort on mismatch, so that a
change in the pipeline cannot silently leave a figure out of step with the text:

* `gen_fig1_landscape.py` recomputes every tumor/normal test from the per-cancer expression matrices and aborts if fewer than 15 cohorts are testable, if fewer than 8 hub tiles are significant, or if the shared set differs from the 12 + 7 split stated in the text.
* `gen_fig8.py` recomputes every Spearman coefficient from the raw DepMap cell-line values before plotting.
* `gen_km_pancan.py` rescans all 33 cohorts, includes a panel only when its hub reaches log-rank $p < 0.05$, and prints the count (6 of 33) alongside the binomial enrichment, so the figure cannot drift from the numbers in the text.
* `gen_km_panel.py` asserts the four BRCA log-rank p-values quoted in the survival section.
* `gen_fig_bio.py` asserts that the named LUAD/BRCA gene sets are present and that no CHOL set reaches p < 0.05.
* `_acyclicity_audit.py` recomputes the achieved $|h(W)|$ and the cyclic edges of every cohort from `_pipeline_notears.json`, and writes `results/_acyclicity.json`. It exists because NOTEARS returns the least-violating iterate within its iteration budget rather than an exactly acyclic matrix: the paper states the violation it actually reaches (16 of 33 thresholded graphs acyclic, 109 of 2,445 edges on a directed cycle) instead of asserting that it is zero.

## Data Availability

TCGA gene expression data are publicly available from the [UCSC Xena browser](https://xenabrowser.net/)
under the HiSeqV2 RSEM pipeline, and overall-survival records from the TCGA PanCancer Atlas studies on
[cBioPortal](https://www.cbioportal.org/) (`_fetch_pancan_os.py` records the exact endpoints used).
DepMap 23Q2 expression and CRISPR data are available from the
[DepMap portal](https://depmap.org/). The 33 cancer types analyzed and their sample sizes are listed
in Table 1 of the manuscript.

## Related

The causal discovery engine used in this paper is available as a standalone Python package:

**[causalscale](https://github.com/sgao-academics/causalscale)** — Unified causal discovery API with 7 engines (NOTEARS, DAGMA, Low-Rank, Causal Transformer, Cluster-Aware, Multi-Modal, Ensemble). Handles d=30 to genome-scale (19,215 genes). `pip install causalscale`. MIT license.

## License

MIT
