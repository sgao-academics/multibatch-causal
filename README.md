# Panel alignment is a precondition for cross-cancer causal comparison: the reproducible structure across 33 TCGA cohorts is a microenvironmental chemokine axis

**Shuaidong Gao** — Chongqing Institute of Foreign Studies

Replication package for the manuscript *"Panel alignment is a precondition for cross-cancer causal comparison: the reproducible structure across 33 TCGA cohorts is a microenvironmental chemokine axis"*, submitted to *Functional & Integrative Genomics* (Springer).

## What the analysis delivers

The pipeline estimates a NOTEARS graph for each of the 33 TCGA cancer types (10,459 samples) and then asks which edges survive a change of tumour type — a question that can only be put once every cohort is aligned on one common 100-gene panel. It releases four products:

1. **Panel alignment, shown to be a precondition for the comparison.** Under the per-cohort top-100 design the selections barely intersect (mean pairwise overlap 7.4 of 100 genes, mean Jaccard 0.040; 1,775 distinct genes across the 33 lists), so two cohorts' adjacency matrices index different gene pairs at the same position and an element-wise comparison between them is not defined. The element-wise median adjacency matrix is then identically zero: no position clears $|\text{median}| > \tau$ and the largest is 0.048. Holding the panel fixed restores it — six positions clear $\tau = 0.3$ and the largest reaches 0.731.

2. **The recurring structure, named and tested against a null.** Of the 9,900 directed pairs the panel admits, 14 recur in ten or more cohorts. Node relabelling gives a maximum of 0 under the null (permutation $p = 5.0 \times 10^{-3}$), and the collision statistic $Q$ falls from 8,037 to 2,715 ± 32 over 500 sample-label permutations ($z = 166.8$, $p \le 0.002$). The two strongest pairs share their middle node and form one chain, CXCL9→CXCL10 (26/33) and CXCL10→CXCL11 (28/33), and each of the 14 sits on one of six non-malignant axes.

3. **An attribution for what that structure is.** The dominant axis of the aligned panel is microenvironmental content rather than a tumour-cell programme: PC1 has $\lambda_1 = 16.04$ against a Marchenko–Pastur upper edge of 1.205 (13.3×), fifteen eigenvalues lie above the edge where a null matrix leaves none, and PC1 correlates at a mean $r = 0.803$ (range 0.415–0.912), positive in 33 of 33 cohorts, with 23 stromal, immune and endothelial markers drawn deliberately from outside the panel. Canonical drivers cannot enter: their median dispersion percentile is 50.8 against the 0.487th percentile of the panel cut, and they enter systematically only at $k \gtrsim 1000$ (9 selections at $k = 100$, 114 at $k = 1000$).

4. **An operational sample-size rule, measured from a negative control.** Shuffling sample labels abolishes the recurring set (14 → 0) while leaving the sample-size dependence almost unchanged ($r = -0.503$ against $-0.652$), which places the dependence in the fitting regime rather than in the biology. Cohorts that still return edges on shuffled data have a maximum of $n = 156$; cohorts that return none have a minimum of $n = 172$. The two regimes separate at $n^{*} \approx 172 \approx 1.7d$ for $d = 100$.

The revision adds this analysis layer on top of the previous submission. The per-cohort estimation of all 33 cohorts (2,445 edges), the STRING/TRRUST comparison (31.7-fold, Fisher $p = 7.7 \times 10^{-206}$), the DepMap cross-platform validation, the GENIE3 and pooled NOTEARS baselines, the survival analysis, the somatic-alteration profiling and the synthetic two-stage validation are all retained and are reproduced by the stages below.

## Quick Start

```bash
# 1. Download TCGA HiSeqV2 RSEM data from https://xenabrowser.net/
#    Place TCGA_XXX_HiSeqV2.tsv files (33 cancer types) in ./data/
#    (set MULTIBATCH_DATA to read them from somewhere else)
#    Optional, for the DepMap panel of Figure 4:
#      OmicsExpressionProteinCodingGenesTPMLogp1.csv and CRISPRGeneEffect.csv in ./data/depmap/
#    Required for the survival panel of Figure 5:
#      validation/pancan_os/*.json under ./data/ -- one file per cohort, fetched from cBioPortal
#      with scripts/figures/_fetch_pancan_os.py (a few minutes; the TCGA download above is the
#      slow part).  Like the TCGA matrices these records are not redistributed here.
#    Optional, for the BRCA p-values quoted in the survival section:
#      brca_survival.json in ./data/validation/
#    Only for regenerating Supplementary Figure S4 from source -- the derived tables ship with
#    the package, so the figure rebuilds without them; fetching them again needs
#      TIL_abundance.zip (TISIDB) and HM450_gencode.tsv.gz in ./data/immune/
#    (set MULTIBATCH_IMMUNE to read them from somewhere else)
#    Nothing else is needed: the aligned panels of Stage 5b ship in ./data/panels/

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline
#    Stages 1-5 take about 25 min on a single CPU core and skip whatever is already
#    checkpointed; the downstream half of Stage 5b takes about 2 min more.
python run_all.py

#    The three NOTEARS re-runs inside Stage 5b (panel A, panel B and the shuffled
#    control, about 25 min each) are skipped by default because their outputs ship in
#    ./data/panels/ and ./results/.  Set FULL=1 to recompute them from scratch:
#      FULL=1 python run_all.py

# 4. Figures, supplementary figures and tables are generated into ./figures/ and ./supplementary/
#    Set SKIP_FIGURES=1 to reproduce only the analysis stages.

# 5. Compile the supplementary figures.  The manuscript source is not part of
#    this package, which releases the replication material only.
pdflatex supplementary_figures.tex  # -> supplementary_figures.pdf (submitted as ESM_5)
```

`run_all.py` reads the TCGA matrices from `./data/`, or from the folder named by the
`MULTIBATCH_DATA` environment variable. If neither exists it stops with the download
instructions instead of a traceback.

### Numerical reproducibility

The checkpoints in `results/` are the ones behind the reported numbers, and `run_all.py`
reuses a cohort's checkpoint whenever it is already present, so a first run returns the
manuscript values unchanged. Deleting `results/` forces a full recomputation: NOTEARS is
solved with L-BFGS-B inside an augmented-Lagrangian loop, and a different SciPy/NumPy
build can settle on a marginally different solution (in a spot check on one of the 33
cohorts, 183 edges were rebuilt as 186, with 9 of 10,000 entries of the weighted adjacency
matrix changing sign about the threshold; the pooled fit moves from 37 edges to 36).
Use the shipped checkpoints to verify the reported numbers, and a clean run to verify
the code.

The GENIE3 and pooled checkpoints are stored in a compact form — a per-cohort edge list
with the edge budget, and the pooled edge count with $h(W)$ — because that is what the
downstream statistics read. The cache predicates in `run_all.py` recognise both the
compact and the full form, so neither is recomputed when it is already there.

## What This Package Contains

| Directory | Contents |
|:----------|:---------|
| `run_all.py` | One-command reproduction: 5 analysis stages, the aligned-panel analysis of Stage 5b, and the figure/table stage, all checkpointed |
| `scripts/analysis/` | The Stage 5b scripts: panel construction (`build_panel_A.py`, `build_panel_B.py`), the NOTEARS re-runs (`fit_panel_A.py`, `fit_panel_B.py`, `shuffle_control_fit.py`) and the downstream analyses (`identifiability.py`, `decompose_aligned_vs_raw.py`, `recurrence_analysis.py`, `permutation_null.py`, `permutation_null_continuous.py`, `shuffle_control_analysis.py`, `latent_axis.py`, `axis_attribution.py`, `driver_screen.py`, `crossmodal_check.py`, `panel_B_robustness.py`, `verify_numbers.py`) |
| `scripts/figures/` | Data derivation (`_prep_fig_extra.py`, `_prep_tau.py`, `_prep_fig1_landscape.py`), the pan-cancer survival scan (`_km_pancan_scan.py`) and one generator per figure, named after the figure it builds (`gen_fig1_main.py`, `gen_fig2_panels.py`, `gen_fig3_ppi.py` … `gen_fig7_alteration.py`, `gen_figS1_sensitivity.py` … `gen_figS4_immune.py`), plus `_figstyle.py`, the shared drawing language every generator imports (palette, rounded cards, hairline matrix grid, the 174 mm canvas), and the helpers that build no figure of their own (`gen_baseline_stats.py` writes the statistics behind Figure 4b, `gen_km_panel.py` the BRCA Kaplan-Meier panel `figures/km_brca_panel.pdf` and the log-rank values quoted in the text, `_context_robustness.py` the sharing statistics under the alternative counting conventions, and the `_fetch_*`/`_analyze_*`/`_prep_*` scripts the derived tables the generators read) |
| `scripts/figures/_make_supplementary.py` | Builds Tables S1–S4 from the result files |
| `scripts/figures/_gene_symbols.py` | Gene-symbol table imported by the figure scripts. Two hub genes are carried by the source data under symbols HGNC has since replaced (`C9orf84` → `SHOC1`, `MGC29506` → `MZB1`), and the Xena matrices spell the unnamed-reading-frame loci in mixed case; a plain string comparison silently drops those rows, so every lookup goes through this module |
| `data/panels/` | The two aligned-panel definitions, `panel_A_100genes.json` (the common 100-gene panel of Stage 5b, with the per-cohort top-100 lists and the dispersion percentiles behind Figure 1d) and `panel_B_100genes.json` (the gene-disjoint second panel). These are the design, not measurements, and they are the only part of `data/` that is redistributed |
| `supplementary_figures.tex` | LaTeX source of the supplementary figures (S1–S4); compiled with `pdflatex` it reproduces `supplementary/ESM_5.pdf` |
| `supplementary/` | The five Online Resources as submitted: `ESM_1`–`ESM_4.xlsx` (Tables S1–S4) and `ESM_5.pdf` (Figures S1–S4), rebuilt by the last two steps of `run_all.py` |
| `scripts/experiments/` | Self-contained V6 synthetic validation (`_synthetic_v6.py`); it writes `synth_ckpt.json` and `_v6_original_output.json`. `_synth_metrics.py` scores that checkpoint against its ground truth and writes `_synth_metrics.json`, the source of Table 2 |
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
| 5b | Aligned-panel analysis (see below): panel construction, the NOTEARS re-runs, and the downstream analyses the manuscript's second and third contributions rest on | `_shared_decomposition.json`, `_permutation_null.json`, `_recurrence_analysis.json`, `_latent_axis.json`, `_axis_attribution.json`, `_driver_screen.json`, `_shuffle_control.json`, `_identifiability.json`, `_crossmodal_check.json`, `_panel_B_robustness.json`, `_numbers_verification.json` |
| 6b | Figures 1–7, supplementary Figures S1–S4 and Tables S1–S4, plus the acyclicity diagnostic of Section 2.7, the baseline table behind Figure 4b and the structure-recovery metrics of Table 2 | `figures/`, `supplementary/`, `_acyclicity.json`, `_baseline_stats.json`, `_synth_metrics.json` |

All stages are idempotent. Re-running resumes from the last checkpoint.

## Aligned-panel analysis (Stage 5b)

Stage 1 lets each cohort select its own 100 most variable genes, so a fixed position in the
adjacency matrix means a different gene in a different cohort and the comparison the
manuscript is built on is undefined. Stage 5b re-estimates every cohort on one panel
shared by all 33, then runs the analyses that follow from it:

| Script | Output | Reports |
|:-------|:-------|:--------|
| `build_panel_A.py` | `data/panels/panel_A_100genes.json` | the common panel: mean pairwise overlap 7.4/100, Jaccard 0.040, 1,775 distinct genes |
| `build_panel_B.py` | `data/panels/panel_B_100genes.json` | the gene-disjoint second panel (median-MAD ranks 201–2,000) |
| `fit_panel_A.py`, `fit_panel_B.py` | `_shared_panel_notears.json`, `_panel2_notears.json` | the 33-cohort NOTEARS re-runs on each panel |
| `shuffle_control_fit.py` | `_shared_panel_negctl.json` | refit on shuffled sample labels |
| `decompose_aligned_vs_raw.py` | `_shared_decomposition.json` | median matrix: 6 edges / max 0.731 aligned against 0 edges / max 0.048 misaligned |
| `recurrence_analysis.py` | `_recurrence_analysis.json` | the 14 directed pairs recovered in ≥10 cohorts |
| `permutation_null.py`, `permutation_null_continuous.py` | `_permutation_null.json` | node relabelling (null max 0), collision statistic $Q$ ($z = 166.8$ over 500 permutations) |
| `shuffle_control_analysis.py` | `_shuffle_control.json` | 14 → 0 shared edges, $r = -0.503$ against $-0.652$, the $n = 156/172$ separation |
| `latent_axis.py` | `_latent_axis.json` | PC1 against the Marchenko–Pastur edge ($\lambda_1 = 16.04$, 13.3×) |
| `axis_attribution.py` | `_axis_attribution.json` | PC1 against 23 external markers ($r = 0.803$, 33/33) and the driver threshold curve |
| `driver_screen.py` | `_driver_screen.json` | driver dispersion percentiles (median 50.8) and the $k \gtrsim 1000$ entry curve |
| `crossmodal_check.py` | `_crossmodal_check.json` | the copy-number check ($r = -0.320$, $p = 4.4 \times 10^{-27}$, $n = 1{,}078$) |
| `panel_B_robustness.py` | `_panel_B_robustness.json` | the same axes recovered without the primary panel's genes |
| `verify_numbers.py` | `_numbers_verification.json` | the earlier positive results re-derived under $n \ge 200$ |

## Figure Map

The table is in the order the manuscript presents the figures, and the artwork now carries the
manuscript's own number: each `gen_figN_<topic>.py` writes `figures/FigN.pdf`, which is Figure N.
The `\includegraphics` lines of the manuscript source remain the normative statement of which
artwork is which figure, and the middle column records it here so a reader can go from a figure in
the paper to the file that draws it.

That agreement is new. Three panels of the previous submission — the DepMap cross-platform
concordance, the estimator baselines and the pathway over-representation — carried the artwork
numbers 8, 9 and 4 and appeared as manuscript Figures 4, 5 and 6, so names and citation order ran
apart after Figure 3. They are now one figure, Figure 4, drawn by `gen_fig4_checks.py`, and the
seven generators below run in the same order as the text.

| Figure | Generator → artwork | Content |
|:-------|:--------------------|:--------|
| 1 | `gen_fig1_main.py` → `Fig1_main.pdf` | Cross-cohort comparison on the common panel: (a) element-wise median adjacency under both designs, with the same 9,900 off-diagonal entries ranked by magnitude below; (b) edges per cohort against sample size; (c) recurrence of the 14 directed pairs recovered in ten or more cohorts, onto six non-malignant axes; (d) PC1 against the external stromal/immune marker score, with the dispersion threshold at which drivers enter |
| 2 | `gen_fig2_panels.py` → `Fig2_panels.pdf` | Robustness to the choice of gene panel: (a) recurrence of directed pairs on the second, gene-disjoint panel against the primary one; (b) sample-size dependence of edge count for both panels, over all 33 cohorts and restricted to $n \ge 200$ |
| 3 | `gen_fig3_ppi.py` → `Fig3.pdf` | External support of the inferred edges: largest supported components, STRING evidence channels, supported pairs per cancer |
| 4 | `gen_fig4_checks.py` → `Fig4.pdf` | The three checks on the reproducible structure: (a) DepMap cell-line concordance of the eight replicated pairs against their CRISPR co-dependency; (b) the cross-cancer sharing rate of both estimators under each counting convention, and pooled against per-cancer NOTEARS; (c) MSigDB C2 pathway over-representation for LUAD, BRCA, CHOL (the null case) and the pan-cancer network |
| 5 | `gen_fig5_survival.py` → `Fig5.pdf` | Overall-survival association of the per-cohort network hubs: (a) hazard ratio with 95% confidence interval for every cohort in the scan, ordered by log-rank $p$, dot area the patient count; (b) Kaplan–Meier curves for the six hubs whose intervals exclude 1 |
| 6 | `gen_fig6_tissue.py` → `Fig6.pdf` | Tissue specificity of the per-cohort hub genes: (a) Yanai's τ index for the 30 hubs against the 1,745 non-hubs of the same networks; (b) cumulative rank of each hub's own cohort against a uniform null; (c) the 33 cohort-median densities of seven hub genes, each row marked with the cohort it belongs to |
| 7 | `gen_fig7_alteration.py` → `Fig7.pdf` | Somatic alteration burden of the hub genes against the canonical drivers: (a) non-synonymous mutation frequency of each cohort's own hub against the highest canonical driver, as a paired comparison; (b) amplification and homozygous deletion frequencies drawn as points, so a measured zero stays visible; (c) mutation frequency across cohorts |

The previous submission's Figures 1 and 2 — the pan-cancer expression landscape and the per-cancer
edge analysis — are likewise reproduced by generators the package carries, `gen_fig1_landscape.py`
and `gen_fig2_edges.py` (the first reads the expression cache that `_prep_fig1_landscape.py`
rebuilds from the TCGA matrices). The revised Figures 1 and 2 are `Fig1_main.pdf` and
`Fig2_panels.pdf`, and the previous artwork is **not** carried here, because the revised manuscript
cites none of it. Neither generator is part of `run_all.py`; running one writes its artwork back
into `figures/`. `gen_fig1_landscape.py` self-checks when it runs: it recomputes every tumor/normal
test from the per-cancer expression matrices and aborts if fewer than 15 cohorts are testable, if
fewer than 8 hub tiles are significant, or if the shared set departs from the composition it was
drawn for (19 pairs in three or more cohorts, 12 of them on the sex chromosomes).

The remaining panels of the previous submission — the DepMap cross-platform concordance, the
estimator baselines and the pathway over-representation — are likewise reproduced by generators the
package carries, `gen_fig8_depmap.py`, `gen_fig9_baselines.py` and `gen_fig4_enrichment.py`. All of
them have been absorbed into the revised Figure 4, drawn by `gen_fig4_checks.py`, and none of the
three scripts is part of `run_all.py`. Their artwork is **not** carried here, because the revised
manuscript cites none of it; running one writes the file back into `figures/`. One caveat for anyone
doing so: `gen_fig4_enrichment.py` writes `figures/Fig4.pdf`, the file the revised Figure 4 now
occupies, so it overwrites the merged plate — use it only in a scratch copy of the package.

The supplementary figures are shipped as a source file of their own — `supplementary_figures.tex`,
compiled with `pdflatex` to `supplementary_figures.pdf` and submitted as Online Resource 5 — and
here the numbering does follow the manuscript:

| Figure | Generator → artwork | Content |
|:-------|:--------------------|:--------|
| S1 | `gen_figS1_sensitivity.py` → `FigS1.pdf` | Parameter sensitivity to λ₁, τ and the per-cancer edge-count distribution |
| S2 | `gen_figS2_synthetic.py` → `FigS2.pdf` | Two-stage decomposition on a synthetic ground-truth DAG |
| S3 | `gen_figS3_coexpression.py` → `FigS3.pdf` | Co-expression structure of the STRING-supported network |
| S4 | `gen_figS4_immune.py` → `FigS4.pdf` | Immune-microenvironment association of each cohort's own hub gene |

Every generator is named after the artwork it builds — `gen_figN_<topic>.py` writes
`figures/FigN.pdf` — so a name never has to be guessed and `run_all.py` can be read straight
down.

Every generator draws on a 6.85 in (174 mm) canvas with 8 pt base type, the width and the type
size of the journal's single-column text block, and puts nothing but a lower-case part letter
inside the artwork: the caption lives in the manuscript, and a figure placed at `\textwidth`
then prints at the size it was drawn at, so the lettering keeps the point size its generator
declared.

Figure 1 needs `_prep_fig_extra.py` and `_fix_family_key.py` to have been run first (they
document the composition of the shared pair set); it reads
`results/_recurrence_analysis.json`, `results/_axis_attribution.json`,
`results/_shuffle_control.json` and `data/panels/panel_A_100genes.json`.
`_prep_fig1_landscape.py` is needed only by the pre-revision `gen_fig1_landscape.py`. Figure 7 needs
`_fetch_hub_variants.py` (the cBioPortal download that writes `results/_hub_variant_landscape.json`)
and then `_prep_fig_variant_landscape.py`, and the DepMap half of Figure 4a needs the two
`./data/depmap/` tables named in Quick Start; Figure 3 and Figure S3 both need
`_fetch_string_channels.py` (the STRING v12 per-channel query that writes
`results/_string_channels.json`) and `_analyze_string_channels.py`, and Figure S3 additionally needs
`_prep_fig_corr.py`. The cross-cancer sharing statistics carry a convention the text states but does
not tabulate, so `_context_robustness.py` writes `results/_context_robustness.json` with the directed
and undirected counts, the edge-weighted reuse rate, the sample-size cut-offs and the pairwise
Jaccard, so none of those choices has to be taken on trust. Figure S4 is
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

* `gen_fig4_checks.py` re-reads the eight DepMap edges and aborts unless there are eight of them, exactly six are paralogous family members, and those six carry the six highest expression correlations; it aborts again unless the five named LUAD/BRCA gene sets are present, unless no CHOL set reaches $p < 0.05$, and unless both global sets the caption names are found.
* `gen_fig5_survival.py` rescans all 33 cohorts, includes a panel only when its hub reaches log-rank $p < 0.05$, and prints the count (6 of 33) alongside the binomial enrichment, so the figure cannot drift from the numbers in the text.
* `gen_km_panel.py` asserts the four BRCA log-rank p-values quoted in the survival section.
* `gen_baseline_stats.py` rebuilds the whole of the baseline comparison from the three checkpoints and writes `results/_baseline_stats.json`; the GENIE3 and pooled numbers quoted in the text and drawn in Figure 4b come from that file, not from the progress lines printed by Stage 3.
* `_acyclicity_audit.py` recomputes the achieved $|h(W)|$ and the cyclic edges of every cohort from `_pipeline_notears.json`, and writes `results/_acyclicity.json`. It exists because NOTEARS returns the least-violating iterate within its iteration budget rather than an exactly acyclic matrix: the paper states the violation it actually reaches (16 of 33 thresholded graphs acyclic, 109 of 2,445 edges on a directed cycle) instead of asserting that it is zero.
* `scripts/analysis/verify_numbers.py` is the Stage 5b counterpart: it re-derives the earlier positive results under $n \ge 200$ and writes `results/_numbers_verification.json`.

## Data Availability

TCGA gene expression data are publicly available from the [UCSC Xena browser](https://xenabrowser.net/)
under the HiSeqV2 RSEM pipeline, and overall-survival records from the TCGA PanCancer Atlas studies on
[cBioPortal](https://www.cbioportal.org/) (`_fetch_pancan_os.py` records the exact endpoints used).
DepMap 23Q2 expression and CRISPR data are available from the
[DepMap portal](https://depmap.org/). The 33 cancer types analyzed and their sample sizes are listed
in Table 1 of the manuscript. The raw TCGA matrices are not redistributed here; the only files under
`data/` that ship with the package are the two aligned-panel definitions in `data/panels/`, which
record which genes every cohort was fitted on rather than any measurement.

## Related

The causal discovery engine used in this paper is available as a standalone Python package:

**[causalscale](https://github.com/sgao-academics/causalscale)** — Unified causal discovery API with 7 engines (NOTEARS, DAGMA, Low-Rank, Causal Transformer, Cluster-Aware, Multi-Modal, Ensemble). Handles d=30 to genome-scale (19,215 genes). `pip install causalscale`. MIT license.

## License

MIT
