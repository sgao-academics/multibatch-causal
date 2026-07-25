# Tissue-Specificity of Causal Gene Regulatory Networks Across 33 Cancers

**Shuaidong Gao** — Chongqing Institute of Foreign Studies

Replication package for the manuscript submitted to *Theory in Biosciences* (Springer).

## Quick Start

```bash
# 1. Download TCGA HiSeqV2 RSEM data from https://xenabrowser.net/
#    Place TCGA_XXX_HiSeqV2.tsv files (33 cancer types) in ./data/

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline (~25 min on a laptop CPU)
python run_all.py

# 4. Figures are generated automatically in ./figures/
```

## What This Repository Contains

| Directory | Contents |
|:----------|:---------|
| `run_all.py` | One-command reproduction: 6 stages with checkpointing |
| `scripts/figures/` | Figure generation (gen_fig1.py through gen_fig4.py) |
| `results/` | Pre-computed checkpoints (9 JSON files, 7.9 MB total) |
| `figures/` | Pre-built figures (PDF + PNG, 8 files) |
| `refs.bib` | All 36 references in BibTeX format |
| `sn-jnl.cls` | Springer Nature LaTeX class file |
| `requirements.txt` | Python dependencies (numpy, scipy, pandas, matplotlib, scikit-learn) |

## Pipeline Stages

| Stage | Description | Checkpoint |
|:------|:------------|:-----------|
| 1 | Per-cancer NOTEARS (33 cancers, L-BFGS-B) | `_pipeline_notears.json` |
| 2 | Cross-cancer gene-pair analysis | `_pipeline_genepair.json` |
| 3 | GENIE3 baseline | `_genie3_lbfgs_ckpt.json` |
| 4 | Pooled NOTEARS | `_pipeline_pooled.json` |
| 5 | Synthetic validation (two-stage decomposition) | `synth_ckpt.json` |
| 6 | Figure generation | 4 figures in `figures/` |

All stages are idempotent. Re-running resumes from the last checkpoint.

## Data Availability

TCGA gene expression data are publicly available from the [UCSC Xena browser](https://xenabrowser.net/) under the HiSeqV2 RSEM pipeline. The 33 cancer types analyzed and their sample sizes are listed in Table 1 of the manuscript.

## Related

The causal discovery engine used in this paper is available as a standalone Python package:

**[causalscale](https://github.com/sgao-academics/causalscale)** — Unified causal discovery API with 7 engines (NOTEARS, DAGMA, Low-Rank, Causal Transformer, Cluster-Aware, Multi-Modal, Ensemble). Handles d=30 to genome-scale (19,215 genes). `pip install causalscale`. MIT license.

## License

MIT
