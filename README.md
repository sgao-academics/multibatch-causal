# Tissue-Specificity of Causal Gene Regulatory Networks Across 33 Cancers

Shuaidong Gao, Chongqing Institute of Foreign Studies

## Quick Start

```bash
# 1. Download TCGA data: https://xenabrowser.net/ (HiSeqV2 RSEM)
#    Place TCGA_XXX_HiSeqV2.tsv files in ./data/

# 2. Install dependencies
pip install numpy scipy pandas matplotlib scikit-learn

# 3. Run the full pipeline (~25 min)
python run_all.py

# 4. Compile the manuscript
pdflatex manuscript.tex && pdflatex manuscript.tex
```

## License

MIT
