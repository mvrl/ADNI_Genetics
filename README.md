# Alzheimer's Disease Classification Using Genetic Data

Code accompanying the paper **"Alzheimer's Disease Classification Using Genetic Data"**, accepted at the **BIBM 2021 Workshop on Machine Learning and Artificial Intelligence in Bioinformatics and Medical Informatics (MABM 2021)**.

**Contact:** Subash Khanal — <subash.khanal.cs@gmail.com>

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{khanal2021adni,
  title     = {Alzheimer's Disease Classification Using Genetic Data},
  author    = {Khanal, Subash and others},
  booktitle = {IEEE International Conference on Bioinformatics and Biomedicine (BIBM) Workshop on
               Machine Learning and Artificial Intelligence in Bioinformatics and Medical Informatics (MABM)},
  year      = {2021}
}
```

---

## Method Overview

This repository investigates Alzheimer's Disease (AD) classification using two types of genetic data from the [ADNI](http://adni.loni.usc.edu/) (Alzheimer's Disease Neuroimaging Initiative) cohort:

1. **GWAS (Genome-Wide Association Study)** — Single Nucleotide Polymorphism (SNP) data is first subjected to quality control, population stratification correction, and association testing (using PLINK). The top associated SNPs are then used as features for machine learning classifiers.

2. **Gene Expression** — Microarray-based gene expression profiles are pre-processed, and differentially expressed genes are ranked via a t-test / FDR procedure (per cross-validation fold to avoid data leakage). The top-ranked gene expression features are then passed to a gradient-boosted classifier.

3. **Combined (GWAS + Gene Expression)** — A late-fusion approach that combines the top features from both modalities to train a joint classifier.

For all three pipelines, subjects are divided into six binary classification tasks spanning different diagnostic groups:

| Group pair | Description |
|---|---|
| `CN_AD` | Cognitively Normal vs Alzheimer's Disease |
| `CN_EMCI` | Cognitively Normal vs Early MCI |
| `CN_LMCI` | Cognitively Normal vs Late MCI |
| `EMCI_LMCI` | Early MCI vs Late MCI |
| `EMCI_AD` | Early MCI vs Alzheimer's Disease |
| `LMCI_AD` | Late MCI vs Alzheimer's Disease |

Classification is performed using **Gradient Boosted Trees** (scikit-learn / XGBoost) with **RFECV feature selection** and **SMOTE oversampling**, evaluated via 5-fold stratified cross-validation. Model performance is reported as mean balanced accuracy and AUC.

---

## Data Access

This project uses data from **ADNI**, which requires an approved application for access.
Apply at: <https://adni.loni.usc.edu/data-samples/access-data/>

The GWAS pipeline additionally relies on **PLINK 1.9**:
<https://www.cog-genomics.org/plink/>

---

## Repository Structure

```
ADNI_Genetics/
├── GWAS_ADNI/          # GWAS quality control and association analysis (PLINK-based)
├── Genomics/           # ML classification on GWAS SNP features
├── gene_expression/    # ML classification on gene expression features
└── GWAS_Gene_Expr/     # Combined GWAS + gene expression classification
```

---

## Environment Setup

### Prerequisites
- [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- PLINK 1.9 (for GWAS steps only)
- R >= 4.0 with the `qqman` package (for Manhattan/QQ plots only)

### Create and activate the conda environment

```bash
conda env create -f environment.yml
conda activate adni_genetics
```

---

## Pipeline

### Step 1 — GWAS Analysis (`GWAS_ADNI/`)

See [`GWAS_ADNI/README.md`](GWAS_ADNI/README.md) for full details.

```bash
cd GWAS_ADNI/

# 1. Prepare covariate and phenotype files
python3 data_prep.py

# 2. Quality control (SNP/sample missingness, MAF, HWE, relatedness)
./QualityControl.sh

# 3. Population stratification (principal component analysis via PLINK)
./PoplnStrat.sh

# 4. Create covariate file with principal components
python3 cov_creator.py

# 5. Logistic regression association testing
./Association_GWAS.sh
```

The outputs (Manhattan plot, QQ-plot, and ranked SNP list) are written to `GWAS_ADNI/GWAS_results/`.

---

### Step 2 — ML on GWAS Features (`Genomics/`)

```bash
cd Genomics/

# Extract top N associated SNPs per cross-validation fold
python3 topsnp_extractor.py

# Prepare cross-validation folds
python3 cv_folds.py

# Full grid-search training (tune hyperparameters via CV)
python3 ML_for_GWAS.py

# Evaluate best model on held-out test folds
python3 ML_for_GWAS_test.py
```

Edit the path variables at the top of each script (e.g. `data_path`, `results_path`) to match your data location before running.

---

### Step 3 — ML on Gene Expression Features (`gene_expression/`)

```bash
cd gene_expression/

# Rank gene expression probes by t-test within each CV fold
python3 ttest.py

# Prepare cross-validation folds
python3 cv_folds.py

# Full grid-search training
python3 ML_for_GeneExpr.py
```

The notebook `top_genes.ipynb` can be used to inspect and export the top-ranked gene features after training.

> **Note:** `ML_for_GeneExpr_inflated.py` is kept for reference only. It contains an earlier version that incorrectly applied SMOTE before cross-validation splitting, leading to inflated (over-optimistic) results. Do **not** use it for reporting performance.

---

### Step 4 — Combined GWAS + Gene Expression (`GWAS_Gene_Expr/`)

```bash
cd GWAS_Gene_Expr/

# Prepare combined feature matrix
# (open data_prep.ipynb in Jupyter and run all cells)
jupyter notebook data_prep.ipynb

# Train classifier on combined features (full grid search)
python3 ML_for_combined.py

# Evaluate best combined model on test folds
python3 ML_for_combined_test.py
```

Edit path variables (`GWAS_data_path`, `GeneExpr_data_path`, `results_path`) at the top of each script to point to your data directory.

---

## Acknowledgements

The GWAS quality control pipeline follows the tutorial by Marees et al.:
> *A tutorial on conducting Genome-Wide-Association Studies: Quality control and statistical analysis.*
> <https://github.com/MareesAT/GWA_tutorial> | <https://www.ncbi.nlm.nih.gov/pubmed/29484742>

Data used in preparation of this article were obtained from the Alzheimer's Disease Neuroimaging Initiative (ADNI) database (<http://adni.loni.usc.edu>).
