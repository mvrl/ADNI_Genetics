# GWAS_ADNI
GWAS analysis for ADNI data.

This is based on the analysis steps provided in https://github.com/MareesAT/GWA_tutorial

For the details it is recommended to read the accompanying paper "A tutorial on conducting Genome-Wide-Association Studies: Quality control and statistical analysis " (https://www.ncbi.nlm.nih.gov/pubmed/29484742)

## Usage (Steps)
1. python3 data_prep.py
2. ./QualityControl.sh
3. ./PoplnStrat.sh
4. python3 cov_creator.py
5. ./Association_GWAS.sh

## Generate Manhattan and QQ plots.
Open R studio with R version >4.0
set cuurent directory to where the results of ./Association_GWAS.sh are located.
eg:

Manhattan Plot

`>setwd('/Users/subashkhanal/Desktop/GWAS_ADNI/results')`

`>install.packages("qqman")`

`>library('qqman')`

`>results_log <- read.table("logistic_results.assoc_2.logistic", head=TRUE)`

`>png("manhattan.png")`

`>manhattan(results_log, chr = "CHR", bp = "BP", p = "P", snp = "SNP", col=c("#EE799F", "#27408B", "#FFB6C1", "#FF00FF", "#666666", "#8B0000", "#141414", "#6495ED","#00CD00", "#BDB76B", "#B452CD", "#00CED1"), suggestiveline = -log10(1e-05), genomewideline = -log10(5e-08),main = "Manhattan plot CN_AD: logistic")`

`>dev.off()`

Q-Plot

`>results_log <- read.table("logistic_results.assoc_2.logistic", head=TRUE)`

`>png("QQ-Plot.png")`

`>qq(results_log$P, main = "Q-Q plot CN_AD: logistic")`

`>dev.off()`
