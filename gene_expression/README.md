In the  direction of multimodal learning, the gene expression data (which is the measure of expression of genes assessed through blood samples of subjects) is considered to be used with MRI images. Owing to large numbe of genes in human genome, there are large gene probes at which the gene expression is evaluated. With low sample size, building effective Machine Learning models using such high dimensional data is challenging. Therefore several feature selection techniques should be use. As a easy first step, currently following two approaches have been tried for feature selection.
1. Perform a student's ttest between the desired groups and select top N gene probes which are the most statistically significant locations. Here N=200 is picked for now and the results labeled as "full" are the classification performance for these 200 genetic features. 
2. Refer the literature specifically this: (https://core.ac.uk/download/pdf/206791729.pdf) and use the probes corresponding to the genes that are reported to have some association with Alzheimer's disease.

For current study the TTEST based feature selection was used. 

Previously, gene expression features in ADNI database was used as it is. After coming across the paper "https://www.nature.com/articles/s41598-020-60595-1" it felt necessary to perform following pre-processing before using the data for our use.
1. Only keep the high quality gene expression data for subjects with RIN (RNA integrity number) value is >6.9
2. To reduce the background noise of ANDI, exclude the probes with an intensity value ≤ the median of all gene expression values in 100 or more samples.
3. If there were multiple probes annotated in one gene, then the median value of those was selected.


## TODO:
1. Try Other feature selection methods
2. Parse log to get final result table

