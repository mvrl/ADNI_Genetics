#!/bin/sh

diagnosis='DX_bl' # DX or DX_bl from ADNIMERGE.csv column
control='CN'
delimiter='_'
case='AD'
analysis=$control$delimiter$case

#Here I used plink1.9 as some of the functionalities were not supported in plink1.07
PoplnStrat_path='/home/skh259/LinLab/LinLab/ADNI/GWAS_ADNI/'$analysis'/PoplnStrat/'
QualityControl_path='/home/skh259/LinLab/LinLab/ADNI/GWAS_ADNI/'$analysis'/QualityControl/'
root_path='/home/skh259/LinLab/LinLab/ADNI/GWAS_ADNI/'
cov_path=$root_path$analysis"/data/cov_pheno.txt"

cd $PoplnStrat_path
cp $QualityControl_path"GWAS_1_2_3_clean_"$analysis"12.bed" .
cp $QualityControl_path"GWAS_1_2_3_clean_"$analysis"12.bim" .
cp $QualityControl_path"GWAS_1_2_3_clean_"$analysis"12.fam" .

plink --bfile "GWAS_1_2_3_clean_"$analysis"12" --pca --noweb
#This generates two PCA related files plink.eigenval (containing eigen values for the default
#20 dimension) and plink.eigenvec. The eigenval tells you in order of each PC ( so PC1,PC2….)
# the percentage each eigenvalue contributes to the variance. The eigenvec contains the coordinates
# for each sample. This file has no headers, and is tab seperated and contains the sample name in columns
# one and two, and then subsequently the eigenvals for each PC.

#This selection is based on inspection of egien values . Here we select 5 PCs
echo "FID IID PC1 PC2 PC3 PC4 PC5" > "GWAS_1_2_3_clean_"$analysis"12.pca"
awk '{ print $1,$2,$3,$4,$5,$6,$7 }' plink.eigenvec >> "GWAS_1_2_3_clean_"$analysis"12.pca"

#Now merge the earlier created covariate file with the PCA covariates
python $root_path"cov_creator.py" 

echo "PCA FOR POPULATION STRATIFICATION COMPLETE!!"



