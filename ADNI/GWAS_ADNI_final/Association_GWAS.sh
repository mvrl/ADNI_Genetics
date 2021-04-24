#!/bin/bash

#Here I used plink1.9
diagnosis='DX_bl' # DX or DX_bl from ADNIMERGE.csv column
control='CN'
delimiter='_'
case='AD'
analysis=$control$delimiter$case

association='logistic' # 'logistic' for binary traits, 'linear' for continuous like MMSE
##Paths
root_path='/home/skh259/LinLab/LinLab/ADNI/GWAS_ADNI/'
work_path=$root_path$analysis"/Assoc/"
data_path=$root_path$analysis"/data/GWAS_1_2_3_clean_"$analysis"12"
cov_path=$root_path$analysis"/data/final_cov.txt"
final_path=$root_path$analysis"/data/"

#Utility Scripts path
Assoc_path=$root_path"GWA_tutorial/3_Association_GWAS/"
cd $work_path

###########################################################
### Association analyses ###

cp $final_path"*" .

# Association
# We will be using 5 principal components as covariates in this association analysis. We use the MDS components calculated from the previous tutorial: final_cov.txt.
plink --bfile $data_path --pheno $cov_path --pheno-name DIAG --covar $cov_path --covar-name AGE,PTGENDER,PTEDUCAT,PC1,PC2,PC3,PC4,PC5 --adjust --ci 0.95 --logistic --hide-covar --out $association"_results"
# Note, we use the option -–hide-covar to only show the additive results of the SNPs in the output file.

# Remove NA values, those might give problems generating plots in later steps.
awk '!/'NA'/' $association"_results.assoc."$association > $association"_results.assoc_2."$association

awk '!/'NA'/' $association"_results.assoc."$association".adjusted" > $association"_results.assoc_2."$association".adjusted"
# The results obtained from these GWAS analyses will be visualized in the last step. This will also show if the data set contains any genome-wide significant SNPs.

# Note, in case of a quantitative outcome measure the option --logistic should be replaced by --linear. The use of the --assoc option is also possible for quantitative outcome measures (as metioned previously, this option does not allow the use of covariates).

#################################################################


# Generate Manhattan and QQ plots.

# These scripts assume R >= 3.0.0.
# If you changed the name of the .assoc file , please assign those names also to the Rscripts for the Manhattan and QQ plot, otherwise the scripts will not run.

# The following Rscripts require the R package qqman, the scripts provided will automatically download this R package and install it in /home/{user}/ . Additionally, the scripts load the qqman library and can therefore, similar to all other Rscript on this GitHub page, be executed from the command line.
# This location can be changed to your desired directory

Rscript --no-save $Assoc_path/Manhattan_plot.R
Rscript --no-save $Assoc_path/QQ_plot.R

