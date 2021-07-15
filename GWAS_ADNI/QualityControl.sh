#!/bin/bash

#Here I used plink1.9
diagnosis='DX_bl' # DX or DX_bl from ADNIMERGE.csv column
control='CN'
delimiter='_'
case='AD'
analysis=$control$delimiter$case

##HyperParameters
snp_miss_th1=0.2
snp_miss_th2=0.02
ind_miss_th1=0.2
ind_miss_th2=0.02
maf_th=0.05
hwe_th1=1e-6
hwe_th2=1e-10
pihat_th=0.2

FOLD="4" #Which training fold to run GWAS on!
##Paths
root_path="/mnt/gpfs2_16m/pscratch/nja224_uksr/SKH259/LinLab/ADNI_Genetics/Genomics/data/GWAS/CN_AD/cv_folds/fold"$FOLD"/train/"
work_path=$root_path"QualityControl/"
mkdir -p $work_path
data_path=$root_path"GWAS_1_2_3_clean_"$analysis
cov_path=$root_path"cov_pheno.txt"
final_path=$work_path
code_path="/mnt/gpfs2_16m/pscratch/nja224_uksr/SKH259/LinLab/ADNI_Genetics/GWAS_ADNI/"
#Utility Scripts path
QC_path=$code_path"GWA_tutorial/1_QC_GWAS"

# Change directory to a folder on your UNIX device containing all files from GWAS_1_2_3_clean_$analysis
cd $work_path


#Quality control steps
############### START ANALISIS ###############################

### Step 1 ### 

# Investigate missingness per individual and per SNP and make histograms.
plink --bfile $data_path --pheno $cov_path --pheno-name DIAG --missing --noweb

# output: plink.imiss and plink.lmiss, these files show respectively the proportion of missing SNPs per individual and the proportion of missing individuals per SNP.


# Generate plots to visualize the missingness results.
Rscript --no-save  $QC_path/hist_miss.R --noweb

# Delete SNPs and individuals with high levels of missingness, explanation of this and all following steps can be found in box 1 and table 1 of the article mentioned in the comments of this script.
# The following two QC commands will not remove any SNPs or individuals. However, it is good practice to start the QC with these non-stringent thresholds.  
# Delete SNPs with missingness >0.2.
plink --bfile $data_path --pheno $cov_path --pheno-name DIAG --geno $snp_miss_th1 --make-bed --out "GWAS_1_2_3_clean_"$analysis"2" --noweb

# Delete individuals with missingness >0.2.
plink --bfile "GWAS_1_2_3_clean_"$analysis"2" --pheno $cov_path --pheno-name DIAG --mind $ind_miss_th1 --make-bed --out "GWAS_1_2_3_clean_"$analysis"3" --noweb

# Delete SNPs with missingness >0.02.
plink --bfile "GWAS_1_2_3_clean_"$analysis"3" --pheno $cov_path --pheno-name DIAG --geno $snp_miss_th2 --make-bed --out "GWAS_1_2_3_clean_"$analysis"4" --noweb

# Delete individuals with missingness >0.02.
plink --bfile "GWAS_1_2_3_clean_"$analysis"4" --pheno $cov_path --pheno-name DIAG --mind $ind_miss_th2 --make-bed --out "GWAS_1_2_3_clean_"$analysis"5" --noweb

###################################################################
### Step2 ####

# Check for sex discrepancy.
# Subjects who were a priori determined as females must have a F value of <0.2, and subjects who were a priori determined as males must have a F value >0.8. This F value is based on the X chromosome inbreeding (homozygosity) estimate.
# Subjects who do not fulfil these requirements are flagged "PROBLEM" by PLINK.

plink --bfile "GWAS_1_2_3_clean_"$analysis"5" --pheno $cov_path --pheno-name DIAG --check-sex --noweb

# Generate plots to visualize the sex-check results.
Rscript --no-save $QC_path/gender_check.R
# These checks indicate that there is ..

# The following two scripts can be used to deal with individuals with a sex discrepancy.
# Note, please use one of the two options below to generate the bfile hapmap_r23a_6, this file we will use in the next step of this tutorial.

# 1) Delete individuals with sex discrepancy.
grep "PROBLEM" plink.sexcheck| awk '{print$1,$2}'> sex_discrepancy.txt
# This command generates a list of individuals with the status “PROBLEM”.
#plink --bfile "GWAS_1_2_3_clean_"$analysis5 --remove sex_discrepancy.txt --make-bed --out "GWAS_1_2_3_clean_"$analysis6 --noweb
# This command removes the list of individuals with the status “PROBLEM”.

# 2) impute-sex.
plink --bfile "GWAS_1_2_3_clean_"$analysis"5" --pheno $cov_path --pheno-name DIAG --impute-sex --make-bed --out "GWAS_1_2_3_clean_"$analysis"6" --noweb
# This imputes the sex based on the genotype information into your data set.

###################################################
### Step 3 ### 

# Generate a bfile with autosomal SNPs only and delete SNPs with a low minor allele frequency (MAF).

# Select autosomal SNPs only (i.e., from chromosomes 1 to 22).
awk '{ if ($1 >= 1 && $1 <= 22) print $2 }' "GWAS_1_2_3_clean_"$analysis"6".bim > snp_1_22.txt
plink --bfile "GWAS_1_2_3_clean_"$analysis"6" --pheno $cov_path --pheno-name DIAG --extract snp_1_22.txt --make-bed --out "GWAS_1_2_3_clean_"$analysis"7" --noweb

# Generate a plot of the MAF distribution.
plink --bfile "GWAS_1_2_3_clean_"$analysis"7" --pheno $cov_path --pheno-name DIAG --freq --out MAF_check --noweb
Rscript --no-save $QC_path/MAF_check.R

# Remove SNPs with a low MAF frequency.
plink --bfile "GWAS_1_2_3_clean_"$analysis"7" --pheno $cov_path --pheno-name DIAG --maf $maf_th --make-bed --out "GWAS_1_2_3_clean_"$analysis"8" --noweb

# A conventional MAF threshold for a regular GWAS is between 0.01 or 0.05, depending on sample size.


####################################################
### Step 4 ###

# Delete SNPs which are not in Hardy-Weinberg equilibrium (HWE).
# Check the distribution of HWE p-values of all SNPs.

plink --bfile "GWAS_1_2_3_clean_"$analysis"8" --pheno $cov_path --pheno-name DIAG --hardy --noweb
# Selecting SNPs with HWE p-value below 0.00001, required for one of the two plot generated by the next Rscript, allows to zoom in on strongly deviating SNPs. 
awk '{ if ($9 <0.00001) print $0 }' plink.hwe>plinkzoomhwe.hwe
Rscript --no-save $QC_path/hwe.R

# By default the --hwe option in plink only filters for controls.
# Therefore, we use two steps, first we use a stringent HWE threshold for controls, followed by a less stringent threshold for the case data.
plink --bfile "GWAS_1_2_3_clean_"$analysis"8" --pheno $cov_path --pheno-name DIAG --hwe $hwe_th1 --make-bed --out "GWAS_1_2_3_clean_"$analysis"hwe_filter_step1" --noweb

# The HWE threshold for the cases filters out only SNPs which deviate extremely from HWE. 
# This second HWE step only focusses on cases because in the controls all SNPs with a HWE p-value < hwe 1e-6 were already removed
plink --bfile  "GWAS_1_2_3_clean_"$analysis"hwe_filter_step1" --pheno $cov_path --pheno-name DIAG --hwe $hwe_th2 --hwe-all --make-bed --out "GWAS_1_2_3_clean_"$analysis"9" --noweb

# Theoretical background for this step is given in our accompanying article: https://www.ncbi.nlm.nih.gov/pubmed/29484742 .

############################################################
### step 5 ###

# Generate a plot of the distribution of the heterozygosity rate of your subjects.
# And remove individuals with a heterozygosity rate deviating more than 3 sd from the mean.

# Checks for heterozygosity are performed on a set of SNPs which are not highly correlated.
# Therefore, to generate a list of non-(highly)correlated SNPs, we exclude high inversion regions (inversion.txt [High LD regions]) and prune the SNPs using the command --indep-pairwise’.
# The parameters ‘50 5 0.2’ stand respectively for: the window size, the number of SNPs to shift the window at each step, and the multiple correlation coefficient for a SNP being regressed on all other SNPs simultaneously.

plink --bfile "GWAS_1_2_3_clean_"$analysis"9" --pheno $cov_path --pheno-name DIAG --exclude $QC_path/inversion.txt --range --indep-pairwise 50 5 0.2 --out indepSNP --noweb
# Note, don't delete the file indepSNP.prune.in, we will use this file in later steps of the tutorial.

plink --bfile "GWAS_1_2_3_clean_"$analysis"9" --pheno $cov_path --pheno-name DIAG --extract indepSNP.prune.in --het --out R_check --noweb
# This file contains your pruned data set.

# Plot of the heterozygosity rate distribution
Rscript --no-save $QC_path/check_heterozygosity_rate.R

# The following code generates a list of individuals who deviate more than 3 standard deviations from the heterozygosity rate mean.
# For data manipulation we recommend using UNIX. However, when performing statistical calculations R might be more convenient, hence the use of the Rscript for this step:
Rscript --no-save $QC_path/heterozygosity_outliers_list.R

# Output of the command above: fail-het-qc.txt .
# When using our example data/the HapMap data this list contains 2 individuals (i.e., two individuals have a heterozygosity rate deviating more than 3 SD's from the mean).
# Adapt this file to make it compatible for PLINK, by removing all quotation marks from the file and selecting only the first two columns.
sed 's/"// g' fail-het-qc.txt | awk '{print$1, $2}'> het_fail_ind.txt

# Remove heterozygosity rate outliers.
plink --bfile "GWAS_1_2_3_clean_"$analysis"9" --pheno $cov_path --pheno-name DIAG --remove het_fail_ind.txt --make-bed --out "GWAS_1_2_3_clean_"$analysis"10" --noweb


############################################################
### step 6 ###

# It is essential to check datasets you analyse for cryptic relatedness.
# Assuming a random population sample we are going to exclude all individuals above the pihat threshold of 0.2 in this tutorial.

# Check for relationships between individuals with a pihat > 0.2.
plink --bfile "GWAS_1_2_3_clean_"$analysis"10" --pheno $cov_path --pheno-name DIAG --extract indepSNP.prune.in --genome --min $pihat_th --out "pihat_min"$pihat_th --noweb

# The HapMap dataset is known to contain parent-offspring relations. 
# The following commands will visualize specifically these parent-offspring relations, using the z values. 
awk '{ if ($8 >0.9) print $0 }' pihat_min$pihat_th.genome>zoom_pihat.genome

# Generate a plot to assess the type of relationship.
Rscript --no-save $QC_path/Relatedness.R

# The generated plots show a considerable amount of related individuals (explentation plot; PO = parent-offspring, UN = unrelated individuals) in the Hapmap data, this is expected since the dataset was constructed as such.
# Normally, family based data should be analyzed using specific family based methods. In this tutorial, for demonstrative purposes, we treat the relatedness as cryptic relatedness in a random population sample.
# In this tutorial, we aim to remove all 'relatedness' from our dataset.
# To demonstrate that the majority of the relatedness was due to parent-offspring we only include founders (individuals without parents in the dataset).

plink --bfile "GWAS_1_2_3_clean_"$analysis"10" --pheno $cov_path --pheno-name DIAG --filter-founders --make-bed --out "GWAS_1_2_3_clean_"$analysis"11" --noweb

# Now we will look again for individuals with a pihat >0.2.
plink --bfile "GWAS_1_2_3_clean_"$analysis"11" --pheno $cov_path --pheno-name DIAG --extract indepSNP.prune.in --genome --min $pihat_th --out pihat_min0.2_in_founders --noweb
# The file 'pihat_min0.2_in_founders.genome' shows that, after exclusion of all non-founders, only 1 individual pair with a pihat greater than 0.2 remains in the HapMap data.
# This is likely to be a full sib or DZ twin pair based on the Z values. Noteworthy, they were not given the same family identity (FID) in the HapMap data.

# For each pair of 'related' individuals with a pihat > 0.2, we recommend to remove the individual with the lowest call rate. 
plink --bfile "GWAS_1_2_3_clean_"$analysis"11" --pheno $cov_path --pheno-name DIAG --missing --noweb
# Use an UNIX text editor (e.g., vi(m) ) to check which individual has the highest call rate in the 'related pair'. 
# Generate a list of FID and IID of the individual(s) with a Pihat above 0.2, to check who had the lower call rate of the pair.

##THIS WAS THE RESULT FOR THE WHOLE ADNI GWAS WITHOUT CONSIDERATION OF PHENOTYPE
# # inspecting pihat_min0.2_in_founders.genome
# FID1        IID1   FID2        IID2 RT    EZ      Z0      Z1      Z2  PI_HAT PHE       DST     PPC   RATIO           
#      56  057_S_0643    814  057_S_0934 UN    NA  0.2366  0.4809  0.2825  0.5230  -1  0.841746  1.0000 12.1092
#     453  021_S_0159    342  137_S_4466 UN    NA  0.1890  0.6034  0.2076  0.5093  -1  0.833108  1.0000 13.5865
#     591  023_S_0058    620  023_S_0916 UN    NA  0.2227  0.4918  0.2855  0.5314  -1  0.843817  1.0000 11.1905
#     591  023_S_0058    118  023_S_4035 UN    NA  0.2725  0.4922  0.2354  0.4814  -1  0.829118  1.0000  8.8248
#     620  023_S_0916    118  023_S_4035 UN    NA  0.2069  0.5367  0.2564  0.5247  -1  0.840155  1.0000 13.1182
#       3  031_S_4203    165  031_S_4032 UN    NA  0.1894  0.5142  0.2963  0.5534  -1  0.849440  1.0000 15.2902
#      64  024_S_2239     81  024_S_4084 UN    NA  0.2952  0.5419  0.1629  0.4339  -1  0.813273  1.0000  8.5696
#      64  024_S_2239  ADNI3  024_S_6005 UN    NA  0.1894  0.5406  0.2700  0.5403  -1  0.844585  1.0000 13.4884
#      81  024_S_4084  ADNI3  024_S_6005 UN    NA  0.2632  0.5443  0.1925  0.4646  -1  0.822222  1.0000  9.2375
#     179  012_S_4094  ADNI3  002_S_6066 UN    NA  0.0000  0.0000  1.0000  1.0000  -1  1.000000  1.0000      NA
#     396  011_S_4235  ADNI3  011_S_6303 UN    NA  0.0000  0.0000  1.0000  1.0000  -1  1.000000  1.0000      NA
#   ADNI3  024_S_6184  ADNI3  116_S_6119 OT     0  0.0000  0.0000  1.0000  1.0000  -1  1.000000  1.0000      NA
#   ADNI3  037_S_6115  ADNI3  037_S_6125 OT     0  0.0000  0.0000  1.0000  1.0000  -1  1.000000  1.0000      NA
#   ADNI3  168_S_6131  ADNI3  168_S_6492 OT     0  0.2150  0.4886  0.2964  0.5407  -1  0.846675  1.0000 13.4378
#   ADNI3  032_S_6293  ADNI3  032_S_6294 OT     0  0.3411  0.4541  0.2048  0.4318  -1  0.815975  1.0000  7.2101

# Now by inspecting plink.imiss check for the call rates (or N_MISS) and among each pairs of subjects above (IID1, IID2) select IID with higher N_MISS which means lower call rate

# FID1	ID1	N_MISS1	FID2	ID2	N_MISS2	Remove FID	Remove ID
# 56	057_S_0643	7	814	057_S_0934	12	814	057_S_0934
# 453	021_S_0159	39	342	137_S_4466	46	342	137_S_4466
# 591	023_S_0058	8	620	023_S_0916	136	620	023_S_0916
# 591	023_S_0058	8	118	023_S_4035	33	118	023_S_4035
# 620	023_S_0916	136	118	023_S_4035	33	620	023_S_0916
# 3	031_S_4203	203	165	031_S_4032	25	3	031_S_4203
# 64	024_S_2239	58	81	024_S_4084	34	64	024_S_2239
# 64	024_S_2239	58	ADNI3	024_S_6005	45	64	024_S_2239
# 81	024_S_4084	34	ADNI3	024_S_6005	45	ADNI3	024_S_6005
# 179	012_S_4094	32	ADNI3	002_S_6066	54	ADNI3	002_S_6066
# 396	011_S_4235	49	ADNI3	011_S_6303	41	396	011_S_4235
# ADNI3	024_S_6184	27	ADNI3	116_S_6119	52	ADNI3	116_S_6119
# ADNI3	037_S_6115	71	ADNI3	037_S_6125	55	ADNI3	037_S_6115
# ADNI3	168_S_6131	52	ADNI3	168_S_6492	52	ADNI3	168_S_6492
# ADNI3	032_S_6293	78	ADNI3	032_S_6294	19	ADNI3	032_S_6293


# # In our dataset the individuals 165  031_S_4032 and 81  024_S_4084 had the lower call rate.(while inspecting plink.imiss)
# vi 0.2_low_call_rate_pihat.txt
# i 
# 814	057_S_0934
# 342	137_S_4466
# 620	023_S_0916
# 118	023_S_4035
# 620	023_S_0916
# 3	031_S_4203
# 64	024_S_2239
# 64	024_S_2239
# ADNI3	024_S_6005
# ADNI3	002_S_6066
# 396	011_S_4235
# ADNI3	116_S_6119
# ADNI3	037_S_6115
# ADNI3	168_S_6492
# ADNI3	032_S_6293
# Press esc on keyboard!
#:x
# Press enter on keyboard
# In case of multiple 'related' pairs, the list generated above can be extended using the same method as for our lone 'related' pair.

#SAME THING IN CODE:
awk '{ print $1,$2 }' pihat_min0.2_in_founders.genome > pihat_min0.2_in_founders.genome.fam1
awk '{ print $3,$4 }' pihat_min0.2_in_founders.genome > pihat_min0.2_in_founders.genome.fam2
python $code_path"low_callrate_pruning.py" --FOLD $FOLD

# Delete the individuals with the lowest call rate in 'related' pairs with a pihat > 0.2 
plink --bfile "GWAS_1_2_3_clean_"$analysis"11" --pheno $cov_path --pheno-name DIAG --remove 0.2_low_call_rate_pihat.txt --make-bed --out "GWAS_1_2_3_clean_"$analysis"12" --noweb

#cp "GWAS_1_2_3_clean_"$analysis"12".* $final_path

################################################################################################################################

# CONGRATULATIONS!! You've just succesfully completed the first tutorial! You are now able to conduct a proper genetic QC. 

# For the next tutorial, using the script: 2_Main_script_MDS.txt, you need the following files:
# - The bfile "GWAS_1_2_3_clean_"$analysis12 (i.e., "GWAS_1_2_3_clean_"$analysis12.fam,"GWAS_1_2_3_clean_"$analysis12.bed, and "GWAS_1_2_3_clean_"$analysis12.bim)
# - indepSNP.prune.in
