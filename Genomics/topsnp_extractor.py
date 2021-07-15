#This script will use the ranked list of GWAS Association results
# logistic_results.assoc.logistic.adjusted and produces top2000_snps.csv

import os
import pandas as pd
import re

def top_snp(N,data_path,final_path):
    chr_snp =[]
    with open(os.path.join(data_path,'logistic_results.assoc.logistic.adjusted'),'r') as infile:
        text = infile.read().strip().split('\n')
        for i in range(1,len(text)):
            line = text[i].strip()
            line = re.sub(' +', ' ', line)
            chr_snp.append(line.split(' ')[0]+'_'+line.split(' ')[1])

    top_snps = list(chr_snp[:N])
    df = pd.DataFrame(top_snps,columns=['top_snps'])
    df.to_csv(os.path.join(final_path,'top2000_snps.csv'))

######################################################################################################
FOLDS = 5
N=2000
cv_path = '/mnt/gpfs2_16m/pscratch/nja224_uksr/SKH259/LinLab/ADNI_Genetics/Genomics/data/GWAS/CN_AD/cv_folds/'#fold0'

for fold in range(FOLDS):
    fold_path = os.path.join(cv_path,'fold'+str(fold))
    assoc_path = os.path.join(fold_path,'train','Assoc')
    top_snp(N,assoc_path,fold_path)
    print("Top SNP extracted for fold:",fold)

print("Top SNP extraction for each fold complete!")




