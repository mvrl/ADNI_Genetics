#This script will use the ranked list of GWAS Association results
# logistic_results.assoc.logistic.adjusted and produces top2000_snps.csv

import os
import pandas as pd
import re

N = 2000
data_path = '/mnt/gpfs2_16m/pscratch/nja224_uksr/SKH259/LinLab/ADNI_Genetics/Genomics/data/'

chr_snp =[]
with open(os.path.join(data_path,'logistic_results.assoc.logistic.adjusted'),'r') as infile:
    text = infile.read().strip().split('\n')
    for i in range(1,len(text)):
        line = text[i].strip()
        line = re.sub(' +', ' ', line)
        chr_snp.append(line.split(' ')[0]+'_'+line.split(' ')[1])

top_snps = list(chr_snp[:2000])
df = pd.DataFrame(top_snps,columns=['top_snps'])
df.to_csv(os.path.join(data_path,'top2000_snps.csv'))


