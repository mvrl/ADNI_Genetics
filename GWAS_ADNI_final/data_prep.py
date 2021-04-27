##This script prepares covariate file containing covariates and target phenotype for the
## given Case Control group in the merged GWAS data from ADNI1/2/3. Also extracts the GWAS data for the analysis

#Author: Subash Khanal
import shlex
import os
import subprocess
import pandas as pd
from collections import Counter

diagnosis = 'DX_bl' # DX or DX_bl from ADNIMERGE.csv column
control='CN'
case='AD'
analysis = control+'_'+case

root_path = '/home/skh259/LinLab/LinLab/ADNI/GWAS_ADNI/'

diag_path = os.path.join(root_path,'ADNIMERGE.csv')
analysis_path=os.path.join(root_path,analysis)
data_path = os.path.join(analysis_path,'data')
orig_data_path=root_path+'GWAS_1_2_3_clean'

with open(orig_data_path+'.fam', 'r') as famf:
    t = famf.read().split('\n')[:-1]
fid_iid = [l.split(' ')[0]+' '+ l.split(' ')[1] for l in t]
sub = [l.split(' ')[1] for l in t]

adni_merge = pd.read_csv(diag_path)
my_merge = adni_merge[pd.DataFrame(adni_merge.PTID.tolist()).isin(sub).any(1).values]
my_merge = my_merge[my_merge['VISCODE']=='bl']
my_merge = my_merge[(my_merge[diagnosis]==control) | (my_merge[diagnosis]==case)]
my_merge.to_csv(data_path+'/my_MERGE.csv')

fid_iid = [s for s in fid_iid if s.split(' ')[1] in list(my_merge.PTID)]
sub = [l.split(' ')[1] for l in fid_iid]

print('----------------------------------------------------')
print("ANALYSIS FOR THE CASE CONTROL DISTRIBUTION")
print(Counter(my_merge[diagnosis]))
print('----------------------------------------------------')

header = 'FID IID AGE PTGENDER PTEDUCAT MMSE DIAG'

with open(analysis_path+'/data/cov_pheno.txt','w') as outfile:
    myfid_iid = ''
    text = header + '\n'
    for s in fid_iid:
        AGE = my_merge[my_merge['PTID']==s.split(' ')[1]]['AGE'].item()
        PTGENDER = my_merge[my_merge['PTID']==s.split(' ')[1]]['PTGENDER'].item()
        if PTGENDER == 'Male':
            PTGENDER = 1
        elif PTGENDER == 'Female':
            PTGENDER = 2
        else:
            PTGENDER = 0
        PTEDUCAT = my_merge[my_merge['PTID']==s.split(' ')[1]]['PTEDUCAT'].item()
        MMSE = my_merge[my_merge['PTID']==s.split(' ')[1]]['MMSE'].item()
        DIAG = my_merge[my_merge['PTID']==s.split(' ')[1]][diagnosis].item()
        if DIAG == control:
            DIAG = 1
        elif DIAG == case:
            DIAG = 2
        else:
            DIAG = -9
        
        text = text+ ' '.join([s,str(AGE),str(PTGENDER),str(PTEDUCAT),str(MMSE),str(DIAG)])+'\n'
        myfid_iid = myfid_iid +s+'\n'
    outfile.write(text.strip())

with open(data_path+'/myFID_IIDs.txt','w') as outfile:
    outfile.write(myfid_iid.strip())

print('----------------------------------------------------')
print("PREPARATION OF PHENOTYPE AND COVARIATE FILE COMPLETE")
print('----------------------------------------------------')

os.chdir(data_path)

plink_command = "plink --bfile "+orig_data_path+" --keep myFID_IIDs.txt --make-bed --out GWAS_1_2_3_clean_"+analysis+" --noweb"
process = subprocess.Popen(shlex.split(plink_command))

print('----------------------------------------------------')
print("PREPARATION OF DATA FOR CASE CONTROL ANALYSIS STARTED")
print('----------------------------------------------------')

