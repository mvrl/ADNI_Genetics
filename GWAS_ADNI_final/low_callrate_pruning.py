#This script will look for call rates for the pair of individuals that are found to be related
#And selects the individual with high number of miss (low call rate) to be dropped later in the 
#Quality control analysis.
import os
import re

diagnosis='DX_bl' # DX or DX_bl from ADNIMERGE.csv column
control='CN'
delimiter='_'
case='AD'
analysis=control+delimiter+case

root_path='/home/skh259/LinLab/LinLab/ADNI/GWAS_ADNI/'
file1 = os.path.join(root_path,analysis,'QualityControl','pihat_min0.2_in_founders.genome.fam1')
file2 = os.path.join(root_path,analysis,'QualityControl','pihat_min0.2_in_founders.genome.fam2')

ref_file = os.path.join(root_path,analysis,'QualityControl','plink.imiss')

op_file = os.path.join(root_path,analysis,'QualityControl','0.2_low_call_rate_pihat.txt')

with open(file1,'r') as infile1, open(file2,'r') as infile2, open(ref_file,'r') as ref, open(op_file,'w') as outfile:
    f1 = infile1.read().strip().split('\n')[1:]
    f2 = infile2.read().strip().split('\n')[1:]
    ref = ref.read().strip().split('\n')[1:]
    text =''
    for i in range(len(f1)):
        nmiss1 = int([re.sub(' +',' ',line).strip().split(' ')[3] for line in ref if (f1[i].split()[0] in line) & (f1[i].split()[1] in line)][0])
        nmiss2 = int([re.sub(' +',' ',line).strip().split(' ')[3] for line in ref if (f2[i].split()[0] in line) & (f2[i].split()[1] in line)][0])
        if nmiss1 > nmiss2:
            text = text + f1[i].split()[0] + ' '+ f1[i].split()[1] + '\n'
        else:
            text = text + f2[i].split()[0] + ' '+ f2[i].split()[1] + '\n'
            
    outfile.write(text.strip())
            


