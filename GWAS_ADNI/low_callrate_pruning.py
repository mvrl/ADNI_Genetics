#This script will look for call rates for the pair of individuals that are found to be related
#And selects the individual with high number of miss (low call rate) to be dropped later in the 
#Quality control analysis.
import os
import re

def low_call_rate_pruning(FOLD,train_test_flag):

    diagnosis='DX_bl' # DX or DX_bl from ADNIMERGE.csv column
    control='CN'
    delimiter='_'
    case='AD'
    analysis=control+delimiter+case
    root_path="/mnt/gpfs2_16m/pscratch/nja224_uksr/SKH259/LinLab/ADNI_Genetics/Genomics/data/GWAS/CN_AD/cv_folds/fold"+FOLD+"/"+train_test_flag+"/"
    file1 = os.path.join(root_path,'QualityControl','pihat_min0.2_in_founders.genome.fam1')
    file2 = os.path.join(root_path,'QualityControl','pihat_min0.2_in_founders.genome.fam2')

    ref_file = os.path.join(root_path,'QualityControl','plink.imiss')

    op_file = os.path.join(root_path,'QualityControl','0.2_low_call_rate_pihat.txt')

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

if  __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--FOLD', type=str, help='which training fold to do GWAS for', default='0')
    parser.add_argument('--train_test_flag', type=str, help='for train fold or test fold', default='train')
    args = parser.parse_args()

    low_call_rate_pruning(args.FOLD,args.train_test_flag)



