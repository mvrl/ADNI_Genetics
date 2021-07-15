#This script splits the subjects for 5 fold CV experiments later
import os
import pandas as pd
from collections import Counter
from sklearn.model_selection import StratifiedKFold
import numpy as np
import shlex
import os
import subprocess
import pandas as pd
from collections import Counter

def folds_split(subjects,diag,groups,FOLDS=5):
    col_names = []
    for group in groups:
        for i in range(FOLDS):
            col = ['_'.join([group,'fold'+str(i),'train']),'_'.join([group,'fold'+str(i),'test'])]
            col_names = col_names + col
    df = pd.DataFrame(columns=col_names)
        
    for group in groups:
        folds_subjects = []
        group_diags = group.split('_')
        group_subjects_diag = [(subjects[i],diag[i]) for i in range(len(subjects)) if diag[i] in group_diags]
        X = np.array([i[0] for i in group_subjects_diag])
        y = np.array([i[1] for i in group_subjects_diag])
        print('for:',group)
        print(Counter(y))
        fold = 0
        skf = StratifiedKFold(n_splits=FOLDS, random_state=11,shuffle=True)
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            print('Train label dist',Counter(y_train))
            print('Test label dist',Counter(y_test))
            df['_'.join([group,'fold'+str(fold),'train'])] = pd.Series(X_train)
            df['_'.join([group,'fold'+str(fold),'test'])] = pd.Series(X_test)
            fold += 1
    return df  

data_path = '/mnt/gpfs2_16m/pscratch/nja224_uksr/SKH259/LinLab/ADNI_Genetics/Genomics/data/GWAS/CN_AD'
orig_df = pd.read_csv(os.path.join(data_path,'my_MERGE.csv'))
subjects = orig_df['PTID']
diag = orig_df['DX_bl']
groups = ['CN_AD']
FOLDS = 5

df = folds_split(subjects,diag,groups,FOLDS)
df.to_csv(os.path.join(data_path,'CV_folds_original.csv')) #CV of original data for each folds 
                                                           #without accounting for GWAS steps carried out for each fold
print("Subjects splitted for CV folds")

################################################################################################################################################
#                                       NOW data prep for each CV fold training and test data
################################################################################################################################################

def data_prep_fold(sub,root_path,analysis_path,my_merge,fold,train_test):
    diagnosis = 'DX_bl'
    dest_path = os.path.join(analysis_path,'fold'+str(fold),train_test)
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    
    with open(os.path.join(root_path,'myFID_IIDs.txt'),'r') as infile:
        text = infile.read().strip().split('\n')[1:]
    fid_iid = [l for l in text if l.split(' ')[1] in list(sub)]
    header = 'FID IID AGE PTGENDER PTEDUCAT MMSE DIAG'

    with open(dest_path+'/cov_pheno.txt','w') as outfile:
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

    with open(dest_path+'/myFID_IIDs.txt','w') as outfile:
        outfile.write(myfid_iid.strip())

    print('----------------------------------------------------')
    print("PREPARATION OF PHENOTYPE AND COVARIATE FILE COMPLETE for FOLD",fold,train_test)
    print('----------------------------------------------------')

    os.chdir(dest_path)

    plink_command = "plink --bfile "+root_path+"GWAS_1_2_3_clean_CN_AD"+" --keep myFID_IIDs.txt --make-bed --out GWAS_1_2_3_clean_"+analysis+" --noweb"
    process = subprocess.Popen(shlex.split(plink_command))

    print('----------------------------------------------------')
    print("NOW START QC OF DATA FOR CASE CONTROL ANALYSIS for",fold,train_test)
    print('----------------------------------------------------')

########################################################################################################################################
FOLDS = 5

diagnosis = 'DX_bl' # DX or DX_bl from ADNIMERGE.csv column
control='CN'
case='AD'
analysis = control+'_'+case
group = 'CN_AD'
cv_path = '/mnt/gpfs2_16m/pscratch/nja224_uksr/SKH259/LinLab/ADNI_Genetics/Genomics/data/GWAS/CN_AD/cv_folds/'#fold0'
root_path = '/mnt/gpfs2_16m/pscratch/nja224_uksr/SKH259/LinLab/ADNI_Genetics/Genomics/data/GWAS/CN_AD/'

diag_path = os.path.join(root_path,'my_MERGE.csv')
my_merge = pd.read_csv(diag_path)
analysis_path=os.path.join(root_path,analysis)
data_path = os.path.join(analysis_path)
orig_data_path=root_path+'GWAS_1_2_3_clean_CN_AD'

for fold in range(FOLDS):
    #train_subs = df['_'.join([group,'fold'+str(fold),'train'])]
    #data_prep_fold(train_subs,root_path,cv_path,my_merge,fold,'train')
    test_subs = df['_'.join([group,'fold'+str(fold),'test'])]
    data_prep_fold(test_subs,root_path,cv_path,my_merge,fold,'test')
    print("Train test data prepared for fold:",fold)

print("Data Split and GWAS data prep for all folds complete!")

