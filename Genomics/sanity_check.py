#sanity check of folds

cv_path = '/mnt/gpfs2_16m/pscratch/nja224_uksr/SKH259/LinLab/ADNI_Genetics/Genomics/data/GWAS/CN_AD/cv_folds/'#fold0'
import os
from collections import Counter
FOLDS =5
for i in range(FOLDS):
    train_fold_path = os.path.join(cv_path,'fold'+str(i),'train','GWAS_1_2_3_clean_CN_AD.fam')
    test_fold_path = os.path.join(cv_path,'fold'+str(i),'test','GWAS_1_2_3_clean_CN_AD.fam')
    with open(train_fold_path,'r') as train:
        train_text = train.read().strip().split('\n')
        train_subs = [l.split(' ')[1] for l in train_text]
    with open(test_fold_path,'r') as test:
        test_text = test.read().strip().split('\n')
        test_subs = [l.split(' ')[1] for l in test_text]
    print(set(train_subs).intersection(set(test_subs)))
    print("check complete for FOLD",i)


