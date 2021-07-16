#This script splits the subjects for 5 fold CV experiments later
import os
import pandas as pd
from collections import Counter
from sklearn.model_selection import StratifiedKFold
import numpy as np

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

data_path = '/mnt/gpfs2_16m/pscratch/nja224_uksr/SKH259/LinLab/ADNI_Genetics/gene_expression/data'
orig_df = pd.read_csv(os.path.join(data_path,'Unfiltered_gene_expr_dx.csv'))
subjects = orig_df['Unnamed: 0']
diag = orig_df['DX_bl']
groups = ['CN_AD','CN_EMCI','CN_LMCI','EMCI_LMCI','EMCI_AD','LMCI_AD']
FOLDS = 5

df = folds_split(subjects,diag,groups,FOLDS)
df.to_csv(os.path.join(data_path,'CV_folds.csv'))


