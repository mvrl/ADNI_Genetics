# This script takes the features extracted by best models for ML with gene expression and GWAS and just runs ML on those.
from utilities import GRID_search,save_results,RFE
import os
import pandas as pd
from collections import Counter
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE, SMOTENC
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve,roc_auc_score, balanced_accuracy_score
from easydict import EasyDict as edict
import itertools
import warnings
warnings.filterwarnings("ignore")


##########################################################################################################
SEED = 1
SAMPLING = 0.7 #For SMOTENC
np.random.seed(SEED)
results_path = '/mnt/gpfs2_16m/pscratch/nja224_uksr/SKH259/LinLab/ADNI_Genetics/GWAS_Gene_Expr/results_test/'
expr_df = pd.read_csv('/mnt/gpfs2_16m/pscratch/nja224_uksr/SKH259/LinLab/ADNI_Genetics/GWAS_Gene_Expr/data/common_geneExpr.csv').drop(columns=['Unnamed: 0']).reset_index(drop=True)
GWAS_df = pd.read_csv('/mnt/gpfs2_16m/pscratch/nja224_uksr/SKH259/LinLab/ADNI_Genetics/GWAS_Gene_Expr/data/common_gwas_expanded.csv').drop(columns=['Unnamed: 0']).reset_index(drop=True)
combined_df = pd.read_csv('/mnt/gpfs2_16m/pscratch/nja224_uksr/SKH259/LinLab/ADNI_Genetics/GWAS_Gene_Expr/data/common_combined_expanded.csv').drop(columns=['Unnamed: 0']).reset_index(drop=True)

data_type = ['expr','gwas','combined','pruned_combined']
for data in data_type:
    print("Experiment for:",data)
    if data == 'expr':
        my_df = expr_df
        cat_columns_index = []
    if data == 'gwas':
        my_df = GWAS_df
        cat_columns = [col for col in my_df.columns if 'rs' in col or 'GENDER' in col] 
    if data =='combined':
        my_df = combined_df
        cat_columns = [col for col in my_df.columns if 'rs' in col or 'GENDER' in col]   
    
    if data != 'pruned_combined':
        y = my_df['DIAG']
        df = my_df.drop(columns=['PTID','DIAG']).reset_index(drop=True) #Patient ID and DIAG not needed

    if data =='pruned_combined':
        my_df = combined_df
        y = my_df['DIAG']
        df = my_df.drop(columns=['PTID','DIAG','Unnamed: 0']).reset_index(drop=True) #Patient ID and DIAG not needed
        STEP = int(df.shape[1]/20)
        df, y = RFE(df,y,STEP,SEED)
        cat_columns = [col for col in my_df.columns if 'rs' in col or 'GENDER' in col]  
    if data != 'expr':
        cat_columns_index = [i for i in range(len(df.columns)) if df.columns[i] in cat_columns]
    final_N = df.shape[1]
    fname = data
    result = GRID_search(df,y,cat_columns_index,results_path,fname,SEED)
    ###########################################################################################
    #                           FINAL RUN AND SAVE RESULTS
    ###########################################################################################
    tprs = []
    aucs = []
    acc = []
    imp = []
    mean_fpr = np.linspace(0, 1, 100)
    cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=SEED)
    fig, ax = plt.subplots()
    X, y = df, y
    for train, test in cv.split(X, y):
        X_train = X.iloc[train]
        y_train = y[train]
        
        X_test = X.iloc[test]
        y_test = y[test]
        n_estimators = result.best_params_['classifier__n_estimators']
        model = GradientBoostingClassifier(random_state=SEED,n_estimators=n_estimators)
        if len(cat_columns_index) > 0:
            oversample = SMOTENC(sampling_strategy=0.7, k_neighbors=3, categorical_features = cat_columns_index,random_state=SEED)
        else:
            oversample = SMOTE(sampling_strategy=0.7, k_neighbors=3,random_state=SEED)
            
        X_train, y_train = oversample.fit_resample(X_train, y_train)
        probas_ = model.fit(X_train, y_train).predict_proba(X_test)
        y_pred = model.predict(X_test)
        acc.append(balanced_accuracy_score(y_test, y_pred))
        fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1],drop_intermediate='False')
        roc_auc = roc_auc_score(y_test, probas_[:, 1])
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(roc_auc)
        imp.append(model.feature_importances_)

    final_ACC = sum(acc)/len(acc)
    final_AUC = sum(aucs)/len(aucs)
    save_results(X,ax,imp,tprs, mean_fpr,aucs,acc,results_path,final_N,fname)
    print("Done for",data)
    print("for "+data)
    print("ACC:",final_ACC)
    print("AUC:",final_AUC)
    plt.close('all')
print("END OF THE EXPERIMENT\n")


