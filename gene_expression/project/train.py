#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 14:37:28 2021
This script trains classifiers over ADNI blood gene expression features, 
Computes feature importance and saves SHAP plots for them

It makes use of Xin's code in 
https://github.com/linbrainlab/machinelearning.git

@author: subashkhanal
"""

from config import cfg, HyperParameters
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
from sklearn.preprocessing import label_binarize
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from classifiers import classifier
from utilities import plot_SHAP, plot_ROC, save_results
from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score
import os
import itertools
from imblearn.over_sampling import ADASYN, SMOTE

def train_ADNI(feat_selection,diag,features,clf,estimators,classes,repeat,filtered,extra_feats, data_path,results_path,plots_path):
    groups = classes.split('_')
    df_gene_dx = pd.read_csv(os.path.join(data_path, filtered+'_gene_expr_dx.csv'))  #Unfiltered_gene_expr_dx.csv
    df_gene_dx = df_gene_dx[(df_gene_dx[diag]==groups[0]) | (df_gene_dx[diag]==groups[1])]
    if feat_selection == 'ttest':
        if diag == 'DX_bl': #if diagnosis is based on baseline visit    
            ttest_df = pd.read_csv(os.path.join(cfg.ttest,'t_test_0.05_geneExpr_'+filtered+'_bl.csv')) #ttest with FDR alpha 0.05
        if diag == 'DX': #if diagnosis is based on the mapping based on date of gene expression sample collection visit
            ttest_df = pd.read_csv(os.path.join(cfg.ttest,'t_test_geneExpr_'+filtered+'.csv')) #ttest with FDR alpha 0.05
        important_probes =  ttest_df.sort_values(classes).sort_values(classes+'_c')['Gene'][0:features] #suffix _c to use the FDR corrected p values

    if feat_selection == 'clf':
        a = ['AGE','PTGENDER','APOE4','PTEDUCAT']
        rank_df = pd.read_csv(os.path.join(results_path,'Outer_GradientBoosting_2000_'+classes+'_'+filtered+'_extra_'+diag+'.csv'))
        rank_df = rank_df[~rank_df['Gene'].isin(a)]
        important_probes = rank_df.sort_values('importance')['Gene'][0:features]    

    if extra_feats == 'extra': 
        #Add extra features along with gene experession
        df = df_gene_dx[['AGE','PTGENDER','APOE4',diag,'PTEDUCAT']+list(important_probes)]
    else:
        df = df_gene_dx[[diag]+list(important_probes)]
    cols = list(set(df.columns) - set([diag]))
    
#Counter({'CN': 244, 'Dementia': 113, 'MCI': 377, nan: 10}) #labels distribution (Unfiltered based on RIN)
#{'CN': 118, 'Dementia': 56, 'MCI': 196, nan: 6}) #filtered based on  RIN
#Counter({'CN': 260, 'LMCI': 225, 'EMCI': 215, 'AD': 43, nan: 1}) # Unfiltered, based on baseline diagnosis
    if diag == 'DX':
        df_CN = df[df.DX=='CN'] 
        df_MCI = df[df.DX=='MCI']
        df_AD= df[df.DX=='Dementia']
        
        #possible_classes = ['CN_MCI','CN_Dementia','MCI_Dementia']
        if classes == 'CN_MCI':
            df1 = df_CN
            df2 = df_MCI
            label1 = 'CN'
            label2 = 'MCI'
            
        if classes == 'CN_Dementia':
            df1 = df_CN
            df2 = df_AD
            label1 = 'CN'
            label2 = 'Dementia'
            
        if classes == 'MCI_Dementia':
            df1 = df_MCI
            df2 = df_AD
            label1 = 'MCI'
            label2 = 'Dementia'
        
    if diag == 'DX_bl':
        df_CN = df[df['DX_bl']=='CN']
        df_AD = df[df['DX_bl']=='AD']
        df_EMCI = df[df['DX_bl']=='EMCI']
        df_LMCI = df[df['DX_bl']=='LMCI']

        #possible_classes = ['CN_AD','CN_EMCI','CN_LMCI', 'EMCI_LMCI','EMCI_AD','LMCI_AD']
        if classes == 'CN_AD':
            df1 = df_CN
            df2 = df_AD
            label1 = 'CN'
            label2 = 'AD'
            
        if classes == 'CN_EMCI':
            df1 = df_CN
            df2 = df_EMCI
            label1 = 'CN'
            label2 = 'EMCI'
            
        if classes == 'CN_LMCI':
            df1 = df_CN
            df2 = df_LMCI
            label1 = 'CN'
            label2 = 'LMCI'
        
        if classes == 'EMCI_LMCI':
            df1 = df_EMCI
            df2 = df_LMCI
            label1 = 'EMCI'
            label2 = 'LMCI'
            
        if classes == 'EMCI_AD':
            df1 = df_EMCI
            df2 = df_AD
            label1 = 'EMCI'
            label2 = 'AD'
            
        if classes == 'LMCI_AD':
            df1 = df_LMCI
            df2 = df_AD
            label1 = 'LMCI'
            label2 = 'AD'

        
    df_sampled = pd.concat([df1, df2],ignore_index=True)
    df_sampled = shuffle(df_sampled, random_state = 42)
    y = df_sampled[diag]
    X = df_sampled.drop(diag,axis=1)
    if extra_feats == 'extra':
        X['PTGENDER'] = X['PTGENDER'].astype('category').cat.codes 
        extra = np.array(X.loc[:, ['AGE','PTGENDER','APOE4','PTEDUCAT']])
        X = X.drop(columns=['AGE','PTGENDER','APOE4','PTEDUCAT'])
        #print(X.shape)
    
    y = label_binarize(y, classes=[label1, label2])
     
    y=y.ravel()
   
    scaler = StandardScaler()
    scaler.fit(X)
    X=scaler.transform(X)
    X = pd.DataFrame(X)
    if extra_feats == 'extra': #Add extra features along with gene experession 
       extra = pd.DataFrame(extra)
       X = pd.concat([X, extra],ignore_index=True, axis=1)
    
    numFeature=X.shape[1]
    mean_fpr = np.linspace(0, 1, 250)
    aucs = np.zeros((repeat,5))
    acc = np.zeros((repeat,5))
    f1 = np.zeros((repeat,5))
    auc_score = np.zeros((repeat,5))
    importance = np.zeros((repeat,5,numFeature))
    TPRS = []
    AUCS = []
    fname = '_'.join([feat_selection,str(estimators),classes,filtered,str(features),extra_feats,diag])
    for i in range(repeat):
        j = 0
        tprs = []
        aucs = []
        model = classifier(clf, estimators)
        cv = StratifiedKFold(n_splits=5,random_state = 42, shuffle = True)

        #oversample = SMOTE(sampling_strategy=1.0, k_neighbors=7) #Incorrect use of SMOTE
        #X, y = oversample.fit_resample(X, y) #Incorrect use of SMOTE
       
        for train, test in cv.split(X, y):
            oversample = SMOTE(sampling_strategy=1.0, k_neighbors=7)
            X_train, y_train = oversample.fit_resample(X.iloc[train], y[train])
            #X_train = X.iloc[train] #Incorrect use of SMOTE
            #y_train = y[train] #Incorrect use of SMOTE
            probas_ = model.fit(X_train, y_train).predict_proba(X.iloc[test])
            #acc[i,j] = model.score(X[test],y[test])
            y_pred = model.predict(X.iloc[test])
            acc[i,j] = balanced_accuracy_score(y[test], y_pred)
            f1[i,j] = f1_score(y[test], y_pred, average='macro')
            auc_score[i,j] = roc_auc_score(y[test], probas_[:, 1])	    	
            importance[i,j,:] =  model.feature_importances_
            #print('for fold',j,' classification report:')
            #print(classification_report(y[test], y_pred, target_names=[label1,label2]))         
            # Compute ROC curve and area under the curve
            fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1],drop_intermediate='False')
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
           
            aucs.append(auc(fpr, tpr))
            
            j += 1
           
        TPRS.append(tprs)
        AUCS.append(aucs)
    
    TPRS = np.array(TPRS).mean(0) #averaged across the repeated experiments
    AUCS = np.array(AUCS).mean(0) #averaged across the repeated experiments 
    
    importance = importance.mean(0).mean(0)# Averaged across repeated experiments and folds  
    acc = acc.mean(1).mean(0) # Averaged across  folds and then over repeated experiments 
    f1 = f1.mean(1).mean(0) # Averaged across  folds and then over repeated experiments
    auc_score = auc_score.mean(1).mean(0)
    #cols = X.columns
    #Plot SHAP and ROC
    #print("DEBUG COMPLETED SUCCESSFULLY")
    #exit(0) #debug point to see if any typos in code before submitting to lcc
    plot_SHAP(model,X,cols,fname,plots_path)
    plot_ROC(TPRS,AUCS,fname,plots_path)
    save_results(acc, f1,auc_score, importance, cols, fname, results_path)
    
    return acc, f1,auc_score, TPRS, AUCS, importance
                     
            
if  __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--feat_selection', type=str, help='Which type of feature selection. Options:[ttest, clf]', default='ttest')
    parser.add_argument('--diag', type=str, help='To select which diagnosis to select. Options:[DX_bl, DX]', default='DX')
    parser.add_argument('--features', type=int, help='Number of features to select', default=200)
    parser.add_argument('--classifier', type=str, help='ML classifier to use. Options:[RandomForestClassifier,GradientBoosting] ', default='GradientBoosting')
    parser.add_argument('--estimators', type=int, help='number of estimators for classfiers', default=100)
    parser.add_argument('--repeat', type=int, help='number of experiments run', default=1)
    parser.add_argument('--classes', type=str, help='Binary classes to perform classification for. Options:[CN_MCI, CN_Dementia, MCI_Dementia, CN_AD,CN_EMCI,CN_LMCI, EMCI_LMCI,EMCI_AD,LMCI_AD]', default='CN_Dementia')
    parser.add_argument('--filtered', type=str, help='Filter type. Options:[Unfiltered, Probfiltered, Bothfiltered]', default='Probefiltered')
    parser.add_argument('--extra_feats', type=str, help='Whether to use extra features like APOE4, AGE, GENDER or only gene expression Options:[extra, no_extra]', default= 'extra')
    parser.add_argument('--tuning', type=str, help='To perform hyperparameter sweep or not. Options:[sweep, no_sweep]', default='no_sweep')    
    args = parser.parse_args()
    
    if args.tuning == 'no_sweep':
        acc, f1, auc_score, tprs, aucs, importance = train_ADNI(feat_selection=args.feat_selection,
                diag = args.diag,
                features = args.features,
                clf = args.classifier,
                estimators = args.estimators,
                classes = args.classes,
                repeat = args.repeat,
                filtered = args.filtered,
                extra_feats = args.extra_feats,
                data_path = cfg.data,
                results_path = cfg.results,
                plots_path = cfg.plots         
               )
    
    if args.tuning == 'sweep':
        params = list(itertools.product(*HyperParameters.params))
        for hp in params:
            print("For parameters:",hp)
            acc, f1,auc_score, tprs, aucs, importance = train_ADNI(feat_selection=args.feat_selection,
                diag = args.diag,
                features = hp[0],
                clf = hp[1],
                estimators = hp[2],
                classes = hp[3],
                repeat = cfg.repeat,
                filtered = hp[4],
                extra_feats = hp[5],
                data_path = cfg.data,
                results_path = cfg.results,
                plots_path = cfg.plots         
               )
            print(acc, f1,auc_score, tprs, aucs, importance)


    
 
