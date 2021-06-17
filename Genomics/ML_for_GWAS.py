from utilities import prepare_targets, data_prep, sequence_parser, GWAS_data_prep, GridSearch, save_results
import os
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.ensemble import GradientBoostingClassifier                             
from imblearn.over_sampling import SMOTENC
from collections import Counter
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, balanced_accuracy_score
from easydict import EasyDict as edict
import itertools
import warnings
warnings.filterwarnings("ignore")
################################################################################################

SEED = 1
np.random.seed(SEED)
RESULTS = 'results' # Name of results folder, 'results' for overall grid search, results_test1, results_test2, results_test3 for 3 rounds of best model and features.
data_path = '/home/skh259/LinLab/LinLab/ADNI_Genetics/Genomics/'
results_path = os.path.join(data_path,RESULTS)
##########################################################################################################
def train_ADNI(groups='CN_AD',features=1000,pruning='prune'):
    
    groups = groups
    print("EXPERIMENT LOG FOR:",groups)
    print('\n')
    
    #Number of top SNPs to take as features
    N = features
    df, y, _ = GWAS_data_prep(groups,data_path,features)
    print("Shape of final data BEFORE FEATURE SELECTION")
    print(df.shape, y.shape)
    STEP = int(df.shape[1]/20)
    fname = '_'.join([groups,str(features),pruning])
    if pruning == 'prune':
        ########################################################################################
        #                       RECURSIVE FEATURE ELIMINATION
        ########################################################################################

        estimator = GradientBoostingClassifier(random_state=SEED, n_estimators=2*df.shape[1])
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
        selector = RFECV(estimator, n_jobs=-1,step=STEP, cv=cv,scoring='balanced_accuracy')
        selector = selector.fit(df, y)
        df = df.loc[:, selector.support_]
    
    print("Shape of final data AFTER FEATURE SELECTION")
    print(df.shape, y.shape)
    print("Label distribution ater final feature selection")
    label_dist = Counter(y)
    print(label_dist)
    final_N = df.shape[1]
    cat_columns = list(set(df.columns) - set(['AGE','EDU']))
    cat_columns_index = [i for i in range(len(df.columns)) if df.columns[i] in cat_columns]
    result = GridSearch(df,y,cat_columns_index,results_path,fname,SEED)
    ###########################################################################################
    #                           FINAL RUN AND SAVE RESULTS
    ###########################################################################################
    tprs = []
    aucs = []
    acc = []
    imp = []
    mean_fpr = np.linspace(0, 1, 100)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    fig, ax = plt.subplots()
    X, y = df, y
    for train, test in cv.split(X, y):
        X_train = X.iloc[train]
        y_train = y[train]
        
        X_test = X.iloc[test]
        y_test = y[test]
        n_estimators = result.best_params_['classifier__n_estimators']
        model = GradientBoostingClassifier(random_state=SEED,n_estimators=n_estimators)
        oversample = SMOTENC(sampling_strategy=0.7, k_neighbors=7, categorical_features = cat_columns_index,random_state=SEED)
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
    print("END OF THE EXPERIMENT\n")

    plt.close('all')
    return final_ACC, final_AUC, final_N, label_dist, n_estimators


if  __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=int, help='Number of features to be used', default=1000)
    parser.add_argument('--pruning', type=str, help='Do pruning of features or not. Options:[prune,no_prune]', default='prune')
    parser.add_argument('--groups', type=str, help='binary classes to be classified ', default='CN_AD')
    parser.add_argument('--tuning', type=str, help='To perform hyperparameter sweep or not. Options:[sweep, no_sweep]', default='no_sweep')    
    args = parser.parse_args()
    
    if args.tuning == 'no_sweep':
        acc, my_auc, final_N, label_dist,n_estimators = train_ADNI(features=args.features,
                pruning=args.pruning,
                groups = args.groups       
               )
    
    HyperParameters = edict()
    HyperParameters.groups =['CN_AD'] 
    HyperParameters.features= [100,200,300,400,500,750,1000,1250,1500]
    HyperParameters.pruning = ['prune','no_prune']
    HyperParameters.params = [HyperParameters.features, HyperParameters.pruning,HyperParameters.groups]  
    if args.tuning == 'sweep':
        final_result = pd.DataFrame(columns = ['Group', 'Label_distribution','initial_feats','Pruning','final_feats','best_n_estimators','Macro_ACC','Macro_AUC'])
        params = list(itertools.product(*HyperParameters.params))
        for hp in params:
            print("For parameters:",hp)
            acc, my_auc, final_N, label_dist,n_estimators = train_ADNI(
                features = hp[0],
                pruning= hp[1],
                groups = hp[2]     
               )
            print(acc, my_auc)
            print('\n')
            final_result = final_result.append({'Group':hp[2], 'Label_distribution':label_dist,
                                                     'initial_feats':hp[0],'Pruning':hp[1],'final_feats':final_N,
                                                'best_n_estimators':n_estimators,'Macro_ACC':acc,'Macro_AUC':my_auc},
                                                ignore_index = True)
        
        final_result.to_csv(os.path.join(results_path,'sweep_results.csv'))

