from utilities import GeneExpr_data_prep, save_results,importance_extractor
import os
import pandas as pd
from collections import Counter
from sklearn.model_selection import RepeatedStratifiedKFold,GridSearchCV, StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from imblearn.over_sampling import SMOTE, SMOTENC
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, balanced_accuracy_score
from imblearn.pipeline import Pipeline as imbpipeline
from sklearn.pipeline import Pipeline
from easydict import EasyDict as edict
import itertools
import warnings
warnings.filterwarnings("ignore")

overall_groups = ['CN_AD','CN_EMCI','CN_LMCI','EMCI_LMCI','EMCI_AD','LMCI_AD']
SEED = 11
root_path = '/home/skh259/LinLab/LinLab/ADNI_Genetics/gene_expression/'
RESULTS = 'results2'
results_path=os.path.join(root_path,RESULTS)
############################################################################################
def train_val(X_train, y_train, feature_selection, classifier = 'xgb',smote='correct', sampling=0.7, pruning='prune',step=50,seed=11):
    
    if classifier == 'xgb':
        est = xgb.XGBClassifier(n_estimators=2*X_train.shape[1],max_depth=6,eval_metric='logloss',use_label_encoder=False) #eval_metric='logloss' #fixed error warning so!
        clf = xgb.XGBClassifier(eval_metric='logloss',use_label_encoder=False)
        param_grid = {'classifier__n_estimators':range(10,2*X_train.shape[1],50),'classifier__max_depth':range(2,10,2)}
    if classifier == 'GradientBoosting':
        est = GradientBoostingClassifier(n_estimators=2*X_train.shape[1])
        clf = GradientBoostingClassifier()
        param_grid = {'classifier__n_estimators':range(10,2*X_train.shape[1],50)}
          
    if feature_selection == 'RFECV':
        feat_sel = RFECV(estimator=est,step=step,scoring='balanced_accuracy')
    elif feature_selection == 'fromModel':
        feat_sel = SelectFromModel(estimator=est)
    
    if smote == 'correct' and pruning == 'prune':
        pipeline = imbpipeline(steps = [['smote', SMOTE(sampling_strategy=sampling,k_neighbors=7,random_state=seed)],
                                        ['featureSelection',feat_sel],
                                        ['classifier', clf]])
    elif smote == 'incorrect' and pruning == 'prune':
        smote = SMOTE(sampling_strategy=sampling,k_neighbors=7,random_state=seed)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        pipeline = Pipeline(steps = [['featureSelection',feat_sel],['classifier', clf]])
    
    elif pruning == 'no_prune':
        pipeline = imbpipeline(steps = [['smote', SMOTE(sampling_strategy=sampling,k_neighbors=7,random_state=seed)],
                                        ['classifier', clf]])
                                            
    stratified_kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=seed)       
    grid_search = GridSearchCV(estimator=pipeline,
                                param_grid=param_grid,
                                scoring='balanced_accuracy',
                                cv=stratified_kfold,
                                n_jobs=-1,
                                error_score="raise")
    
    grid_search.fit(X_train, y_train)
    cv_score = grid_search.best_score_
    print(cv_score)
    print(grid_search.best_params_)

  # Refit to data using the best parameters
    if classifier == 'xgb':
        model = xgb.XGBClassifier(n_estimators=grid_search.best_params_['classifier__n_estimators'],max_depth=grid_search.best_params_['classifier__max_depth'],eval_metric='logloss',use_label_encoder=False)
    if classifier == 'GradientBoosting':
        model = GradientBoostingClassifier(n_estimators=grid_search.best_params_['classifier__n_estimators'])
        
    mean_fpr = np.linspace(0, 1, 100)
    selector = []
    results = []
    tprs = []
    aucs = []
    acc = []
    imp = []
    summary = dict()
    for train, test in stratified_kfold.split(X_train, y_train):

        X_train_fold = np.array(X_train.iloc[train])
        y_train_fold = np.array(y_train[train])

        X_test_fold = np.array(X_train.iloc[test])
        y_test_fold = np.array(y_train[test])

        oversample = SMOTE(sampling_strategy=sampling, k_neighbors=7,random_state=seed)
        X_train_fold, y_train_fold = oversample.fit_resample(X_train_fold, y_train_fold)

        if pruning == 'prune':
            if feature_selection == 'RFECV':
                selector_fold = pipeline.named_steps['featureSelection'].fit(X_train_fold, y_train_fold).support_ 
            elif feature_selection == 'fromModel':
                selector_fold = pipeline.named_steps['featureSelection'].fit(X_train_fold, y_train_fold).get_support()
            X_train_fold = X_train_fold[:,selector_fold]
            X_test_fold = X_test_fold[:,selector_fold]
        else:
            selector_fold = [True for i in range(len(X_train.columns))] #Take all features no RFECV
            X_train_fold = X_train_fold[:,selector_fold]
            X_test_fold = X_test_fold[:,selector_fold]

        
        probas_ = model.fit(X_train_fold, y_train_fold).predict_proba(X_test_fold)
        y_pred = model.predict(X_test_fold)
        acc = balanced_accuracy_score(y_test_fold, y_pred)
        fpr, tpr, thresholds = roc_curve(y_test_fold, probas_[:, 1],drop_intermediate='False')
        roc_auc = roc_auc_score(y_test_fold, probas_[:, 1])
        
        results.append([acc,roc_auc])
        selector.append(selector_fold)

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        imp.append(model.feature_importances_)
        summary.update({'features':selector,'results':results,'tprs':tprs,'mean_fpr':mean_fpr,'importance':imp})

    return grid_search,summary

def run_ADNI(groups='CN_AD',features=1000,feature_selection='RFECV',classifier = 'xgb',smote='correct',pruning='prune'):
    N = features
    STEP = int(features/20)
    print("EXPERIMENT LOG FOR:",groups)
    print('\n')
    root_path = '/home/skh259/LinLab/LinLab/ADNI_Genetics/gene_expression/'
    fname = '_'.join([groups,feature_selection,classifier,str(features),pruning,smote])

    df, y, SAMPLING = GeneExpr_data_prep(groups,root_path,features)
    print("Shape of final data BEFORE FEATURE SELECTION")
    print(df.shape, y.shape) 
    original_cols = list(df.columns)
    grid_search,summary = train_val(df, y, feature_selection=feature_selection,classifier = classifier,smote=smote, sampling=SAMPLING, pruning=pruning,step=STEP,seed=SEED)
    results = np.array(summary['results'])
    aucs = results[:,1]
    accs = results[:,0]
    acc = np.mean(accs)
    auc = np.mean(aucs)
    imp_df,avg_no_sel_features = importance_extractor(original_cols,summary)

    print("Avg number of features AFTER FEATURE SELECTION")
    print(avg_no_sel_features)
    fig, ax = plt.subplots()
    save_results(ax,imp_df,summary['tprs'],summary['mean_fpr'],aucs,accs,results_path,avg_no_sel_features,fname)
    
    print("END OF THE EXPERIMENT")

    plt.close('all')
    return acc, auc, Counter(y), avg_no_sel_features, grid_search.best_params_


if  __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--classifier', type=str, help='Classifier Options:[xgb,GradientBoosting]', default='xgb')
    parser.add_argument('--smote', type=str, help='Classifier Options:[correct,incorrect]', default='correct')
    parser.add_argument('--features', type=int, help='Number of features to be used', default=100)
    parser.add_argument('--feature_selection', type=str, help='Type of feature selection. Options:[RFECV,fromModel]', default='fromModel')
    parser.add_argument('--pruning', type=str, help='Do pruning of features or not. Options:[prune,no_prune]', default='prune')
    parser.add_argument('--groups', type=str, help='binary classes to be classified ', default='CN_AD')
    parser.add_argument('--tuning', type=str, help='To perform hyperparameter sweep or not. Options:[sweep, no_sweep]', default='no_sweep')    
    args = parser.parse_args()
    
    if args.tuning == 'no_sweep':
        acc, auc, label_dist, avg_no_sel_features, best_params = run_ADNI(
                classifier = args.classifier,
                feature_selection = args.feature_selection,
                smote = args.smote,
                features=args.features,
                pruning=args.pruning,
                groups = args.groups       
               )
        print(args)
        print(acc,auc)

    HyperParameters = edict()
    HyperParameters.groups = ['CN_AD','CN_EMCI','CN_LMCI','EMCI_LMCI','EMCI_AD','LMCI_AD']  
    HyperParameters.classifier = ['xgb']
    HyperParameters.smote = ['correct'] 
    HyperParameters.features= [100,200,300,400,500]
    HyperParameters.pruning = ['prune','no_prune']
    HyperParameters.feature_selection = ['fromModel']#,'RFECV']
    HyperParameters.params = [HyperParameters.groups,HyperParameters.classifier,HyperParameters.smote,HyperParameters.features,HyperParameters.pruning,HyperParameters.feature_selection]  
    if args.tuning == 'sweep':
        final_result = pd.DataFrame(columns = ['Group', 'Label_distribution','classifier','smote','initial_feats','Pruning','feature_selection','final_feats','best_params','Macro_ACC','Macro_AUC'])
        params = list(itertools.product(*HyperParameters.params))
        for hp in params:
            print("For parameters:",hp)
            acc, auc, label_dist, avg_no_sel_features, best_params = run_ADNI(
                groups = hp[0],  
                classifier = hp[1],
                smote = hp[2],
                features=hp[3],         
                pruning=hp[4],
                feature_selection = hp[5]       
               )
            print(acc, auc)
            print('\n')

            final_result = final_result.append({'Group':hp[0], 'Label_distribution':label_dist,'classifier':hp[1],'smote':hp[2],
                                                'initial_feats':hp[3],'Pruning':hp[4],'feature_selection':hp[5],'final_feats':avg_no_sel_features,'best_params':best_params,
                                                'Macro_ACC':acc,'Macro_AUC':auc},
                                                ignore_index = True)
        
        final_result.to_csv(os.path.join(results_path,'sweep_results.csv'))
