#Script for ML using combined data
from utilities import save_results,importance_extractor,prepare_targets
import os
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, GridSearchCV
from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from imblearn.over_sampling import SMOTE, SMOTENC
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, balanced_accuracy_score
from imblearn.pipeline import Pipeline as imbpipeline
from sklearn.pipeline import Pipeline
from easydict import EasyDict as edict
from collections import Counter
import itertools
import warnings
warnings.filterwarnings("ignore")
################################################################################################

SEED = 11
np.random.seed(SEED)
RESULTS = 'results2' # Name of results folder, 'results' for overall grid search, results_test1, results_test2, results_test3 for 3 rounds of best model and features.
data_path = '/home/skh259/LinLab/LinLab/ADNI_Genetics/GWAS_Gene_Expr/'
results_path = os.path.join(data_path,RESULTS)
##########################################################################################################

def train_val(X_train, y_train, data_type,feature_selection,smote,classifier = 'xgb',smote_flag='correct', sampling=0.7, pruning='prune',step=50,seed=11):
    
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

    if smote_flag == 'correct' and pruning == 'prune':
        pipeline = imbpipeline(steps = [['smote', smote],
                                        ['featureSelection',feat_sel],
                                        ['classifier', clf]])
    elif smote_flag == 'incorrect' and pruning == 'prune':
        smote = smote
        X_train, y_train = smote.fit_resample(X_train, y_train)
        pipeline = Pipeline(steps = [['featureSelection',feat_sel],['classifier', clf]])
    
    elif  pruning == 'no_prune':
        pipeline = imbpipeline(steps = [['smote', smote],
                                        ['classifier', clf]])
                                            
    stratified_kfold = StratifiedKFold(n_splits=3,shuffle=True,random_state=seed)       
    grid_search = GridSearchCV(estimator=pipeline,
                                param_grid=param_grid,
                                scoring='balanced_accuracy',
                                cv=stratified_kfold,
                                n_jobs=-1,
                                error_score="raise")
    
    grid_search.fit(X_train, y_train)
    cv_score = grid_search.best_score_
    print(cv_score)

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
    original_cols = X_train.columns
    num_cols = ['AGE','EDU']
    for train, test in stratified_kfold.split(X_train, y_train):
        
        X_train_fold = X_train.iloc[train]
        y_train_fold = y_train[train]

        X_test_fold = X_train.iloc[test]
        y_test_fold = y_train[test]

        oversample = smote
        X_train_fold, y_train_fold = oversample.fit_resample(X_train_fold, y_train_fold)
        
        if pruning == 'prune':
            if feature_selection == 'RFECV':
                selector_fold = pipeline.named_steps['featureSelection'].fit(X_train_fold, y_train_fold).support_ 
            elif feature_selection == 'fromModel':
                selector_fold = pipeline.named_steps['featureSelection'].fit(X_train_fold, y_train_fold).get_support()
        else:
            selector_fold = [True for i in range(len(X_train.columns))] #Take all features no RFECV

        X_train_fold = X_train_fold.iloc[:,selector_fold]
        X_test_fold = X_test_fold.iloc[:,selector_fold]

        X_train_fold = np.array(X_train_fold)
        y_train_fold = np.array(y_train_fold)

        X_test_fold = np.array(X_test_fold)
        y_test_fold = np.array(y_test_fold)

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
###############################################################################################################################################################################################################    
def run_ADNI(data_type='combined',groups='CN_AD',features=1000,feature_selection='RFECV',classifier = 'xgb',smote_flag='correct',pruning='prune'):
    SAMPLING = 0.7
    groups = groups
    print("EXPERIMENT LOG FOR:",groups)
    print('\n')
    
    ttest = pd.read_csv(os.path.join(data_path,'data','t_test_0.10_geneExpr_Unfiltered_bl.csv'),low_memory=False).sort_values(groups).reset_index()
    important_probes = ttest.sort_values(groups+'_c')['Gene'][0:features] #suffix _c to use the FDR corrected p values
    df_expr = pd.read_csv(os.path.join(data_path,'data','common_expr.csv'),low_memory=False)
    df_expr = df_expr.loc[:,['Unnamed: 0','AGE','PTEDUCAT','DX_bl']+list(important_probes)]
    df_expr.columns = ['PTID','AGE','EDU','DIAG']+list(important_probes)
    df_expr = df_expr[(df_expr['DIAG']==groups.split('_')[0])| (df_expr['DIAG']==groups.split('_')[1])]

    top_snps = list(pd.read_csv(os.path.join(data_path,'data','top2000_snps.csv'),low_memory=False)['top_snps'])[:features]
    df_gwas = pd.read_csv(os.path.join(data_path,'data','common_gwas.csv'),low_memory=False)
    diag_map = {0:'CN',1:'AD'}
    df_gwas['DIAG'] = [diag_map[i] for i in list(df_gwas.DIAG)]
    cat_cols = [i for i in df_gwas.columns if 'rs' in i]  #Categorical features
    num_cols = ['PTID','AGE','EDU','DIAG']
    top_feats = [feat for feat in cat_cols if '_'.join(feat.split('_')[:2]) in top_snps] + num_cols
    df_gwas = df_gwas.loc[:,top_feats]
    
    if data_type == 'gwas':
        df = df_gwas
        y = df.DIAG
        df = df.drop(columns=['PTID','DIAG']).reset_index(drop=True) #Patient ID and DIAG not needed 
        all_cols = df.columns
        cat_columns_index = [i for i in range(len(all_cols)) if 'rs' in all_cols[i]]
        smote = SMOTENC(sampling_strategy=SAMPLING, k_neighbors=7,random_state=SEED,categorical_features = cat_columns_index)     

    if data_type == 'expr':
        df = df_expr
        y = df.DIAG
        df = df.drop(columns=['PTID','DIAG']).reset_index(drop=True) #Patient ID and DIAG not needed 
        smote = SMOTE(sampling_strategy=SAMPLING, k_neighbors=7,random_state=SEED)
        
        
    if data_type == 'combined':
        df = pd.merge(df_gwas,df_expr,how='left', on=['PTID','AGE','EDU','DIAG']) #EARLY FUSION
        y = df.DIAG
        df = df.drop(columns=['PTID','DIAG']).reset_index(drop=True) #Patient ID and DIAG not needed 
        all_cols = df.columns
        cat_columns_index = [i for i in range(len(all_cols)) if 'rs' in all_cols[i]]
        smote = SMOTENC(sampling_strategy=SAMPLING, k_neighbors=7,random_state=SEED,categorical_features = cat_columns_index)     

    print("Shape of final data BEFORE FEATURE SELECTION")
    print(df.shape, y.shape)
    y = prepare_targets(y)
    print("Label distribution")
    print(Counter(y))
    STEP = int(df.shape[1]/20)
    fname = '_'.join([data_type,feature_selection,classifier,str(features),pruning,smote_flag])
    grid_search,summary = train_val(df, y, data_type = data_type,feature_selection=feature_selection,smote=smote,classifier = classifier,smote_flag=smote_flag, sampling=SAMPLING, pruning=pruning,step=STEP,seed=SEED)
    results = np.array(summary['results'])
    original_cols = df.columns
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
    parser.add_argument('--data_type', type=str, help='Classifier Options:[expr,gwas,combined]', default='combined')
    parser.add_argument('--classifier', type=str, help='Classifier Options:[xgb,GradientBoosting]', default='xgb')
    parser.add_argument('--smote_flag', type=str, help='Classifier Options:[correct,incorrect]', default='correct')
    parser.add_argument('--features', type=int, help='Number of features to be used', default=100)
    parser.add_argument('--feature_selection', type=str, help='Type of feature selection. Options:[RFECV,fromModel]', default='fromModel')
    parser.add_argument('--pruning', type=str, help='Do pruning of features or not. Options:[prune,no_prune]', default='prune')
    parser.add_argument('--groups', type=str, help='binary classes to be classified ', default='CN_AD')
    parser.add_argument('--tuning', type=str, help='To perform hyperparameter sweep or not. Options:[sweep, no_sweep]', default='no_sweep')    
    args = parser.parse_args()
    
    if args.tuning == 'no_sweep':
        acc, auc, label_dist, avg_no_sel_features, best_params = run_ADNI(
                data_type = args.data_type,
                classifier = args.classifier,
                feature_selection = args.feature_selection,
                smote_flag = args.smote,
                features=args.features,
                pruning=args.pruning,
                groups = args.groups       
               )
        print(args)
        print(acc,auc)

    HyperParameters = edict()
    #HyperParameters.data_type = ['expr','gwas','combined']
    HyperParameters.data_type = ['expr','gwas','combined']
    HyperParameters.groups = ['CN_AD']  
    HyperParameters.classifier = ['xgb']
    HyperParameters.smote_flag = ['correct'] 
    HyperParameters.features= [100,200,300,400,500]
    HyperParameters.pruning = ['prune','no_prune']
    HyperParameters.feature_selection = ['fromModel']#,'RFECV']
    HyperParameters.params = [HyperParameters.data_type,HyperParameters.groups,HyperParameters.classifier,HyperParameters.smote_flag,HyperParameters.features,HyperParameters.pruning,HyperParameters.feature_selection]  
    if args.tuning == 'sweep':
        final_result = pd.DataFrame(columns = ['data_type','Label_distribution','classifier','smote_flag','initial_feats','Pruning','feature_selection','final_feats','best_params','Macro_ACC','Macro_AUC'])
        params = list(itertools.product(*HyperParameters.params))
        for hp in params:
            print("For parameters:",hp)
            acc, auc, label_dist, avg_no_sel_features, best_params = run_ADNI(
                data_type = hp[0],
                groups = hp[1],  
                classifier = hp[2],
                smote_flag = hp[3],
                features=hp[4],
                pruning=hp[5],
                feature_selection = hp[6]        
               )
            print(acc, auc)
            print('\n')

            final_result = final_result.append({'data_type':hp[0], 'Label_distribution':label_dist,'classifier':hp[2],'smote_flag':hp[3],
                                                'initial_feats':hp[4],'Pruning':hp[5],'feature_selection':hp[6],'final_feats':avg_no_sel_features,'best_params':best_params,
                                                'Macro_ACC':acc,'Macro_AUC':auc},
                                                ignore_index = True)
        
        final_result.to_csv(os.path.join(results_path,'sweep_results.csv'))
