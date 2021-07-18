from utilities import save_results,importance_extractor
import os
import pandas as pd
from collections import Counter
from sklearn.model_selection import RepeatedStratifiedKFold,GridSearchCV, StratifiedKFold
from sklearn.feature_selection import RFE
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
from scipy import stats
from scipy.stats import ttest_ind
import statsmodels.stats.multitest as multi
warnings.filterwarnings("ignore")

overall_groups = ['CN_AD','CN_EMCI','CN_LMCI','EMCI_LMCI','EMCI_AD','LMCI_AD']
SEED = 11
FOLDS = 5
root_path = '/home/skh259/LinLab/LinLab/ADNI_Genetics/Genomics/'
CV_path = '/mnt/gpfs2_16m/pscratch/nja224_uksr/SKH259/LinLab/ADNI_Genetics/Genomics/data/GWAS/CN_AD/cv_folds/'
RESULTS = 'results'
results_path=os.path.join(root_path,RESULTS)
############################################################################################
def train_val(groups,features, feature_selection, classifier = 'xgb',smote='correct',pruning='prune',seed=11):
    N = features
    step = int(features/20)
    if N < 50:
        n_estimators = range(10,3*features,10)
    else:
        n_estimators = range(100,3*features,100)
    max_depths = range(2,10,2)
    SAMPLING = 1.0

    HyperParameters = edict()
    HyperParameters.n_estimators = n_estimators
    HyperParameters.max_depths = max_depths

    if classifier  == 'xgb':
        HyperParameters.params = [HyperParameters.n_estimators,HyperParameters.max_depths]  
    elif classifier == 'GradientBoosting':
        HyperParameters.params = [HyperParameters.n_estimators]
    
    if groups == 'CN_AD':
        SAMPLING = 0.7 
    
    params = list(itertools.product(*HyperParameters.params))
    overall_summary = []
    for hp in params: #Perform the entire process for each combination of hyper parameters
        fname = '_'.join([groups,classifier,str(features),pruning,feature_selection,smote])

        if classifier  == 'xgb':
            n_estimators = hp[0]
            max_depth = hp[1]
            est = xgb.XGBClassifier(n_estimators=n_estimators,max_depth=max_depth,eval_metric='logloss',use_label_encoder=False)

        elif classifier == 'GradientBoosting':
            n_estimators = hp[0]
            est = GradientBoostingClassifier(n_estimators=n_estimators)
    
        mean_fpr = np.linspace(0, 1, 100)
        selector = []
        results = []
        tprs = []
        aucs = []
        acc = []
        imp = []
        summary = dict()
        for fold in range(FOLDS):
            ############################### Plink based feature selected data ######################################################
            train_df = pd.read_csv(os.path.join(CV_path,'fold'+str(fold),'train','QualityControl','final_SNP_data.csv'))
            if 'Unnamed: 0' in train_df.columns:
                train_df = train_df.drop(columns=['Unnamed: 0'])
            drop_cols = ['nan','00']
            keep_cols = ['PTID','AGE','EDU','DIAG']
            train_snp_cols = [col for col in list(train_df.columns) if col not in keep_cols]
            train_cols_keep = [col for col in train_snp_cols if col.split('_')[2] not in drop_cols]

            train_df = train_df[list(keep_cols) + list(train_cols_keep)]
            
            test_df = pd.read_csv(os.path.join(CV_path,'fold'+str(fold),'test','QualityControl','final_SNP_data.csv'))
            if 'Unnamed: 0' in test_df.columns:
                test_df = test_df.drop(columns=['Unnamed: 0'])
            test_snp_cols = [col for col in list(test_df.columns) if col not in keep_cols]
            test_cols_keep = [col for col in test_snp_cols if col.split('_')[2] not in drop_cols]
            test_df = test_df[list(keep_cols)+list(test_cols_keep)]
        
            snp_rank = pd.read_csv(os.path.join(CV_path,'fold'+str(fold),'top2000_snps.csv'))['top_snps'] #Ranked list of snps after running GWAS on training fold
        
            top_snps = list(snp_rank[:N]) #eg feature name: 19_rs2075650_AA', eg: SNP from plink 19_rs2075650
    
            top_snp_feats_train = [feat for feat in train_cols_keep  if '_'.join(feat.split('_')[:2]) in top_snps]
            top_snp_feats_test = [feat for feat in test_cols_keep if '_'.join(feat.split('_')[:2]) in top_snps]

            #Just for the case: some features might be missing between test and train while dropping NAN columns so
            top_snp_feats = list(set(top_snp_feats_train).intersection(set(top_snp_feats_test)))
            Gwas_train = train_df[['PTID','AGE','EDU','DIAG']+list(top_snp_feats)]
            Gwas_test = test_df[['PTID','AGE','EDU','DIAG']+list(top_snp_feats)]

            y_train = Gwas_train.DIAG
            X_train = Gwas_train.drop(columns=['PTID','DIAG']).reset_index(drop=True) #Patient ID and DIAG not needed  

            y_test = Gwas_test.DIAG
            X_test = Gwas_test.drop(columns=['PTID','DIAG']).reset_index(drop=True) #Patient ID and DIAG not needed 
            
            X_train_fold = np.array(X_train)
            y_train_fold = np.array(y_train)
            X_test_fold = np.array(X_test)
            y_test_fold = np.array(y_test)

            num_cols = ['AGE','EDU']
            original_cols = list(X_train.columns)
    
            cat_columns_index = [i for i in range(len(original_cols)) if original_cols[i] not in num_cols]
             ############################### SMOTE to balance training fold ######################################################
            oversample = SMOTENC(sampling_strategy=SAMPLING, k_neighbors=7,categorical_features = cat_columns_index,random_state=seed)
            X_train_fold, y_train_fold = oversample.fit_resample(X_train_fold, y_train_fold)
             ############################### Model based feature selection ######################################################
            if pruning == 'prune':
                if feature_selection == 'RFE':
                    selector_fold = RFE(estimator=est,step=step).fit(X_train_fold, y_train_fold).support_ 
                elif feature_selection == 'fromModel':
                    selector_fold = SelectFromModel(estimator=est).fit(X_train_fold, y_train_fold).get_support()
            else: ##NO PRUNING
                selector_fold = [True for i in range(X_train_fold.shape[1])] #Take all features no RFE

            X_train_fold = X_train_fold[:,selector_fold]
            X_test_fold = X_test_fold[:,selector_fold]
            ############################### TRAINING AND TESTING ######################################################
            probas_ = est.fit(X_train_fold, y_train_fold).predict_proba(X_test_fold)
            y_pred = est.predict(X_test_fold)
            acc = balanced_accuracy_score(y_test_fold, y_pred)
            fpr, tpr, thresholds = roc_curve(y_test_fold, probas_[:, 1],drop_intermediate='False')
            roc_auc = roc_auc_score(y_test_fold, probas_[:, 1])
            
            results.append([acc,roc_auc])
            selector.append(selector_fold)

            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            imp.append(est.feature_importances_)
            summary.update({'params':hp,'features':selector,'results':results,'tprs':tprs,'mean_fpr':mean_fpr,'importance':imp})
            #print("for hp:",hp)
            #print("Performance for fold:",fold," ACC:",acc," AUC:",roc_auc)
        overall_summary.append(summary)
    return overall_summary, original_cols

def run_ADNI(groups='CN_AD',features=1000,feature_selection='RFE',classifier = 'xgb',smote='correct',pruning='prune'):
    
    fname = '_'.join([groups,classifier,str(features),pruning,feature_selection,smote])
    summary,original_cols= train_val(groups,features = features,feature_selection=feature_selection,classifier = classifier,smote=smote,pruning=pruning,seed=SEED)
    overall_results = []
    for hp in range(len(summary)):
         #Finding best results and hyper parameters
        results = np.array(summary[hp]['results'])
        accs = results[:,0]
        acc = np.mean(accs)
        overall_results.append(acc)
    best = np.argmax(np.array(overall_results))
    best_summary = summary[best]
    hp = best_summary['params']
    results = np.array(best_summary['results'])
    aucs = results[:,1]
    accs = results[:,0]
    acc = np.mean(accs)
    auc = np.mean(aucs)
    imp_df,avg_no_sel_features = importance_extractor(original_cols,best_summary,results_path,fname)

    print("Best hyperparameters for",classifier,":",hp)
    print("best Macro ACC:",acc,"best Macro AUC:",auc)
    print("Avg number of features AFTER FEATURE SELECTION:",avg_no_sel_features)
    fig, ax = plt.subplots()
    save_results(ax,imp_df,best_summary['tprs'],best_summary['mean_fpr'],aucs,accs,results_path,avg_no_sel_features,fname)
    
    print("END OF THE EXPERIMENT")
    plt.close('all')
    return acc, auc,avg_no_sel_features, hp


if  __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--classifier', type=str, help='Classifier Options:[xgb,GradientBoosting]', default='xgb')
    parser.add_argument('--smote', type=str, help='Classifier Options:[correct,incorrect]', default='correct')
    parser.add_argument('--features', type=int, help='Number of features to be used', default=100)
    parser.add_argument('--feature_selection', type=str, help='Type of feature selection. Options:[RFE,fromModel]', default='fromModel')
    parser.add_argument('--pruning', type=str, help='Do pruning of features or not. Options:[prune,no_prune]', default='prune')
    parser.add_argument('--groups', type=str, help='binary classes to be classified ', default='CN_AD')
    parser.add_argument('--tuning', type=str, help='To perform hyperparameter sweep or not. Options:[sweep, no_sweep]', default='no_sweep')    
    args = parser.parse_args()
    
    if args.tuning == 'no_sweep':
        acc, auc, avg_no_sel_features, best_params = run_ADNI(
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
    HyperParameters.groups = ['CN_AD']
    HyperParameters.classifier = ['xgb']
    HyperParameters.smote = ['correct'] 
    HyperParameters.features= [25,50,100,200,300,500,750,1000]
    HyperParameters.pruning = ['prune','no_prune']
    HyperParameters.feature_selection = ['RFE']#,'fromModel'] 
    HyperParameters.params = [HyperParameters.groups,HyperParameters.classifier,HyperParameters.smote,HyperParameters.features,HyperParameters.pruning,HyperParameters.feature_selection]  
    if args.tuning == 'sweep':
        final_result = pd.DataFrame(columns = ['Group','classifier','smote','initial_feats','Pruning','feature_selection','final_feats','best_params','Macro_ACC','Macro_AUC'])
        params = list(itertools.product(*HyperParameters.params))
        for hp in params:
            print("For parameters:",hp)
            acc, auc, avg_no_sel_features, best_params = run_ADNI(
                groups = hp[0],  
                classifier = hp[1],
                smote = hp[2],
                features=hp[3],         
                pruning=hp[4],
                feature_selection = hp[5]       
               )
            print(acc, auc)
            print('\n')

            final_result = final_result.append({'Group':hp[0],'classifier':hp[1],'smote':hp[2],
                                                'initial_feats':hp[3],'Pruning':hp[4],'feature_selection':hp[5],'final_feats':avg_no_sel_features,'best_params':best_params,
                                                'Macro_ACC':acc,'Macro_AUC':auc},
                                                ignore_index = True)
        
        final_result.to_csv(os.path.join(results_path,'sweep_results_RFE.csv'))
