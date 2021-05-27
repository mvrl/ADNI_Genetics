from utilities import data_prep1,GRID_search,GWAS_data_prep,GeneExpr_data_prep,combined_data_prep,save_results,RFE,prepare_targets
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
RESULTS = 'results' #Change this path accordingly
GWAS_data_path = '/home/skh259/LinLab/LinLab/ADNI_Genetics/Genomics/'
GeneExpr_data_path = '/home/skh259/LinLab/LinLab/ADNI_Genetics/gene_expression/'
results_path = '/home/skh259/LinLab/LinLab/ADNI_Genetics/GWAS_Gene_Expr/'+RESULTS 
##########################################################################################################
def train_ADNI(groups='CN_AD',features=1000,pruning='prune',data_type = 'combined',fusion='late'):

    fname = '_'.join([groups,fusion,str(features),pruning,data_type])
    groups = groups
    print("EXPERIMENT LOG FOR:",groups)
    print('\n')
    print("Using data:",data_type)
    print('\n')

    df_final_GWAS = GWAS_data_prep(GWAS_data_path,results_path,features)
    curr_df = GeneExpr_data_prep(groups,GeneExpr_data_path,features)
    GWAS_data_final,Gene_expr_final,df1,y1,df2,y2,num_cols = combined_data_prep(groups, df_final_GWAS, curr_df)

    
    STEP1 = int(df1.shape[1]/20)
    STEP2 = int(df2.shape[1]/20)
    GWAS_data_final['GENDER']=GWAS_data_final['GENDER'].astype('category').cat.codes
    Gene_expr_final['EDU']=Gene_expr_final['EDU'].astype('float64')
    
    if fusion == 'early':
        ###EARLY FUSION OF INPUT FEATURES
        GWAS_GeneExpr_df = pd.merge(GWAS_data_final,Gene_expr_final,how='left', on=['PTID','AGE','GENDER','EDU','DIAG'])
        df, y = data_prep1(GWAS_GeneExpr_df,groups,num_cols)
        print("Shape of common combined data")
        print(df.shape, y.shape)

        if data_type == 'expr':
            df, y = df2, y2
        
        if data_type == 'gwas':
            df, y = df1, y1
            num_cols = []
        
        if pruning == 'prune':
            STEP = int(df.shape[1]/20)
            estimator = GradientBoostingClassifier(random_state=SEED, n_estimators=2*df.shape[1])
            cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=SEED)
            selector = RFECV(estimator, n_jobs=-1,step=STEP,cv=cv,scoring='balanced_accuracy',min_features_to_select=10)
            selector = selector.fit(df, y)
            df = df.loc[:, selector.support_]


    if fusion == 'late':
        ###LATE FUSION OF INPUT FEATURES
        if data_type == 'expr':
            df, y = df1, y1
            STEP = STEP1
            if pruning == 'prune':
                df, y = RFE(df,y,STEP)
            print("Shape of final "+ data_type+" data AFTER FEATURE SELECTION")
            
        if data_type == 'gwas':
            df, y = df2, y2
            STEP = STEP2
            if pruning == 'prune':
                df, y = RFE(df,y,STEP)
            print("Shape of final "+ data_type+" data AFTER FEATURE SELECTION")

        if data_type == 'combined':
            GWAS_PTID = GWAS_data_final.PTID
            GeneExpr_PTID = Gene_expr_final.PTID
            GWAS_DIAG = GWAS_data_final.DIAG
            GeneExpr_DIAG = Gene_expr_final.DIAG
            if pruning == 'prune':
                df1, y1 = RFE(df1,y1,STEP1)
            df1['PTID'] = list(GWAS_PTID)
            df1['DIAG'] = list(GWAS_DIAG)
            if pruning == 'prune':
                df2, y2 = RFE(df2,y2,STEP2)
            df2['PTID'] = list(GeneExpr_PTID)
            df2['DIAG'] = list(GeneExpr_DIAG)

            ## LATE FUSION
            if 'AGE' in df1.columns and 'AGE' in df2.columns:
                GWAS_GeneExpr_df = pd.merge(df1,df2,how='left',on=['PTID','DIAG','AGE'])
            else:
                GWAS_GeneExpr_df = pd.merge(df1,df2,how='left',on=['PTID','DIAG'])
            y = prepare_targets(list(GWAS_GeneExpr_df.DIAG),groups).ravel()
            df = GWAS_GeneExpr_df.drop(columns=['PTID','DIAG']).reset_index(drop=True) #Patient ID and DIAG not needed
            print("Shape of final "+ data_type+" data AFTER FEATURE SELECTION")

    print(df.shape)
    print("Label distribution ater final feature selection")
    label_dist = Counter(y)
    print(label_dist)
    final_N = df.shape[1]
    cat_columns = list(set(df.columns) - set(['AGE','EDU']+num_cols))
    cat_columns_index = [i for i in range(len(df.columns)) if df.columns[i] in cat_columns]
 
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
    print("END OF THE EXPERIMENT\n")

    plt.close('all')
    return final_ACC, final_AUC, final_N, label_dist, n_estimators


if  __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=int, help='Number of features to be used', default=100)
    parser.add_argument('--fusion', type=str, help='Early or Late fusion of input features. Options:[early,late]', default='early')
    parser.add_argument('--groups', type=str, help='binary classes to be classified ', default='CN_AD')
    parser.add_argument('--pruning', type=str, help='Do pruning of features or not. Options:[prune,no_prune]', default='prune')
    parser.add_argument('--data_type', type=str, help='type of genetic data to use ', default='combined')
    parser.add_argument('--tuning', type=str, help='To perform hyperparameter sweep or not. Options:[sweep, no_sweep]', default='no_sweep')    
    args = parser.parse_args()
    
    if args.tuning == 'no_sweep':
        acc, my_auc, final_N, label_dist,n_estimators = train_ADNI(
                features=args.features,
                fusion = args.fusion,
                pruning=args.pruning,
                groups = args.groups, 
                data_type = args.data_type,
               )
    
    HyperParameters = edict()
    HyperParameters.groups =['CN_AD'] 
    HyperParameters.features= [100,200,300,400,500]
    #HyperParameters.features= [100]
    HyperParameters.fusion = ['early','late']
    HyperParameters.data_type= ['expr','gwas','combined']
    HyperParameters.pruning = ['prune','no_prune']
    HyperParameters.params = [HyperParameters.features,HyperParameters.fusion,HyperParameters.pruning,HyperParameters.groups,HyperParameters.data_type]  
    if args.tuning == 'sweep':
        final_result = pd.DataFrame(columns = ['Group', 'Label_distribution', 'feat_type','initial_feats','Pruning','final_feats','best_n_estimators','Macro_ACC','Macro_AUC'])
        params = list(itertools.product(*HyperParameters.params))
        for hp in params:
            print("For parameters:",hp)
            acc, my_auc, final_N, label_dist,n_estimators = train_ADNI(
                features = hp[0],
                fusion = hp[1],
                pruning = hp[2],
                groups = hp[3],
                data_type = hp[4],   
               )
            print(acc, my_auc)
            print('\n')
            final_result = final_result.append({'Group':hp[3], 'Label_distribution':label_dist, 'feat_type':hp[4],
                                                     'initial_feats':hp[0],'Pruning':hp[2],'final_feats':final_N,
                                                'best_n_estimators':n_estimators,'Macro_ACC':acc,'Macro_AUC':my_auc},
                                                ignore_index = True)
        
        final_result.to_csv(os.path.join(results_path,'sweep_results.csv'))
        
