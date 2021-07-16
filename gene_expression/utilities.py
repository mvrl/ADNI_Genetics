# Utilities

from pandas import read_csv
import os
import pandas as pd
from collections import Counter
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, SMOTENC
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import auc
from sklearn.feature_selection import RFECV
from easydict import EasyDict as edict

def prepare_targets(y,groups):
    class1 = groups.split('_')[0]
    class2 = groups.split('_')[1]
    count_dict = Counter(y)
    class1_count = count_dict[class1]
    class2_count = count_dict[class2]
    ## Label minority class = 1 and majority class = 0
    if  class1_count > class2_count:
        count_dict[class1] = int(0)
        count_dict[class2] = int(1)
    else:
        count_dict[class1] = int(1)
        count_dict[class2] = int(0)

    op = [count_dict[i] for i in y]
    return np.asarray(op)

def data_prep(df,groups): 
    target = prepare_targets(list(df.DX_bl),groups)
    df1 = df.drop(columns=['Unnamed: 0','DX_bl']).reset_index(drop=True) #Patient ID and DIAG not needed  
    return df1, target.ravel()


def GeneExpr_data_prep(groups,root_path,features):
    
    # N = features
    # #Gene ranking based on ttest
    # ttest = read_csv(os.path.join(root_path,'data','t_test_0.10_geneExpr_Unfiltered_bl.csv')).sort_values(groups).reset_index()
    # important_probes = ttest.sort_values(groups+'_c')['Gene'][0:N] #suffix _c to use the FDR corrected p values 
    # #CHANGE THE LINE ABOVE ACCORDINGLY FOR DIFFERENT CLASSES

    # #Gene Expression Data
    df = pd.read_csv(os.path.join(root_path,'data','Unfiltered_gene_expr_dx.csv'),low_memory=False)
    #Gene_expr = df[['Unnamed: 0','AGE','PTEDUCAT','DX_bl']+list(important_probes)]
    df = Gene_expr
    print('Label distribution of overall data:')
    print(Counter(df.DX_bl))
    df_CN = df[df['DX_bl']=='CN']
    df_AD = df[df['DX_bl']=='AD']
    df_EMCI = df[df['DX_bl']=='EMCI']
    df_LMCI = df[df['DX_bl']=='LMCI']
    SAMPLING = 1.0
    if groups == 'CN_AD':
        curr_df = pd.concat([df_CN, df_AD], ignore_index=True)
        SAMPLING = 0.7 
    if groups == 'CN_EMCI':
        curr_df = pd.concat([df_CN, df_EMCI], ignore_index=True)
    if groups == 'CN_LMCI':
        curr_df = pd.concat([df_CN, df_LMCI], ignore_index=True)
    if groups == 'EMCI_LMCI':
        curr_df = pd.concat([df_EMCI, df_LMCI], ignore_index=True)
    if groups == 'EMCI_AD':
        curr_df = pd.concat([df_EMCI, df_AD], ignore_index=True)
    if groups == 'LMCI_AD':
        curr_df = pd.concat([df_LMCI, df_AD], ignore_index=True)

    print('Label distribution of current experiment:')
    print(Counter(curr_df.DX_bl))
    df, y = data_prep(curr_df,groups)

    return df, y, SAMPLING

def save_results(ax,imp_df,tprs, mean_fpr,aucs,accs,results_path,avg_no_sel_features,fname):
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = round(auc(mean_fpr, mean_tpr),2)
    std_auc = round(np.std(aucs),2)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
        title="Receiver operating characteristic")
    ax.legend(loc="lower right")
    plt.show()
    plt.savefig(os.path.join(results_path,'ROC_for:'+fname+'.png'))
    print('for Avg',avg_no_sel_features,"Features")
    print('Mean Balanced Accuracy:',sum(accs)/len(accs))
    print('Mean AUC:',sum(aucs)/len(aucs))

    imp_df.to_csv(os.path.join(results_path,'Features_ranked_for'+'_'+fname+'.csv'))


def importance_extractor(original_cols,summary):
    #This extracts the average importance of the common features selected in each folds by RFE
    FOLDS = len(summary['features'])
    selectors = summary['features']
    selected_feats_dict = []
    selected_feats = []
    sel_feats_count = []
    for fold in range(FOLDS):
        sel_col = [x for x, y in zip(original_cols, summary['features'][fold]) if y] #selected features for each fold
        sel_feat_dict = {'features':sel_col,'importance':summary['importance'][fold]} #Importance for the selected features
        selected_feats.append(sel_col)
        selected_feats_dict.append(sel_feat_dict)
        sel_feats_count.append(len(sel_col))
    avg_no_sel_features = int(np.mean(np.array(sel_feats_count)))
    
    common_feats = list(set(selected_feats[0]).intersection(*selected_feats))
    avg_imp = []
    imp_df = pd.DataFrame(columns =['features', 'importance'])
    for feat in common_feats:
        imps = []
        for fold in range(FOLDS):
            feat_idx = selected_feats_dict[fold]['features'].index(feat)
            imps.append(selected_feats_dict[fold]['importance'][feat_idx]) 
        avg_imp.append((feat,np.mean(np.array(imps))))
    if len(common_feats) > 0:
        imp_df['features'] = common_feats
        imp_df['importance'] = avg_imp
        imp_df = imp_df.sort_values(by=['importance'],ascending=False)

    return imp_df, avg_no_sel_features


