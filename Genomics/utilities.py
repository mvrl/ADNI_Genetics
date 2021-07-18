# Utilities

import os
import numpy as np
from collections import Counter                      
import pandas as pd
from sklearn.metrics import auc
from matplotlib import pyplot as plt

 
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


def importance_extractor(original_cols,summary,results_path,fname):
    #This extracts the average importance of the seperate as well as common features selected in each folds by RFE
    FOLDS = len(summary['features'])
    imp_df_whole_cols = []
    for f in range(FOLDS):
        imp_df_whole_cols = imp_df_whole_cols + ['features_fold'+str(f),'importance_fold'+str(f)]
    imp_df_whole = pd.DataFrame(columns =imp_df_whole_cols)
    selectors = summary['features']
    selected_feats_dict = []
    selected_feats = []
    sel_feats_count = []
    for fold in range(FOLDS):
        sel_col = [x for x, y in zip(original_cols, summary['features'][fold]) if y] #selected features for each fold
        sel_feat_dict = {'features':sel_col,'importance':summary['importance'][fold]} #Importance for the selected features
        imp_df_whole['features_fold'+str(fold)] = pd.Series(sel_col)
        imp_df_whole['importance_fold'+str(fold)] = pd.Series(summary['importance'][fold])
        selected_feats.append(sel_col)
        selected_feats_dict.append(sel_feat_dict)
        sel_feats_count.append(len(sel_col))
    avg_no_sel_features = int(np.mean(np.array(sel_feats_count)))
    imp_df_whole.to_csv(os.path.join(results_path,fname+"_whole_imp.csv"))
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

