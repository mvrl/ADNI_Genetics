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
    
    N = features
    #Gene ranking based on ttest
    ttest = read_csv(os.path.join(root_path,'data','t_test_0.10_geneExpr_Unfiltered_bl.csv')).sort_values(groups).reset_index()
    important_probes = ttest.sort_values(groups+'_c')['Gene'][0:N] #suffix _c to use the FDR corrected p values 
    #CHANGE THE LINE ABOVE ACCORDINGLY FOR DIFFERENT CLASSES

    #Gene Expression Data
    df = pd.read_csv(os.path.join(root_path,'data','Unfiltered_gene_expr_dx.csv'),low_memory=False)
    Gene_expr = df[['Unnamed: 0','AGE','PTGENDER','PTEDUCAT','DX_bl']+list(important_probes)]
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

    curr_df['PTGENDER'] = curr_df['PTGENDER'].astype('category').cat.codes 
    print('Label distribution of current experiment:')
    print(Counter(curr_df.DX_bl))
    df, y = data_prep(curr_df,groups)

    return df, y, SAMPLING


def GridSearch(df,y,cat_columns_index,SAMPLING,results_path,fname,SEED):
    ########################################################################################
    #                       HYPERPARAMETER GRID SEARCH
    ########################################################################################
    #Adapted from #https://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html#sphx-glr-auto-examples-model-selection-plot-multi-metric-evaluation-py

    # Author: Raghav RV <rvraghav93@gmail.com>
    # License: BSD
    if len(cat_columns_index) > 0:
        model = Pipeline([
                ('sampling', SMOTENC(sampling_strategy=SAMPLING, k_neighbors=7,random_state=SEED,categorical_features = cat_columns_index)),
                ('classifier', GradientBoostingClassifier(random_state=SEED))
            ])
    else:
        model = Pipeline([
                ('sampling', SMOTE(sampling_strategy=SAMPLING, k_neighbors=7,random_state=SEED)),
                ('classifier', GradientBoostingClassifier(random_state=SEED))
            ])

    space = dict()
    X, y = df, y
    
    # define evaluation
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    # define search space
    space = dict()
    if 7*X.shape[1] < 50:
        space['classifier__n_estimators'] = range(50,200,50) #for case where number of features is too low. 
    else:
        space['classifier__n_estimators'] = range(50,7*X.shape[1],50)

    scoring = {'AUC': 'roc_auc', 'balanced_accuracy':'balanced_accuracy'}
    # define search
    search = GridSearchCV(model, space,n_jobs=-1, cv=cv,scoring=scoring, refit='balanced_accuracy', return_train_score=True)
    # execute search
    result = search.fit(X, y)
    # summarize result
    print('Best Score: %s' % result.best_score_)
    print('Best Hyperparameters: %s' % result.best_params_)
    results = search.cv_results_

    print(__doc__)
    plt.figure(figsize=(13, 13))
    plt.title("GridSearchCV evaluating using multiple scorers simultaneously",
            fontsize=16)

    plt.xlabel("param_n_estimators")
    plt.ylabel("Score")

    ax = plt.gca()
    ax.set_xlim(min(space['classifier__n_estimators']), max(space['classifier__n_estimators'])+2)
    ax.set_ylim(0.50, 1)

    # Get the regular numpy array from the MaskedArray
    X_axis = np.array(results['param_classifier__n_estimators'].data, dtype=float)

    for scorer, color in zip(sorted(scoring), ['g', 'k']):
        for sample, style in (('train', '--'), ('test', '-')):
            sample_score_mean = results['mean_%s_%s' % (sample, scorer)]
            sample_score_std = results['std_%s_%s' % (sample, scorer)]
            ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                            sample_score_mean + sample_score_std,
                            alpha=0.1 if sample == 'test' else 0, color=color)
            ax.plot(X_axis, sample_score_mean, style, color=color,
                    alpha=1 if sample == 'test' else 0.7,
                    label="%s (%s)" % (scorer, sample))

        best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
        best_score = results['mean_test_%s' % scorer][best_index]

        # Plot a dotted vertical line at the best score for that scorer marked by x
        ax.plot([X_axis[best_index], ] * 2, [0, best_score],
                linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

        # Annotate the best score for that scorer
        ax.annotate("%0.2f" % best_score,
                    (X_axis[best_index], best_score + 0.005))

    plt.legend(loc="best")
    plt.grid(False)
    plt.savefig(os.path.join(results_path,'Grid_search_Using'+'_'+fname+'.png'))

    return result


def save_results(X,ax,imp,tprs, mean_fpr,aucs,acc,results_path,final_N,fname):
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
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
    print('for total of ',final_N,"Features")
    print('Mean Balanced Accuracy:',sum(acc)/len(acc))
    print('Mean AUC:',sum(aucs)/len(aucs))

    imp = np.array(imp)
    imp = imp.mean(axis=0)

    imp_df = pd.DataFrame(columns=['features','importance'])
    imp_df['features'] = list(X.columns)
    imp_df['importance'] = imp

    imp_df_sorted = imp_df.sort_values(by=['importance'],ascending=False)
    imp_df_sorted.to_csv(os.path.join(results_path,'Features_ranked_for'+'_'+fname+'.csv'))