## This is just .py file for the jupyter notebook. So it is not well organized

import os
import pandas as pd
from pandas import read_csv
from collections import Counter
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.ensemble import (RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.model_selection import RandomizedSearchCV
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import ADASYN, SMOTE, SMOTENC
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc, f1_score, roc_auc_score, balanced_accuracy_score
from sklearn.feature_selection import RFECV
from easydict import EasyDict as edict
import itertools
overall_groups = ['CN_AD','CN_EMCI','CN_LMCI','EMCI_LMCI','EMCI_AD','LMCI_AD']
N=1000
############################################################################################
#                               SOME UTILITIES
###########################################################################################
def prepare_targets(y):
    le = LabelBinarizer()
    le.fit(y)
    y = le.transform(y)
    return y

#Very inefficient approach! but is easier to visualize in my head
def data_prep(df): #This takes the dataframe and returns the one hot encoded expansion of input features
    target = prepare_targets(list(df.DX_bl))
    df1 = df.drop(columns=['Unnamed: 0','DX_bl']).reset_index(drop=True) #Patient ID and DIAG not needed  
    return df1, target.ravel()

############################################################################################

def train_ADNI(groups='CN_AD',features=1000):


    ############################################################################################
                                            # DATA PREPARATION
    ##############################################################################################
    groups = groups
    N = features
    print("EXPERIMENT LOG FOR:",groups)
    print('\n')
    data_path = '/Users/subashkhanal/Desktop/BMI633/ADNI_Genetics/gene_expression/data'
    classes = groups.split('_')
    #
    #Gene ranking based on ttest
    ttest = read_csv(os.path.join(data_path,'t_test_0.10_geneExpr_Unfiltered_bl.csv')).sort_values(groups).reset_index()
    important_probes = ttest.sort_values(groups+'_c')['Gene'][0:N] #suffix _c to use the FDR corrected p values 
    #CHANGE THE LINE ABOVE ACCORDINGLY FOR DIFFERENT CLASSES

    #Gene Expression Data
    data_path = '/Users/subashkhanal/Desktop/BMI633/ADNI_Genetics/gene_expression/data/'
    df = pd.read_csv(os.path.join(data_path,'Unfiltered_gene_expr_dx.csv'),low_memory=False)
    Gene_expr = df[['Unnamed: 0','AGE','PTGENDER','PTEDUCAT','DX_bl']+list(important_probes)]
    df = Gene_expr
    df_CN = df[df['DX_bl']=='CN']
    df_AD = df[df['DX_bl']=='AD']
    df_EMCI = df[df['DX_bl']=='EMCI']
    df_LMCI = df[df['DX_bl']=='LMCI']

    if groups == 'CN_AD':
        curr_df = pd.concat([df_CN, df_AD], ignore_index=True) 
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
    
    df, y = data_prep(curr_df)
    print("Shape of final data BEFORE FEATURE SELECTION")
    print(df.shape, y.shape)

    ########################################################################################
    #                       RECURSIVE FEATURE ELIMINATION
    ########################################################################################

    estimator = GradientBoostingClassifier(random_state=1,n_estimators=2*df.shape[1])
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    selector = RFECV(estimator, n_jobs=-1,step=1, cv=cv)
    selector = selector.fit(df, y)
    df = df.loc[:, selector.support_]
    print("Shape of final data AFTER FEATURE SELECTION")
    print(df.shape, y.shape)
    final_N = df.shape[1]
    
    ########################################################################################
    #                       HYPERPARAMETER GRID SEARCH
    ########################################################################################
    #Adapted from #https://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html#sphx-glr-auto-examples-model-selection-plot-multi-metric-evaluation-py

    # Author: Raghav RV <rvraghav93@gmail.com>
    # License: BSD

    model = Pipeline([
            ('sampling', SMOTE(sampling_strategy=0.7, k_neighbors=7,random_state=1)),
            ('classifier', GradientBoostingClassifier(random_state=1))
        ])
    space = dict()
    X, y = df, y
    # define model
    model = GradientBoostingClassifier(random_state=1)
    # define evaluation
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    # define search space
    space = dict()
    space['n_estimators'] = range(50,5*X.shape[1],50)

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
    ax.set_xlim(min(space['n_estimators']), max(space['n_estimators'])+2)
    ax.set_ylim(0.50, 1)

    # Get the regular numpy array from the MaskedArray
    X_axis = np.array(results['param_n_estimators'].data, dtype=float)

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
    plt.savefig('./results/Grid_search_Using_GeneExpr_for:'+groups+'.png')

    ###########################################################################################
    #                           FINAL RUN AND SAVE RESULTS
    ###########################################################################################
    tprs = []
    aucs = []
    acc = []
    imp = []
    mean_fpr = np.linspace(0, 1, 100)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    fig, ax = plt.subplots()
    X, y = df, y
    for train, test in cv.split(X, y):
        X_train = X.iloc[train]
        y_train = y[train]
        
        X_test = X.iloc[test]
        y_test = y[test]
        n_estimators = result.best_params_['n_estimators']
        model = GradientBoostingClassifier(random_state=1,n_estimators=n_estimators)
        oversample = SMOTE(sampling_strategy=0.7, k_neighbors=7,random_state=1)
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
    plt.savefig('./results/ROC_for:'+groups+'.png')
    print('for total of ',final_N,"Features")
    print('Mean Balanced Accuracy:',sum(acc)/len(acc))
    print('Mean AUC:',sum(aucs)/len(aucs))

    imp = np.array(imp)
    imp = imp.mean(axis=0)

    imp_df = pd.DataFrame(columns=['features','importance'])
    imp_df['features'] = list(X.columns)
    imp_df['importance'] = imp

    imp_df_sorted = imp_df.sort_values(by=['importance'],ascending=False)
    imp_df_sorted.to_csv('./results/'+groups+'_Classification_ranked_'+str(final_N)+'_GeneExpr_features.csv')

    print("END OF THE EXPERIMENT")

    return sum(acc)/len(acc), sum(aucs)/len(aucs)


if  __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=int, help='Number of features to be used', default=100)
    parser.add_argument('--groups', type=str, help='binary classes to be classified ', default='CN_AD')
    parser.add_argument('--tuning', type=str, help='To perform hyperparameter sweep or not. Options:[sweep, no_sweep]', default='no_sweep')    
    args = parser.parse_args()
    
    if args.tuning == 'no_sweep':
        acc, auc = train_ADNI(features=args.features,
                groups = args.groups       
               )
    
    cfg = edict()
    HyperParameters = edict()
    HyperParameters.groups =['CN_AD','CN_EMCI','CN_LMCI', 'EMCI_LMCI','EMCI_AD','LMCI_AD'] 
    HyperParameters.features= [1000]
    HyperParameters.params = [HyperParameters.features,HyperParameters.groups]  
    if args.tuning == 'sweep':
        params = list(itertools.product(*HyperParameters.params))
        for hp in params:
            print("For parameters:",hp)
            acc, auc = train_ADNI(
                features = hp[0],
                groups = hp[1]     
               )
            print(acc, auc)
