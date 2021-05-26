## This is just .py file for the jupyter notebook. So it is not well organized

import os
import pandas as pd
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
from easydict import EasyDict as edict
import itertools
import warnings
warnings.filterwarnings("ignore")
################################################################################################
#                                           UTILITIES
################################################################################################
SEED = 1
np.random.seed(SEED)
def sequence_parser(t):
        
        t1 = [t[i].strip() for i in range(len(t)) if i%2 !=0]
        t2 = [t[i].strip() for i in range(len(t)) if i%2 ==0]
        Geno = [t1[i]+t2[i] for i in range(len(t1))]
        
        return Geno

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

#Very inefficient approach! but is easier to visualize in my head
def data_prep(df,groups): #This takes the dataframe and returns the one hot encoded expansion of input features
    target = prepare_targets(list(df.DIAG),groups)
    df1 = df.drop(columns=['PTID','DIAG']).reset_index(drop=True) #Patient ID and DIAG not needed
    num_cols = ['AGE','EDU']
    cat_cols = list(set(df1.columns) - set(num_cols)) #Categorical features
    expand_cat = ['AGE','EDU'] #List of expanded categorical columns
    for cat in cat_cols:
        expand_cat = expand_cat + [str(cat)+'_'+ c for c in list(set(df1[cat]))]
    df_out = pd.DataFrame(columns=list(expand_cat))
    df_out['AGE'] = df1.AGE
    df_out['EDU'] = df1.EDU
    for i in range(len(df1)):
        row = df1.iloc[i]
        for col in cat_cols:
            item = row[col]
            df_out.at[i,str(col)+'_'+ item] = str(1)
        
    df_out = df_out.fillna(str(0))
    return df_out, target.ravel()

##########################################################################################################
def train_ADNI(groups='CN_AD',features=1000):
    
    groups = groups
    print("EXPERIMENT LOG FOR:",groups)
    print('\n')
    data_path = '/home/skh259/LinLab/LinLab/ADNI_Genetics/Genomics/'
    #Number of top SNPs to take as features
    N = features
    ########################################################################################
    #                       DATA PREPERATION
    ########################################################################################
    df = pd.read_csv(os.path.join(data_path,'data','ADNIMERGE.csv'),low_memory=False)
    df_bl = df[df['VISCODE']=='bl']
    print('Overall label distribution on ADNIMERGE.csv')
    print(Counter(df[df['VISCODE']=='bl']['DX_bl']))

    with open(os.path.join(data_path,'data','GWAS_CN_AD12.fam'),'r') as infile:
        text = infile.read().strip().split('\n')

    PTID = [line.strip().split(' ')[1] for line in text]
        
    df_GWAS = df_bl[pd.DataFrame(df_bl.PTID.tolist()).isin(PTID).any(1).values]

    print('Label distribution on GWAS generated file')
    print(Counter(df_GWAS['DX_bl']))

    data = []
    with open(os.path.join(data_path,'data','GWAS_CN_AD12.ped'),'r') as infile:   
        text = infile.read().strip().split('\n')
        for line in text:
            gene = line.split(' ')[6:]
            PTID = line.split(' ')[1]
            AGE = df_GWAS[df_GWAS['PTID'] == PTID].AGE.item()
            GENDER = df_GWAS[df_GWAS['PTID'] == PTID].PTGENDER.item()
            EDU = df_GWAS[df_GWAS['PTID'] == PTID].PTEDUCAT.item()
            DIAG = df_GWAS[df_GWAS['PTID'] == PTID].DX_bl.item()
            GENOME = sequence_parser(gene)
            output = [PTID] + [AGE] + [GENDER] + [EDU] + [DIAG]+ GENOME
            data.append(output)


    with open(os.path.join(data_path,'data','top2000_snps.txt'),'r') as infile:
        snps = infile.read().strip().split('\n')

    column_names = ['PTID','AGE','GENDER','EDU']+['DIAG']+snps

    df_final = pd.DataFrame(data,columns=column_names)
    df_final.to_csv(os.path.join(data_path,'data','final_'+str(features)+'_GWAS12_data_Dx_bl.csv'))

    df_final = pd.read_csv(os.path.join(data_path,'data','final_'+str(features)+'_GWAS12_data_Dx_bl.csv'),na_values=["00"])
    df_final = df_final.iloc[:, 0:N+6] #Only top N snps
    df_final = df_final.drop(columns=['Unnamed: 0'])
    df_final.dropna(inplace=True)
    print('Label distribution on GWAS generated file after dropping Missing individuals')
    print(Counter(df_final.DIAG))
    df, y = data_prep(df_final,groups)
    print("Shape of final data BEFORE FEATURE SELECTION")
    print(df.shape, y.shape)
    STEP = int(df.shape[1]/20)
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
    final_N = df.shape[1]
    cat_columns = list(set(df.columns) - set(['AGE','EDU']))
    cat_columns_index = [i for i in range(len(df.columns)) if df.columns[i] in cat_columns]
    ########################################################################################
    #                       HYPERPARAMETER GRID SEARCH
    ########################################################################################
    #Adapted from #https://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html#sphx-glr-auto-examples-model-selection-plot-multi-metric-evaluation-py

    # Author: Raghav RV <rvraghav93@gmail.com>
    # License: BSD

    model = Pipeline([
            ('sampling', SMOTENC(sampling_strategy=0.7, k_neighbors=7, categorical_features = cat_columns_index,random_state=SEED)),
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
    plt.savefig(os.path.join(data_path,'results','Grid_search_Using'+'_'+str(features)+'features_for:'+groups+'.png'))

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
    plt.savefig(os.path.join(data_path,'results','ROC_for:'+groups+'_'+str(features)+'.png'))
    print('for total of ',final_N,"Features")
    print('Mean Balanced Accuracy:',sum(acc)/len(acc))
    print('Mean AUC:',sum(aucs)/len(aucs))

    imp = np.array(imp)
    imp = imp.mean(axis=0)

    imp_df = pd.DataFrame(columns=['features','importance'])
    imp_df['features'] = list(X.columns)
    imp_df['importance'] = imp

    imp_df_sorted = imp_df.sort_values(by=['importance'],ascending=False)
    imp_df_sorted.to_csv(os.path.join(data_path,'results',groups+'_Classification_ranked'+'_'+str(features)+'_features.csv'))

    print("END OF THE EXPERIMENT\n")

    plt.close('all')
    return sum(acc)/len(acc), sum(aucs)/len(aucs)


if  __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=int, help='Number of features to be used', default=100)
    parser.add_argument('--groups', type=str, help='binary classes to be classified ', default='CN_AD')
    parser.add_argument('--tuning', type=str, help='To perform hyperparameter sweep or not. Options:[sweep, no_sweep]', default='no_sweep')    
    args = parser.parse_args()
    
    if args.tuning == 'no_sweep':
        acc, my_auc = train_ADNI(features=args.features,
                groups = args.groups       
               )
    
    HyperParameters = edict()
    HyperParameters.groups =['CN_AD'] 
    HyperParameters.features= [100,200,300,400,500,750,1000,1250,1500]
    HyperParameters.params = [HyperParameters.features,HyperParameters.groups]  
    if args.tuning == 'sweep':
        params = list(itertools.product(*HyperParameters.params))
        for hp in params:
            print("For parameters:",hp)
            acc, my_auc = train_ADNI(
                features = hp[0],
                groups = hp[1]     
               )
            print(acc, my_auc)
            print('\n')
