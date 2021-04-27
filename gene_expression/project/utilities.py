#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 18:16:18 2021
Few utility functions like plot ROC, SHAP plots
@author: subashkhanal
"""

import matplotlib.pyplot as plt
import shap
import numpy as np
from sklearn.metrics import roc_curve, auc
import pandas as pd
import os

def plot_ROC(tprs,aucs,fname,savepath):
    figname = os.path.join(savepath,'ROC',fname+'.png')
    
    mean_fpr = np.linspace(0, 1, 250)
    
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
     label='Groudline', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
     label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
     lw=2, alpha=.8)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
             label=r'$\pm$ 1 std. dev.')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig(figname)
    plt.close('all')

def plot_SHAP(model,X,cols,fname,savepath):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    f = shap.summary_plot(shap_values, pd.DataFrame(X,columns=cols),show=False, max_display= 25, plot_size= (100.0, 50.0))
    figname = os.path.join(savepath,'SHAP',fname+'.png')
    plt.savefig(figname)
    plt.close()
    
def save_results(acc, f1,auc_score, importance, cols, fname, results_path):
    feature_importance = pd.DataFrame(importance,index = cols, columns=['importance']).sort_values('importance',ascending=False)
    feature_importance.to_csv(os.path.join(results_path,fname+'.csv'))
    txtfilename = os.path.join(results_path,'results.txt')
    Result = '\nFor: '+fname+'\n'+'Averaged ACC='+str(acc)+'\n'+'Averaged F1='+str(f1)+'\n'+'Averaged AUC='+str(auc_score)+'\n'
    with open(txtfilename,'a') as outfile:
        outfile.write(Result)
        
    
    
