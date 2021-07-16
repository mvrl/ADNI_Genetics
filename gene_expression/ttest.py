#This is a script to perform ttset on ADNI gene expression data between the different binary groups. The resulting ranking of the probes will be used for first stage feature selection.
import pandas as pd
import os
from collections import Counter
from scipy import stats
from scipy.stats import ttest_ind
import statsmodels.stats.multitest as multi

FOLDS = 5
fdr_alpha =0.10 ##DEFAULT IS 0.05
CV_path = '/mnt/gpfs2_16m/pscratch/nja224_uksr/SKH259/LinLab/ADNI_Genetics/gene_expression/data'
final_path = CV_path
df_orig = pd.read_csv(os.path.join(final_path,'Unfiltered_gene_expr_dx.csv'),low_memory=False)
df_gene = df_orig.loc[:,'11715100_at_HIST1H3G': 'AFFX-r2-TagQ-5_at_nan'].astype('float') #Extract only the probes
gene_probes = df_gene.columns
subs_df = pd.read_csv(os.path.join(final_path,'CV_folds.csv'))
groups = ['CN_AD','CN_EMCI','CN_LMCI','EMCI_LMCI','EMCI_AD','LMCI_AD']
groups_col = ['CN_AD','CN_AD_c','CN_EMCI','CN_EMCI_c','CN_LMCI','CN_LMCI_c','EMCI_LMCI','EMCI_LMCI_c','EMCI_AD','EMCI_AD_c','LMCI_AD','LMCI_AD_c']

for fold in range(FOLDS):
    print('for fold',fold)    
    fname = os.path.join(CV_path,'fold'+str(fold)+'_t_test_0.10_geneExpr_Unfiltered_bl.csv')
    df_p = pd.DataFrame(columns=groups_col, index = gene_probes) #dataframe for collecting p values for test
    for group in groups:
        print('for group:',group) #CN_EMCI_fold0_train
        train_subs = list(subs_df['_'.join([group,'fold'+str(fold),'train'])]) #Use only training data to rank features
        df = df_orig[pd.DataFrame(df_orig['Unnamed: 0'].tolist()).isin(train_subs).any(1).values]
        print(Counter(df['DX_bl']))
        
        diag1 = group.split('_')[0]
        diag2 = group.split('_')[1]

        df_1 = df[df['DX_bl']==diag1]
        df_2 = df[df['DX_bl']==diag2]

        df_1_exp = df_1.loc[:,'11715100_at_HIST1H3G': 'AFFX-r2-TagQ-5_at_nan'].astype('float')
        df_2_exp = df_2.loc[:,'11715100_at_HIST1H3G': 'AFFX-r2-TagQ-5_at_nan'].astype('float')
        p = []
        data = df_1_exp
        for col in data.columns:
            p.append(stats.normaltest(data[col])[1])
        print("Average P value for normality test for all features for ",diag1," group is:",sum(p)/len(p))
        # null hypothesis: x comes from a normal distribution

        p = []
        data = df_2_exp
        for col in data.columns:
            p.append(stats.normaltest(data[col])[1])
            
        print("Average P value for normality test for all features for ",diag2," group is:",sum(p)/len(p))
        # null hypothesis: x comes from a normal distribution

        alpha = fdr_alpha
        
        for gene in gene_probes:
            data1 = df_1_exp[gene]
            data2 = df_2_exp[gene]
            stat, p = ttest_ind(data1, data2)
            df_p.loc[gene,group] = p
        hyp, corr_p = multi.fdrcorrection(df_p[group],alpha = fdr_alpha) #FDR correction 
        df_p[group+'_c'] = corr_p 
                 
    df_p.to_csv(fname)

    print("For all folds: Ttest calculated for all diagnostic binary groups in Gene Expression data")
