#This is a script to perform ttset on ADNI gene expression data between the different binary groups. The resulting ranking of the probes will be used for first stage feature selection.
import pandas as pd
import os
from collections import Counter
from scipy import stats
from scipy.stats import ttest_ind
import statsmodels.stats.multitest as multi

final_path = '/home/skh259/LinLab/LinLab/MLforAlzheimers/data/ADNI/genetics/final'
df = pd.read_csv(os.path.join(final_path,'Unfiltered_gene_expr_dx.csv'),low_memory=False)

print(Counter(df['DX_bl']))

df_CN = df[df['DX_bl']=='CN']
df_AD = df[df['DX_bl']=='AD']
df_EMCI = df[df['DX_bl']=='EMCI']
df_LMCI = df[df['DX_bl']=='LMCI']


df_CN_exp = df_CN.loc[:,'11715100_at_HIST1H3G': 'AFFX-r2-TagQ-5_at_nan'].astype('float')
df_AD_exp = df_AD.loc[:,'11715100_at_HIST1H3G': 'AFFX-r2-TagQ-5_at_nan'].astype('float')
df_EMCI_exp = df_EMCI.loc[:,'11715100_at_HIST1H3G': 'AFFX-r2-TagQ-5_at_nan'].astype('float')
df_LMCI_exp = df_LMCI.loc[:,'11715100_at_HIST1H3G': 'AFFX-r2-TagQ-5_at_nan'].astype('float')

p = []
data = df_CN_exp
for col in data.columns:
    p.append(stats.normaltest(data[col])[1])
    
print("Average P value for normality test for all features for CN group is:",sum(p)/len(p))
# null hypothesis: x comes from a normal distribution

p = []
data = df_EMCI_exp
for col in data.columns:
    p.append(stats.normaltest(data[col])[1])
    
print("Average P value for normality test for all features for EMCI group is:",sum(p)/len(p))
# null hypothesis: x comes from a normal distribution

p = []
data = df_LMCI_exp
for col in data.columns:
    p.append(stats.normaltest(data[col])[1])
    
print("Average P value for normality test for all features for LMCI group is:",sum(p)/len(p))
# null hypothesis: x comes from a normal distribution

p = []
data = df_AD_exp
for col in data.columns:
    p.append(stats.normaltest(data[col])[1])
    
print("Average P value for normality test for all features for AD group is:",sum(p)/len(p))
# null hypothesis: x comes from a normal distribution



fdr_alpha =0.10 ##DEFAULT IS 0.05
alpha = fdr_alpha
gene_probes = df_AD_exp.columns
groups = ['CN_AD','CN_AD_c','CN_EMCI','CN_EMCI_c','CN_LMCI','CN_LMCI_c', 'EMCI_LMCI','EMCI_LMCI_c','EMCI_AD','EMCI_AD_c','LMCI_AD','LMCI_AD_c']
df_p = pd.DataFrame(columns=groups, index = gene_probes) #dataframe for collecting p values for test
for gene in gene_probes:
    data1 = df_CN_exp[gene]
    data2 = df_AD_exp[gene]
    stat, p = ttest_ind(data1, data2)
    df_p.loc[gene,'CN_AD'] = p
hyp, corr_p = multi.fdrcorrection(df_p['CN_AD'],alpha = fdr_alpha) #FDR correction 
df_p['CN_AD_c'] = corr_p 
       

for gene in gene_probes:
    data1 = df_CN_exp[gene]
    data2 = df_EMCI_exp[gene]
    stat, p = ttest_ind(data1, data2)
    df_p.loc[gene,'CN_EMCI'] = p
hyp, corr_p = multi.fdrcorrection(df_p['CN_EMCI'],alpha = fdr_alpha) #FDR correction
df_p['CN_EMCI_c'] = corr_p 

for gene in gene_probes:
    data1 = df_CN_exp[gene]
    data2 = df_LMCI_exp[gene]
    stat, p = ttest_ind(data1, data2)
    df_p.loc[gene,'CN_LMCI'] = p
hyp, corr_p = multi.fdrcorrection(df_p['CN_LMCI'],alpha = fdr_alpha) #FDR correction
df_p['CN_LMCI_c'] = corr_p 

for gene in gene_probes:
    data1 = df_EMCI_exp[gene]
    data2 = df_LMCI_exp[gene]
    stat, p = ttest_ind(data1, data2)
    df_p.loc[gene,'EMCI_LMCI'] = p
hyp, corr_p = multi.fdrcorrection(df_p['EMCI_LMCI'],alpha = fdr_alpha) #FDR correction 
df_p['EMCI_LMCI_c'] = corr_p 
       

for gene in gene_probes:
    data1 = df_EMCI_exp[gene]
    data2 = df_AD_exp[gene]
    stat, p = ttest_ind(data1, data2)
    df_p.loc[gene,'EMCI_AD'] = p
hyp, corr_p = multi.fdrcorrection(df_p['EMCI_AD'],alpha = fdr_alpha) #FDR correction
df_p['EMCI_AD_c'] = corr_p 

for gene in gene_probes:
    data1 = df_LMCI_exp[gene]
    data2 = df_AD_exp[gene]
    stat, p = ttest_ind(data1, data2)
    df_p.loc[gene,'LMCI_AD'] = p
hyp, corr_p = multi.fdrcorrection(df_p['LMCI_AD'],alpha = fdr_alpha) #FDR correction
df_p['LMCI_AD_c'] = corr_p 

df_p.to_csv(os.path.join(final_path,'t_test_0.10_geneExpr_Unfiltered_bl.csv'))

print("Ttest calculated for all the unfiltered probes among all baseline diagnostic groups in Gene Expression data")
