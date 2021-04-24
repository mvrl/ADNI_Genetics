#This prepares rank file from the provided Probefiltered ranked gene list
import os
import pandas as pd

group='CN_AD'
features = 2000
#rank_file = 't_test_geneExpr_Probefiltered.csv'
rank_file = 't_test_0.05_geneExpr_Unfiltered_bl.csv'
#rank_file = 'GradientBoosting_1000_CN_AD_Unfiltered_50_extra_DX_bl.csv'
#rank_file = 'GradientBoosting_700_CN_AD_Unfiltered_500_no_extra_DX_bl.csv'
df = pd.read_csv(rank_file)
if 't_test' in rank_file:
    rank = df.sort_values(group).sort_values(group+'_c').iloc[0:features] #suffix _c to use the FDR corrected p values
else:
    rank = df.sort_values('importance')

with open(rank_file.split('.')[0]+'.rnk', 'w') as outfile:
    text= "#Gene\tscore\n"
    for i in range(len(rank)):
        if 't_test' in rank_file:
            gene = rank.iloc[i]['Gene']
            if gene in text or gene == 'nan':
                continue
            if "||" in gene:
                gene = gene.split(' ')[0].strip()
            score = rank.iloc[i][group]
        else:
            probe_gene = rank.iloc[i]['Gene'].split('_at_')
            if len(probe_gene) == 2:
                gene = probe_gene[1]
            if gene in text or gene == 'nan':
                continue
            if "||" in gene:
                gene = gene.split(' ')[0].strip()
            score = rank.iloc[i]['importance']

        text = text + gene+ "\t" + str(score) + "\n"

    text = text.strip()

    outfile.write(text)




