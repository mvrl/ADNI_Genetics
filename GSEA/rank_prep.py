#This prepares rank file from the provided Probefiltered ranked gene list
import os
import pandas as pd

group='CN_Dementia'
features = 2000
df = pd.read_csv('t_test_geneExpr_Probefiltered.csv')
ttest = df.sort_values(group).sort_values(group+'_c').iloc[0:features] #suffix _c to use the FDR corrected p values

print(ttest.head())
print(ttest.iloc[0])
with open('genes_ranked.rnk', 'w') as outfile:
    text= "#Gene\tpvalue\n"
    for i in range(len(ttest)):
        gene = ttest.iloc[i]['Gene']
        if "||" in gene:
            gene = gene.split(' ')[0].strip()
        pvalue = ttest.iloc[i][group]

        text = text + gene+ "\t" + str(pvalue) + "\n"

    text = text.strip()

    outfile.write(text)




