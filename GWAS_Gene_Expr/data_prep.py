#Data prep script for the combined data experiment

import os
import pandas as pd

Gene_Expr_data_path = '/mnt/gpfs2_16m/pscratch/nja224_uksr/SKH259/LinLab/ADNI_Genetics/GWAS_Gene_Expr/data/Unfiltered_gene_expr_dx.csv'
Gene_Expr_rank_path = '/mnt/gpfs2_16m/pscratch/nja224_uksr/SKH259/LinLab/ADNI_Genetics/GWAS_Gene_Expr/data/t_test_0.10_geneExpr_Unfiltered_bl.csv'

GWAS_data_path = '/mnt/gpfs2_16m/pscratch/nja224_uksr/SKH259/LinLab/ADNI_Genetics/GWAS_Gene_Expr/data/final_SNP_data.csv'
GWAS_rank_path = '/mnt/gpfs2_16m/pscratch/nja224_uksr/SKH259/LinLab/ADNI_Genetics/GWAS_Gene_Expr/data/top2000_snps.csv'

data_path = '/mnt/gpfs2_16m/pscratch/nja224_uksr/SKH259/LinLab/ADNI_Genetics/GWAS_Gene_Expr/data/'

expr = pd.read_csv(Gene_Expr_data_path)
expr_rank = pd.read_csv(Gene_Expr_rank_path)

gwas = pd.read_csv(GWAS_data_path)
gwas_rank = pd.read_csv(GWAS_rank_path)

common_PTID = list(set(expr['Unnamed: 0']).intersection(set(gwas.PTID)))

common_expr = expr[expr['Unnamed: 0'].isin(common_PTID)]
common_GWAS = gwas[gwas['PTID'].isin(common_PTID)]

#Save these
print(common_expr.shape)
common_expr.to_csv(os.path.join(data_path,'common_expr.csv'))
print(common_GWAS.shape)
common_GWAS.to_csv(os.path.join(data_path,'common_gwas.csv'))



