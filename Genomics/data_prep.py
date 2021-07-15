#This is data preparatory script for the SNP data
import os
from collections import Counter
import numpy as np                         
import pandas as pd
import shlex
import subprocess
from tqdm import tqdm

#BEFORE RUNNING THIS CONVERT .BED FILES FOR EACH TRAIN AND TEST FOLDS TO .PED TO READ LIKE TEXT FILE
# Being in QualityControl folder for each train and test folder for each CV fold run following:
#plink --bfile GWAS_1_2_3_clean_CN_AD12 --recode --out GWAS_1_2_3_clean_CN_AD12
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

#Inefficient approach! but is easier to visualize in my head
def data_prep(df,groups): #This takes the dataframe and returns the one hot encoded expansion of input features

    target = prepare_targets(list(df.DIAG),groups)
    df1 = df.drop(columns=['PTID','DIAG']).reset_index(drop=True) #Patient ID and DIAG not needed
    num_cols = ['AGE','EDU']
    cat_cols = [i for i in df1.columns if i not in num_cols]  #Categorical features
    expand_cat = [] #List of expanded categorical columns
    for cat in cat_cols:
        expand_cat = expand_cat + [str(cat)+'_'+ str(c) for c in list(set(df1[cat]))]
    nan_cols = [col for col in expand_cat if '_00' in col or '_nan' in col]
    nonnan_cols = [i for i in expand_cat if i not in nan_cols]
    df_out = pd.DataFrame(columns=list(nonnan_cols))
    for i in tqdm(range(len(df1))):
        row = df1.iloc[i]
        for col in nonnan_cols:
            main_col = '_'.join(col.split('_')[:2]) #column name in original dataFrame
            item = row[main_col]
            df_out.at[i,str(main_col)+'_'+ str(item)] = str(1)  #One hot encoding

    nan_cols = [col for col in df_out.columns if '_nan' in col]  
    df_out = df_out.drop(columns=nan_cols)
    df_out['AGE'] = df1.AGE
    df_out['EDU'] = df1.EDU
    df_out = df_out.fillna(int(0)) #One hot encoding
    return df_out, target.ravel()

def GWAS_data_prep(root_path,data_path,top_snp_path,train_test_flag,groups='CN_AD'):

    # To be able to run this following files should be ready ADNIMERGE.csv from the ADNI website, GWAS_CN_AD12.{fam,ped,map} and top2000_snps.csv
    # top2000_snps.csv is the list of top 2000 SNPs as shown in the Association analysis step in GWAS using PLINK (Refer GWAS_ADNI folder in this repo)
    # Using top2000_snps.txt, only those SNPs are extracted from the files "GWAS_1_2_3_clean_CN_AD.{fam,bed,bim} produced at the end of Quality Control 
    #step in GWAS using PLINK. (Refer GWAS_ADNI folder in this repo)
    # Now the curated SNP data (with only 2000 SNPs) is converted to ped file set: GWAS_CN_AD12.{fam,ped,map} using --recode option in PLINK

    # In summary: Extract top 2000 snps from association analyis results, use it to curate the Quality Controlled and SNP filtered dataset,
    # Finally, convert it to .ped and .map file to read like regular text file
    final_data_path = os.path.join(data_path,'final_SNP_data.csv')
    df = pd.read_csv(os.path.join(root_path,'ADNIMERGE.csv'),low_memory=False)
    df_bl = df[df['VISCODE']=='bl']
    print('Overall label distribution on ADNIMERGE.csv')
    print(Counter(df[df['VISCODE']=='bl']['DX_bl']))
    data_file = os.path.join(data_path,'GWAS_1_2_3_clean_CN_AD12')
    filename = 'GWAS_1_2_3_clean_CN_AD12'
    
    with open(data_file+'.fam','r') as infile:
        text = infile.read().strip().split('\n')

    PTIDs = [line.strip().split(' ')[1] for line in text]
        
    df_GWAS = df_bl[pd.DataFrame(df_bl.PTID.tolist()).isin(PTIDs).any(1).values]

    print('Label distribution on GWAS generated file')
    print(Counter(df_GWAS['DX_bl']))

    # os.chdir(data_path)

    # plink_command = "plink --bfile "+filename+" --recode --out "+filename #Convert .bed file to .ped to read like text file
    # process = subprocess.Popen(shlex.split(plink_command),start_new_session=True)

    data = []
    print(data_file+'.ped')
    with open(data_file+'.ped','r') as infile:  
        text = infile.read().strip().split('\n')
        for line in tqdm(text):
            gene = line.split(' ')[6:]
            PTID = line.split(' ')[1]
            AGE = df_GWAS[df_GWAS['PTID'] == PTID].AGE.item()
            EDU = df_GWAS[df_GWAS['PTID'] == PTID].PTEDUCAT.item()
            DIAG = df_GWAS[df_GWAS['PTID'] == PTID].DX_bl.item()
            GENOME = sequence_parser(gene)
            output = [PTID] + [AGE] + [EDU] + [DIAG]+ GENOME
            data.append(output)
    print("Read ped file")
    snps = []
    with open(data_file+'.map','r') as infile:
        text = infile.read().strip().split('\n')
        for line in text:
            snps.append(line.split('\t')[0]+'_'+line.split('\t')[1])
    print("Read map file")
    column_names = ['PTID','AGE','EDU']+['DIAG']+snps

    df_final = pd.DataFrame(data,columns=column_names)
    # df_final.to_csv(os.path.join(data_path,'final_GWAS_data_Dx_bl.csv'))
    # df_final = pd.read_csv(os.path.join(data_path,'final_GWAS_data_Dx_bl.csv'),na_values=["00"])
    
    my_snps = list(pd.read_csv(os.path.join(top_snp_path,'top2000_snps.csv'))['top_snps'])
    selected_features = ['PTID','AGE','EDU']+['DIAG'] + my_snps
    df_final = df_final.reindex(columns = selected_features)
    print(df_final.shape)
    df, y = data_prep(df_final,groups)
    print("Prepared final SNP data")
    df['DIAG'] = y
    df['PTID'] = PTIDs
    df.to_csv(final_data_path)

    return PTIDs,y


#################################################################################################

FOLDS = 5
cv_path = '/mnt/gpfs2_16m/pscratch/nja224_uksr/SKH259/LinLab/ADNI_Genetics/Genomics/data/GWAS/CN_AD/cv_folds/'
root_path = '/mnt/gpfs2_16m/pscratch/nja224_uksr/SKH259/LinLab/ADNI_Genetics/Genomics/data/GWAS/CN_AD/'

col_names = []
group = 'CN_AD'
for i in range(FOLDS):
    col = ['_'.join([group,'fold'+str(i),'train']),'_'.join([group,'fold'+str(i),'test'])]
    col_names = col_names + col
df = pd.DataFrame(columns=col_names)

for i in range(FOLDS):
    fold_path = os.path.join(cv_path,'fold'+str(i))
    QC_train_path = os.path.join(fold_path,'train','QualityControl')
    QC_test_path = os.path.join(fold_path,'test','QualityControl')
    top_snp_path = fold_path
    train_subs,y_train = GWAS_data_prep(root_path,QC_train_path,top_snp_path,train_test_flag='qc_train')#Prep data for QC SNP data 
    test_subs,y_test = GWAS_data_prep(root_path,QC_test_path,top_snp_path,train_test_flag='qc_test')##Prep data for test data with QC

    print('label dist of QC train',Counter(y_train))
    print('label dist of QC test',Counter(y_test))
    print("data prep for fold",str(i),'complete')

    df['_'.join([group,'fold'+str(i),'train'])] = pd.Series(train_subs)
    df['_'.join([group,'fold'+str(i),'test'])] = pd.Series(test_subs)

print("data prep for ALL FOLDS complete")

df.to_csv(os.path.join(root_path,'CV_folds_QC.csv'))




