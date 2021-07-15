#I know there should be easir awk based solution to this but I am using python for it as I am not
#Quite an expert in shell scripting, YET! :)

#This simply joins the Principal components obtained for the QC genetic data with its other covariates

import os
def overall_cov_creator(FOLD):

    diagnosis='DX_bl' # DX or DX_bl from ADNIMERGE.csv column
    control='CN'
    delimiter='_'
    case='AD'
    analysis=control+delimiter+case
    FOLD = FOLD
    PoplnStrat_path="/mnt/gpfs2_16m/pscratch/nja224_uksr/SKH259/LinLab/ADNI_Genetics/Genomics/data/GWAS/CN_AD/cv_folds/fold"+FOLD+"/train/PoplnStrat/"
    root_path="/mnt/gpfs2_16m/pscratch/nja224_uksr/SKH259/LinLab/ADNI_Genetics/GWAS_ADNI/"
    cov_path="/mnt/gpfs2_16m/pscratch/nja224_uksr/SKH259/LinLab/ADNI_Genetics/Genomics/data/GWAS/CN_AD/cv_folds/fold"+FOLD+"/train/cov_pheno.txt"
    pca_path= os.path.join(PoplnStrat_path,"GWAS_1_2_3_clean_"+analysis+"12.pca")
    final_cov_path ="/mnt/gpfs2_16m/pscratch/nja224_uksr/SKH259/LinLab/ADNI_Genetics/Genomics/data/GWAS/CN_AD/cv_folds/fold"+FOLD+"/train/final_cov.txt"

    with open(pca_path,'r') as pca, open(cov_path,'r') as cov, open(final_cov_path,'w') as final:
        p = pca.read().strip().split('\n')
        c = cov.read().strip().split('\n')
        header = c[0].strip()+' '+' '.join(p[0].strip().split(' ')[2:]) + '\n'
        p = p[1:]
        c = c[1:]
        text=header

        for line in p:
            fid1 = line.split(' ')[0]
            iid1 = line.split(' ')[1]
            for l in c:
                fid2 = l.split(' ')[0]
                iid2 = l.split(' ')[1]

                if (fid1 == fid2) & (iid1 == iid2):
                    text = text + l + ' ' + ' '.join(line.strip().split(' ')[2:]) + '\n'
    
        text = text.strip()
        final.write(text)

if  __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--FOLD', type=str, help='which training fold to do GWAS for', default='0')
    args = parser.parse_args()

    overall_cov_creator(args.FOLD)
