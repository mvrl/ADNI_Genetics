from easydict import EasyDict as edict

cfg = edict()
HyperParameters = edict()

### Data locations
cfg.data = '/home/skh259/LinLab/LinLab/MLforAlzheimers/data/ADNI/genetics/final/'
cfg.ttest = '/home/skh259/LinLab/LinLab/MLforAlzheimers/data/ADNI/genetics/final/'
### Storage for results 
cfg.results = '/home/skh259/LinLab/LinLab/MLforAlzheimers/ADNI/gene_expression/results/'
### Storage for plots  
cfg.plots = '/home/skh259/LinLab/LinLab/MLforAlzheimers/ADNI/gene_expression/plots/'
#Number of times to repeat experiments for
cfg.repeat = 5

##HyperParameters
HyperParameters.features = [50,250,500,750,1000] #number of gene features to be selected
HyperParameters.classifier = ['GradientBoosting','RandomForestClassifier']
#HyperParameters.classifier = ['GradientBoosting']
HyperParameters.estimators = [100,300,500,700,1000]
#HyperParameters.classes = ['CN_Dementia','MCI_Dementia'] #'CN_MCI' did not have any significant genes based on FDR corrected ttest
HyperParameters.classes = ['CN_AD','CN_EMCI','CN_LMCI', 'EMCI_LMCI','EMCI_AD','LMCI_AD'] #diagnosis based on baseline visit
HyperParameters.filtered = ['Unfiltered'] #Unfiltered had better performance
HyperParameters.extra_feats = ['extra','no_extra'] #For comparision how much ['AGE','PTGENDER','APOE4','PTEDUCAT'] would improve the performance
HyperParameters.params = [HyperParameters.features,HyperParameters.classifier,HyperParameters.estimators,HyperParameters.classes,HyperParameters.filtered,HyperParameters.extra_feats]





