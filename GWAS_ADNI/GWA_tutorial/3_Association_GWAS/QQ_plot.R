install.packages("qqman",repos="https://cran.r-project.org/package=qqman",lib="~" ) # location of installation can be changed but has to correspond with the library location 
library("qqman") 
results_log <- read.table("logistic_results.assoc_2.logistic", head=TRUE)
jpeg("QQ-Plot_logistic.jpeg")
qq(results_log$P, main = "Q-Q plot of GWAS p-values : log")
dev.off()

#results_as <- read.table("assoc_results.assoc", head=TRUE)
#jpeg("QQ-Plot_assoc.jpeg")
#qq(results_as$P, main = "Q-Q plot of GWAS p-values : log")
#dev.off()

