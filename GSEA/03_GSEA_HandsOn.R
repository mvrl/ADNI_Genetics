rm(list=ls()); graphics.off(); closeAllConnections();
#!/usr/bin/env R

#------------------------------------------------------------------------------
# This code is a part of hands-on practice in BMI633
#
# Author   : Jong Cheol Jeong (JongCheol.Jeong@uky.edu)
# Copyright: Copyright 2019 University of Kentucky Cancer Research Informatics Shared Resource Facility
# License  : Copyright 2019 University of Kentucky Cancer Research Informatics Shared Resource Facility
# Buit     : March 16, 2020
# Update   : March 16, 2020
# Reference: http://bioconductor.org/packages/release/bioc/vignettes/DESeq2/inst/doc/DESeq2.html
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Install required packages
#------------------------------------------------------------------------------
if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager")
}

bio.packages <- c("airway", "DESeq2","apeglm", "ComplexHeatmap")
new.packages <- bio.packages[!(bio.packages %in% installed.packages()[,"Package"])]
if (length(new.packages) > 0) {
  BiocManager::install(new.packages)
}



library("airway")
library("DESeq2")
library("apeglm")
library("biomaRt")
library("ComplexHeatmap")
library("scales")

#------------------------------------------------------------------------------
# Parameters
#------------------------------------------------------------------------------
out_dir <- './'


#------------------------------------------------------------------------------
# Main
# Airway data is SummarizedExperiment class
# https://www.bioconductor.org/packages/release/bioc/vignettes/SummarizedExperiment/inst/doc/SummarizedExperiment.html
#------------------------------------------------------------------------------

# The airway package contains an example dataset from an RNA-Seq experiment of read counts per gene for airway smooth muscles.
# rows = features (e.g., gene, gtranscripts, exons, etc)
# cols = samples/experiments (e.g., sample name, cell line, )
data("airway")
se <- airway

# Play with SumarizedExperiment class
#------------------------------------------------------------------------------
rowData(se) # information about rows
names(se)   # get row names
rowRanges(se) # to view the range information GRangesList object, where each list element corresponds to one gene transcript and the ranges in each GRanges correspond to the exons in the transcript.

rowRanges(se)$ENSG00000001460 # search with row name

assays(se) # to list assay attributes
# Since airway data contains only 'counts' information so following commands are same
assays(se)$counts   # get experiment results: dataframe (genes x samples)
assay(se)           # get experiment results: dataframe (genes x samples)

assays(se[, se$dex == 'trt'])$counts
assay(se[, se$dex == 'trt'])

# column data contains sample information
colData(se)


# Read data based on Dexamethasone treatment and store them into CSV format
#------------------------------------------------------------------------------
bothSamp <- assays(se)$counts
treated <- assays(se[, se$dex == 'trt'])$counts
untreat <- assays(se[, se$dex == 'untrt'])$counts

dfTreat <- as.data.frame(treated)
dfUntreat <- as.data.frame(untreat)

out_treat <- paste0(out_dir, 'treated.csv')
write.csv(dfTreat, file=out_treat)

out_untreat <- paste0(out_dir, 'untreated.csv')
write.csv(dfUntreat, file=out_untreat)

out_both <- paste0(out_dir, 'treat_untreat.csv')
write.csv(bothSamp, file=out_both)



#------------------------------------------------------------------------------
# Running differential Expression
#------------------------------------------------------------------------------

# Step1: create DEseq data set from SummarizedExperiment data
#----------------------------------------------------------
ddsSE <- DESeqDataSet(se, design = ~ cell + dex)



# Step2: prefiltering remove reads < 10
#----------------------------------------------------------
keep <- rowSums(assays(ddsSE)$counts) >= 10
dds <- ddsSE[keep,]
#keep2 <- rowSums(counts(ddsSE)) >= 10
#dds2 <- ddsSE[keep2,]



# Step3 (Optional): change the order of experiment condition (i.e., dds$dex)
#         > dds$dex
#         [1] untrt trt   untrt trt   untrt trt   untrt trt
#         Levels: trt untrt
#         logarithmic fold change log2(trt/untrt)
#         + means OVER expressed 'trt'
#         - means UNDER expressed 'trt'
#----------------------------------------------------------
dds$dex <- factor(dds$dex, levels = c("trt", "untrt"))



# Step4: run differential expression analysis and store the results into CSV file
#----------------------------------------------------------
dds <- DESeq(dds)
resultsNames(dds) # check the name of possible outputs (~cell + dex)
res <- results(dds, contrast=c('dex', 'trt', 'untrt') ) # make sure the order is log(trt/untrt)
resSig <- subset(res, padj <= 0.05)

#-- add gene symbole information
mart <- useDataset("hsapiens_gene_ensembl", useMart("ensembl"))
df_resSig <- as.data.frame(resSig)
genes <- rownames(df_resSig)
df_resSig$ensemble_gene_id <- genes

#-- retrieve ensemble id and gene symbol information and merge them
G_list <- getBM(filters= "ensembl_gene_id", attributes= c("ensembl_gene_id","hgnc_symbol"), values=genes, mart=mart)
df_resSig_geneSymbol <- merge(df_resSig, G_list, by.x="ensemble_gene_id", by.y="ensembl_gene_id")
resOrdered <- df_resSig_geneSymbol[order(df_resSig_geneSymbol$log2FoldChange, decreasing=TRUE),]

#-- remove rows with no hgnc_symbol value
resOrdered <- resOrdered[!(is.na(resOrdered$hgnc_symbol) | resOrdered$hgnc_symbol==""), ]

out_DEorder <- paste0(out_dir, 'treat_vs_untreat_p05.csv')
write.csv(resOrdered, row.names=FALSE, file=out_DEorder)



# Step5: using 'apeglm' for LFC shrinkage. If used in published research, please cite:
#        Zhu, A., Ibrahim, J.G., Love, M.I. (2018) Heavy-tailed prior distributions for
#        sequence count data: removing the noise and preserving large differences.
#        Bioinformatics. https://doi.org/10.1093/bioinformatics/bty895
#----------------------------------------------------------
resultsNames(dds) # check the name of possible outputs (~cell + dex)
resLFC <- lfcShrink(dds, coef="dex_untrt_vs_trt", type="apeglm")
resLFCOrdered <- resLFC[order(resLFC$log2FoldChange, decreasing=TRUE),]
plotMA(resLFC, ylim=c(-2,2))

#-- add gene symbole information
mart <- useDataset("hsapiens_gene_ensembl", useMart("ensembl"))
df_resLFC <- as.data.frame(resLFC)
genes <- rownames(df_resLFC)
df_resLFC$ensemble_gene_id <- genes

#-- retrieve ensemble id and gene symbol information and merge them
G_list <- getBM(filters= "ensembl_gene_id", attributes= c("ensembl_gene_id","hgnc_symbol"), values=genes, mart=mart)
df_resLFC_geneSymbol <- merge(df_resLFC, G_list, by.x="ensemble_gene_id", by.y="ensembl_gene_id")
resLFC_Ordered <- df_resLFC_geneSymbol[order(df_resLFC_geneSymbol$log2FoldChange, decreasing=TRUE),]

#-- remove rows with no hgnc_symbol
resLFC_Ordered <- resLFC_Ordered[!(is.na(resLFC_Ordered$hgnc_symbol) | resLFC_Ordered$hgnc_symbol==""), ]

out_DEshrink <- paste0(out_dir, 'treat_vs_untreat_p05_shrink.csv')
write.csv(resLFC_Ordered, row.names=FALSE, file=out_DEshrink)



# Step6: Drawing heatmaps
#----------------------------------------------------------

#-- select top and bottom Ensemble IDs
top50 <- head(resLFC_Ordered, n=50)$ensemble_gene_id
btm50 <- tail(resLFC_Ordered, n=50)$ensemble_gene_id
selRows <- c(top50, btm50)

#-- combine original data and choose the slected rows
exData <- data.matrix(cbind(dfTreat, dfUntreat))
exData <- exData[selRows, ]

#-- normalize data between 0 to 1
norm_exData <- apply(exData, 2, scales::rescale)

colSplit <- c(rep('Treated', 4), rep('Untreated', 4))
ht <- Heatmap(norm_exData, name="Expression Level", cluster_rows=TRUE, cluster_columns=FALSE, show_column_dend=FALSE,
        row_names_gp = gpar(fontsize=8),
        column_split = factor(colSplit, levels = c('Treated', 'Untreated')),
        column_names_gp = gpar(col = c("orange", "purple"), fontsize = c(8))
        )

pdfW <- 7 # width inch
pdfH <- 12 # height inch
out_ht <- paste0(out_dir, 'DEheatmap3.pdf')
pdf(file=out_ht, width=pdfW, height=pdfH)
draw(ht)
dev.off()
