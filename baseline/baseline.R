#!/usr/bin/env Rscript
#
# This command line tool runs R baseline methods sequentially to transfer
# annotations between a training dataset and target dataset, 
# provided as paths to AnnData objects.
# 
# ARGUMENTS
# input_counts : path to input anndata
# test_counts : path to test counts
# cell_type_col : string of cell type column in `adata.obs`
# model_path : path to model outputs including train and val sets
# out_path : path for outputs
#
args = commandArgs(trailingOnly=TRUE)

print('ARGUMENTS')
for (i in 1:length(args)){
    print(args[i])
}

cell_type_col = args[3]
print(paste("Cell type column is", cell_type_col))

# Run baseline benchmarks using R packages
library(reticulate)
library(SingleCellExperiment)
library(singleCellNet)
library(ramify) # for `argmax`
library(stringr) # for `str_replace_all`
library(irr) # for `kappa2`

######################################################
# Load anndata objects
######################################################

#' Convert a python anndata object loaded with `reticulate`
#' into a set of R objects
#'
ad2robj = function(adata){
    X = reticulate::py_to_r(adata$X)
    # transpose to follow R convention
    X = t(X)
    
    obs = reticulate::py_to_r(adata$obs)
    obs["cell_barcode"] = rownames(obs)
    var = reticulate::py_to_r(adata$var)
    var_names = rownames(var)
    
    # set row and column names on the matrix
    rownames(X) = var_names
    colnames(X) = rownames(obs)
    
    return(list(X=X, obs=obs, var=var, var_names=var_names))
}

print('Importing anndata')
anndata = reticulate::import("anndata", convert=FALSE)

# Load training data and convert to a single cell experiment
print('Loading train and test data')
train_adata = anndata$read_h5ad(args[1])
train_robj = ad2robj(train_adata)

test_adata = anndata$read_h5ad(args[2])
test_robj = ad2robj(test_adata)

######################################################
# Load training and testing indices
######################################################

print('Loading train idx')
train_idx = read.csv(
    file.path(args[4], 'train_idx.csv'),
    header=FALSE
)$V1

######################################################
# Fit LIGER harmonization
######################################################

# time LIGER runs
start_time = proc.time()

library(liger)
library(e1071) # contains `svm()` function
library(FNN) # contains `knn()` function

# subset data to appropriate cells and genes
common_genes = intersect(
    train_robj$var_names,
    test_robj$var_names
)

train_data = train_robj$X[,train_idx]
train_data = train_data[common_genes,]
train_meta = train_robj$obs[train_idx,]
test_data = test_robj$X
test_data = test_data[common_genes,]
test_meta = test_robj$obs

ligerex = createLiger(
    list(train = train_data, test = test_data)
)
ligerex = normalize(ligerex)
ligerex = selectGenes(ligerex, var.thresh = 0.1)
ligerex = scaleNotCenter(ligerex)

start_time = proc.time()

ligerex = optimizeALS(ligerex, k = 20)
ligerex = quantileAlignSNF(ligerex) # SNF clustering and quantile alignment

# extract the integrated NMF embedding
# H is the matrix of iNMF activity scores
H_train = ligerex@H$train # [n_train_cells, k]
H_test = ligerex@H$test   # [n_test_cells, k]
# `ligerex` also contains `ligerex@H`, which is the above
# two matrices row-wise concatenated

# train a kNN classifier
# `pred` is a factor of length [n_test,]
# with predictions for each cell in the target dataset
knn_pred = FNN::knn(
    train=H_train,
    test=H_test,
    cl=train_meta[[cell_type_col]], # ground truth labels. data.frame[[]] gets a col as a vector.
    k=15
)
write.csv(
    knn_pred,
    file.path(args[5], 'liger_knn_preds.csv')
)

# train an SVM classifier
# SVMs were our best performing baseline, so this is the 
# best comparison.
# Cost is equivalent to `C` in the `sklearn` implementation
# `linear` kernel is equivalent to `sklearn.svm.LinearSVC`,
# but doesn't appear to have the same performance benefits

svm_data = data.frame(H_train)
svm_data['y'] = as.factor(train_meta[[cell_type_col]])
model = svm(y=svm_data[['y']], x=H_train, kernel='linear', cost=1.0)
svm_pred = predict(model, data.frame(H_test))
write.csv(
    svm_pred,
    file.path(args[5], 'liger_svm_preds.csv')
)

# compute accuracies
gt = test_meta[[cell_type_col]]
n_total = length(gt)

knn_pred_v = as.vector(knn_pred)

n_correct = sum(knn_pred_v == gt)
liger_knn_acc = n_correct/n_total

stopifnot(all(rownames(test_meta) == names(svm_pred)))

svm_pred_v = as.vector(svm_pred)

n_correct = sum(svm_pred_v == gt)
liger_svm_acc = n_correct/n_total

K = irr::kappa2(data.frame(p=knn_pred_v, t=gt))
liger_knn_kappa = K$value
K = irr::kappa2(data.frame(p=svm_pred_v, t=gt))
liger_svm_kappa = K$value

print('LIGER kNN Accuracy:')
print(liger_knn_acc)
print('LIGER SVM Accuracy:')
print(liger_svm_acc)

write.csv(
    data.frame(
        liger_knn_acc=liger_knn_acc,
        liger_knn_kappa=liger_knn_kappa,        
        liger_svm_acc=liger_svm_acc,
        liger_svm_kappa=liger_svm_kappa
    ), 
    file=file.path(args[5], 'liger_acc.csv'), 
    row.names=F
)


ligerex = runUMAP(ligerex)

# save UMAP coordinates for downstream visualization
write.csv(
    ligerex@tsne.coords,
    file.path(args[5], 'liger_umap_coords.csv')
)

end_time = proc.time()

diff_time = end_time-start_time

line2write = paste("@@@", "Runtime", "LIGER", diff_time[3], sep=",")
print(line2write)
write(line, file=file.path(args[5], 'r_baseline_runtimes.txt'), append=TRUE)

# extract the iNMF coefficients
W = ligerex@W # [k, n_common_genes]

# save LIGER matrices
write.csv(
    H_train,
    file.path(args[5], 'liger_H_train.mtx')
)
write.csv(
    H_test,
    file.path(args[5], 'liger_H_test.mtx')
)
write.csv(
    W,
    file.path(args[5], 'liger_W.mtx')
)

######################################################
# Train scmap
######################################################
library(scmap)
print('Preparing data for scmap')

# subset data to appropriate cells and genes
common_genes = intersect(
    train_robj$var_names,
    test_robj$var_names
)

train_data = train_robj$X[,train_idx]
train_data = train_data[common_genes,]
train_meta = train_robj$obs[train_idx,]
test_data = test_robj$X
test_data = test_data[common_genes,]
test_meta = test_robj$obs

# scmap requires cell types to be named `cell_type1`
train_meta['cell_type1'] = train_meta[cell_type_col]

train_sce = SingleCellExperiment(
    assays = list(logcounts = as.matrix(train_data)),
    colData = DataFrame(celltypes = train_meta['cell_type1']),
)
counts(train_sce) = logcounts(train_sce)
normcounts(train_sce) = logcounts(train_sce)
rowData(train_sce)$feature_symbol <- rownames(train_sce)
colnames(colData(train_sce)) = c("cell_type1")

test_sce = SingleCellExperiment(
    assays = list(logcounts = as.matrix(test_data))
)
counts(test_sce) = logcounts(test_sce)
normcounts(test_sce) = logcounts(test_sce)

rowData(test_sce)$feature_symbol <- rownames(test_sce)


# feature selection
n_features = 1000
train_sce <- scmap::selectFeatures(
    train_sce, 
    n_features=n_features, 
    suppress_plot = T
)

########################
# scmap-cluster 
########################

# Fit scmap-cluster
train_sce <- scmap::indexCluster(train_sce)

# predict with scmap-cluster
scmap_cluster_results <- scmap::scmapCluster(
  projection = test_sce, 
  index_list = list(
    train_data = metadata(train_sce)$scmap_cluster_index
  ),
  threshold=0.0
)

pred_class = scmap_cluster_results$scmap_cluster_labs

write.csv(
    pred_class,
    file=file.path(args[5], 'scmap_cluster_preds.csv')
)

gt = test_meta[cell_type_col]
n_correct = sum(pred_class == gt)
n_total = length(pred_class)
scmap_cluster_acc = n_correct/n_total

K = irr::kappa2(data.frame(p=pred_class, t=gt))
scmap_cluster_kappa = K$value

print('scmap-cluster Accuracy:')
print(scmap_cluster_acc)

write.csv(
    data.frame(
        scmap_cluster_acc=scmap_cluster_acc, 
        scmap_cluster_kappa=scmap_cluster_kappa
    ), 
    file=file.path(args[5], 'scmap_cluster_acc.csv'), 
    row.names=F
)

########################
# scmap-cell 
########################

set.seed(1)

# build NN graph
train_sce <- scmap::indexCell(train_sce)

scmap_cell_results <- scmap::scmapCell(
  test_sce, 
  list(
    train_data = metadata(train_sce)$scmap_cell_index
  )
)

# get class labels from the object
# the k-NN are stored in `scmap_cell_results$train_data$cells`
# a [k x n_test_cells] matrix where values are the
# integer indices of cells in the training index
getCellType = function(indices){
    cl = names(sort(table(train_meta[indices, "cell_type1"]), decreasing=T)[1])
    return(cl)
}

pred_class = apply(scmap_cell_results$train_data$cells,2, getCellType)

write.csv(
    pred_class,
    file=file.path(args[5], 'scmap_cell_preds.csv')
)

gt = test_meta[cell_type_col]
n_correct = sum(pred_class == gt)
n_total = length(pred_class)
scmap_cell_acc = n_correct/n_total

K = irr::kappa2(data.frame(p=pred_class, t=gt))
scmap_cell_kappa = K$value

print('scmap-cell Accuracy:')
print(scmap_cell_acc)

write.csv(
    data.frame(
        scmap_cell_acc=scmap_cell_acc, 
        scmap_cell_kappa=scmap_cell_kappa
    ), 
    file=file.path(args[5], 'scmap_cell_acc.csv'), 
    row.names=F
)

# clear memory
rm(train_data)
rm(train_meta)
rm(test_data)
rm(test_meta)
rm(train_sce)
rm(test_sce)

######################################################
# Train singleCellNet
######################################################

print('Preparing data for singleCellNet')
# Find common genes between train and test data per singleCellNet
common_genes = intersect(
    train_robj$var_names,
    test_robj$var_names
)

# subset to training cells and common genes
expTrain = train_robj$X[,train_idx]
expTrain = expTrain[common_genes,]
stTrain = train_robj$obs[train_idx,]

# subset test data to common genes
expTest = test_robj$X[common_genes,]
stTest = test_robj$obs

print('Training singleCellNet')

# train model
class_info = scn_train(
    stTrain = stTrain, 
    expTrain = expTrain, 
    nTopGenes = 10, 
    nRand = 70, 
    nTrees = 1000, 
    nTopGenePairs = 25,
    dLevel = cell_type_col, 
    colName_samp = "cell_barcode"
)

print('Predicting with singleCellNet')

# predict on test data
pred = scn_predict(
    cnProc=class_info[['cnProc']], 
    expDat=expTest, 
    nrand = 0
)

# save prediction probabilities
write.csv(
    pred,
    file=file.path(args[5], 'singlecellnet_probs.csv')
)

# get the maximum possible column value
pred_class_idx = argmax(pred, rows=FALSE)
classes_by_idx = rownames(pred)
pred_class = classes_by_idx[pred_class_idx]

write.csv(
    pred_class,
    file=file.path(args[5], 'singlecellnet_preds.csv')
)

gt = stTest[cell_type_col]
n_correct = sum(pred_class == gt)
n_total = length(pred_class)
scn_acc = n_correct/n_total

K = irr::kappa2(data.frame(p=pred_class, t=gt))
scn_kappa = K$value

print('singleCellNet Accuracy:')
print(scn_acc)

write.csv(
    data.frame(scn_acc=scn_acc, scn_kappa=scn_kappa), 
    file=file.path(args[5], 'singlecellnet_acc.csv'), 
    row.names=F
)

# free up memory
rm(expTrain)
rm(expTest)
rm(stTrain)
rm(stTest)
rm(pred)
rm(class_info)
gc()

######################################################
# Setup CHETAH and scPred data
######################################################

# subset data to appropriate cells and genes
common_genes = intersect(
    train_robj$var_names,
    test_robj$var_names
)

train_data = train_robj$X[,train_idx]
train_data = train_data[common_genes,]
train_meta = train_robj$obs[train_idx,]
test_data = test_robj$X
test_data = test_data[common_genes,]
test_meta = test_robj$obs

######################################################
# Train CHETAH
######################################################

library(CHETAH)

# Make SingleCellExperiments
# reusing train_data and test_data with common genes
# from above
train_meta['celltypes'] = train_meta[cell_type_col]
train_sce = SingleCellExperiment(
    assays = list(counts = train_data),
    colData = DataFrame(celltypes = train_meta$celltypes),
)

test_sce = SingleCellExperiment(
    assays = list(counts = test_data)
)

# train and predict with the CHETAH model
chetah_out = CHETAHclassifier(
    input = test_sce, 
    ref_cells = train_sce,
    thresh = 0
)

## Extract celltypes:
pred = chetah_out$celltype_CHETAH
pred_class = as.vector(chetah_out$celltype_CHETAH)

gt = test_meta[names(pred), cell_type_col]

n_correct = sum(pred_class == gt)
n_total = length(pred_class)
chetah_acc = n_correct/n_total

K = irr::kappa2(data.frame(p=pred_class, t=gt))
chetah_kappa = K$value

write.csv(
    pred_class,
    file=file.path(args[5], 'chetah_preds.csv')
)

write.csv(
    data.frame(chetah_acc=chetah_acc, chetah_kappa=chetah_kappa), 
    file=file.path(args[5], 'chetah_acc.csv'), 
    row.names=F
)

######################################################
# Train scPred
######################################################

library(scPred)

# perform eigendecomposition and feature selection preprocessing
scp = scPred::eigenDecompose(as.matrix(train_data))
scPred::metadata(scp) = train_meta

scp = scPred::getFeatureSpace(scp, pVar = cell_type_col)

# train model
scp = scPred::trainModel(scp, seed = 77)

# predict with trained model
scp = scPred::scPredict(
    scp, 
    newData = as.matrix(test_data),
    threshold = 0.0
)

pred = getPredictions(scp)

raw_pred_class = pred$predClass
# fix weird formatting from scpred
pred_class = character(length(raw_pred_class))
for (i in 1:length(raw_pred_class)){
    pred_class[i] = stringr::str_replace_all(raw_pred_class[i], '[.]', ' ')
}

pred$pred_class = pred_class

write.csv(
    pred,
    file=file.path(args[5], 'scpred_probs.csv')
)

write.csv(
    pred_class,
    file=file.path(args[5], 'scpred_preds.csv')
)

gt = test_meta[rownames(pred), cell_type_col]

n_correct = sum(pred_class == gt)
n_total = length(pred_class)
scpred_acc = n_correct/n_total

K = irr::kappa2(data.frame(p=pred_class, t=gt))
scpred_kappa = K$value

write.csv(
    data.frame(scpred_acc=scpred_acc, scpred_kappa=scpred_kappa), 
    file=file.path(args[5], 'scpred_acc.csv'), 
    row.names=F
)

print('scPred Accuracy:')
print(scpred_acc)

# free memory
rm(scp)
rm(pred_class)
rm(pred)
gc()


print('BENCHMARKING FINISHED.')