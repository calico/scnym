#!/bin/bash
#################################################################
# This script provides an example of an scNym training run.
# Here, we download data from the Tabula Muris and train an
# scNym model to predict cell types from these data.
#################################################################

#################################################################
# SET PARAMETERS
#################################################################

# Set the directory for downloaded files
DOWNLOAD_DIR=tmp
mkdir ${DOWNLOAD_DIR}

#################################################################
# DOWNLOAD AND PREPARE TABULA MURIS DATA
#################################################################

# Download Tabula Muris Senis lung data
echo "DOWNLOADING DATA"
cd ${DOWNLOAD_DIR}
wget https://ndownloader.figshare.com/files/15467792
mv 15467792 lung.h5ad

# export metadata as a separate CSV for scNym
echo "EXPORTING METADATA AND GENE NAMES"
echo "NORMALIZING COUNTS TO LOG(CPM + 1)"
python -c "import anndata; import numpy as np; import scanpy.api as sc; a=anndata.read_h5ad('lung.h5ad'); a.obs.to_csv('metadata.csv'); np.savetxt('gene_names.csv', a.var_names, fmt='%s'); sc.pp.normalize_per_cell(a, counts_per_cell_after=1e6); sc.pp.log1p(a); a.write_h5ad('lung.h5ad')"

# return to the original directory
cd -

#################################################################
# TRAIN SCNYM
#################################################################

echo "TRAINING SCNYM"
scnym train_tissue_independent \
    -c configs/default_config.txt \
    --input_counts ${DOWNLOAD_DIR}/lung.h5ad \
    --input_gene_names ${DOWNLOAD_DIR}/gene_names.csv \
    --training_metadata ${DOWNLOAD_DIR}/metadata.csv \
    --lower_group cell_ontology_class \
    --upper_group tissue \
    --n_epochs 50 \
    --out_path ./tmp 

echo "DONE"
