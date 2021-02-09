# Processed Data

We provide preprocessed versions of public data sets used to evaluate scNym below.
All datasets are formatted following the [`AnnData` conventions](https://anndata.readthedocs.io/en/stable/index.html).

## Human

1. [**hPBMS Stimulated/Control**](https://storage.googleapis.com/calico-website-mca-storage/kang_2017_stim_pbmc.h5ad) -- We used the authors' annotations in the `.obs["cell"]` attribute. We split experimental conditions based on the authors annotations in the `.obs["stim"]` attribute. We obtained source data from GEO at accession GSE96583.

## Mouse

1. [**Tabula Muris 10x**](https://storage.googleapis.com/calico-website-scnym-storage/reference_data/mouse_tabula_muris_10x_log1p_cpm.h5ad) -- We used the authors' annotations in the `.obs["cell_ontology_class"]` attribute. We obtained [source data from the authors' website.](https://github.com/czbiohub/tabula-muris/blob/master/tabula-muris-on-aws.md)
2. [**Tabula Muris Smart-seq2**](https://storage.googleapis.com/calico-website-scnym-storage/reference_data/mouse_tabula_muris_facs_log1p_cpm.h5ad) -- We used the authors' annotations in the `.obs["cell_ontology_class"]` attribute. We obtained [source data from the authors' website.](https://github.com/czbiohub/tabula-muris/blob/master/tabula-muris-on-aws.md)
3. [**Mouse Cell Atlas Lung data with manual annotations**](https://storage.googleapis.com/calico-website-scnym-storage/reference_data/mouse_mca_lung_log1p_cpm.h5ad) -- We added manual annotations in the `.obs["cell_ontology_class"]` attribute. We obtained [source data from the authors' website.](http://bis.zju.edu.cn/MCA/)
4. [**10x Visium Spatial Transcriptomics of Mouse Brain Section 0**](https://storage.googleapis.com/calico-website-scnym-storage/reference_data/mouse_spatial_brain_section0.h5ad) -- We added manual annotations in the `.obs["cell_ontology_class"]` attribute. We obtained [source data from 10x Genomics.](https://www.10xgenomics.com/resources/datasets/)
5. [**10x Visium Spatial Transcriptomics of Mouse Brain Section 1**](https://storage.googleapis.com/calico-website-scnym-storage/reference_data/mouse_spatial_brain_section1.h5ad) -- We added manual annotations in the `.obs["cell_ontology_class"]` attribute. We obtained [source data from 10x Genomics.](https://www.10xgenomics.com/resources/datasets/)
6. [**Cross-Technology comparison in the Mouse Cortex**](https://storage.googleapis.com/calico-website-scnym-storage/reference_data/mouse_cortex_methods_comparison_log1p_cpm.h5ad) -- We used the authors' annotations in the `"CellType"` attribute. We obtained source data from the [Broad Institute Single Cell Portal](https://singlecell.broadinstitute.org/single_cell/study/SCP425/single-cell-comparison-cortex-data).
7. [**Cross-Technology comparison in the Mouse Kidney**](https://storage.googleapis.com/calico-website-scnym-storage/reference_data/mouse_kidney_methods_comparison_log1p_cpm.h5ad) -- We mapped the authors' annotations into the "cell ontology" namespace and stored these annotations in the `"cell_ontology_class"` attribute. We removed cells with ambiguous cell types under the cell ontology (2310 cells total). We identified unique experimental domains using the `"Protocol"` attribute. We obtained data from GEO at [GSE141115](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE141115).

## Rat

1. [**Rat Aging Cell Atlas**](https://storage.googleapis.com/calico-website-scnym-storage/reference_data/rat_aging_cell_atlas_ma_2020.h5ad) -- We translated the authors' annotations into the "cell ontology class" namespace and stored these annotations in the `.obs["cell_ontology_class"]` attribute. We obtained source data from GEO at accession GSE137869.
