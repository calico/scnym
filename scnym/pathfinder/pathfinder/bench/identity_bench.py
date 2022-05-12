"""Evaluate recovery of cell identity specific GRNs using
a GRN activity scoring model

Input:
    Models should output an AnnData object of shape [Obs, Features]
    where Features are GRN scores, named by the regulator.
    e.g. `Pax5` and `miR-9` may both be features.
    AnnData inputs should contain a `"cell_ontology_class"` column
    in `adata.obs`.
    
Output:
    Returns AUROC scores for a specified set of cell identity : GRN pairs,
    evaluating how well inferred GRN activities serve to classify cell
    identities.
"""

import anndata
import scanpy as sc
import numpy as np
import pandas as pd
import os
import logging

from sklearn.metrics import roc_auc_score


CWD = os.path.split(os.path.abspath(__file__))[0]
logger = logging.getLogger(__name__)

# keys : str dataset names
# values : dict[{"path", "cell_type_col"}] for anndata objects of shape [Cells, Genes]
IDENT_DATASETS = {
    "tabula_muris_senis": {
        "path": "/group/singlecell/mouse/tabula_muris_senis/joint/joint_droplet_log1p_cpm.h5ad",
        "cell_type_col": "cell_ontology_class",
    },
    "tabula_muris_senis_liver": {
        "path": "/group/singlecell/mouse/tabula_muris_senis/log1p_cpm/Liver_droplet_log1p_cpm.h5ad",
        "cell_type_col": "cell_ontology_class",        
    },
    "descartes_human_fetal": {
        "path": "/group/singlecell/human/cao_2021_human_fetal_expression/annotated_log1p_cpm_select_tissues.h5ad",
        "cell_type_col": "cell_ontology_class",
    },
    "ximerakis_brain": {
        "path": "/group/singlecell/mouse/ximerakis_aging_brain/annotated_log1p_cpm.h5ad",
        "cell_type_col": "cell_ontology_class",
    },
    "gene_sets": {
        "mouse": {
            "enrichr_tfs": os.path.join(CWD, "assets", "enrichr_tfs_mm.gmt"),
            "dorothea_tfs": os.path.join(CWD, "assets", "dorothea_mm.gmt"),
            "dorothea_mod_tfs": os.path.join(CWD, "assets", "dorothea_mod_mm.gmt"),
        },
        "human": {
            "enrichr_tfs": os.path.join(CWD, "assets", "enrichr_tfs_hs.gmt"),
            "dorothea_tfs": os.path.join(CWD, "assets", "dorothea_hs.gmt"),
            "dorothea_mod_tfs": os.path.join(CWD, "assets", "dorothea_mod_hs.gmt"),
        },
    },
}

# keys : cell ontology classes
# values: List[str] GRN names
# sources of information are general knowledge
# and Tabula Muris Figure 5
# keratinocyte from Kurita 2018 Nature
IDENT_GRN_PAIRS = {
    "B cell": ["Pax5", "Ikzf1", "Ebf1", "Tcf3"],
    "T cell": ["Tbx21", "Eomes", "T", "Tcf7", "Gata3", "Runx1", "Runx3"],
    "skeletal muscle satellite cell": ["Pax7", "Myod1", "Myf5", "Pax3", "Tcf4"],
    "skeletal muscle stem cell": ["Pax7", "Myod1", "Myf5", "Pax3", "Tcf4"],
    "skeletal muscle satellite stem cell": ["Pax7", "Myod1", "Myf5", "Pax3", "Tcf4"],
    # "mesenchymal stem cell": ["Pparg",],
    "hepatocyte": ["Hnf1a", "Hnf4a", "Creb3l3", "Foxa2", "Foxa1", "Foxa3", "Gata4", "Onecut1"],
    "oligodendrocyte": ["Olig1", "Olig2",],
    "dopaminergic neuron": ["Lmx1a", "Lmx1b", "Ascl1", "Nr4a2",],
    "neuron": ["Lmx1a", "Lmx1b", "Ascl1", "Nr4a2", "Myt1", "Myt1l", "Neurod1", "Neurog1", "Neurog2"],
    "classical monocyte": ["Spi1", "Klf4", "Irf8", "Nr4a1", "Cebpa", "Cebpb"],
    "non-classical monocyte": ["Spi1", "Klf4", "Irf8", "Nr4a1", "Cebpa", "Cebpb"],
    "macrophage": ["Irf5", "Irf1", "Irf8", "Stat1", "Cebpe"],
    "endothelial cell": ["Etv1", "Etv2", "Foxc1"],
    "microglial cell": ["Spi1", "Mafb", "Irf8", "Runx1"],
    "keratinocyte": ["Mafb", "Trp63", "Grhl2", "Tfap2a"],
    "basal keratinocyte": ["Mafb", "Trp63", "Grhl2", "Tfap2a"],
    "mature keratinocyte": ["Mafb", "Trp63", "Grhl2", "Tfap2a"],
    "myocyte": ["Myog", "Myf6", "Myod1"],
    "melanocyte": ["Mitf"],
}


def load_gene_sets(
    adata: anndata.AnnData=None,
    gene_sets_path: str=IDENT_DATASETS["gene_sets"]["mouse"]["enrichr_tfs"],
) -> dict:
    """Load gene sets from a GMT file path and subset to genes
    that are present in the provided `adata` object.
    
    Parameters
    ----------
    adata : anndata.AnnData
        [Cells, Genes]
    gene_sets_path : str
        file path to a `.gmt` formatted gene set file.
        
    Returns
    -------
    gene_sets : dict
        keys - str program names, values - List[str] of gene names.
    """
    gene_sets = {}
    with open(gene_sets_path, "r") as f:
        for line in f:
            values = line.split("\t")
            gene_sets[values[0]] = [x.strip("\n") for x in values[1:]]

    if adata is not None:
        # filter for genes in adata
        for k in gene_sets.keys():
            gene_sets[k] = [x for x in gene_sets[k] if x in adata.var_names]

    return gene_sets


def load_dataset(
    dataset: str,
    dataset_map: dict=IDENT_DATASETS,
) -> anndata.AnnData:
    """Load a chosen benchmarking dataset and split into train
    and test sets.
    
    Parameters
    ----------
    dataset : str
        name of a benchmarking dataset.
    dataset_map : dict
        keys are dataset names, values are paths to `h5ad` objects.
        
    Returns
    -------
    adata : anndata.AnnData
        train and test splits are recorded in `.obs["data_split"]`.
        species is recorded in `.obs["species"]`.
    """
    np.random.seed(42)
    
    if dataset not in dataset_map.keys():
        msg = f"{dataset} not in `dataset_map`"
        raise ValueError(msg)

    adata = anndata.read_h5ad(
        dataset_map[dataset]["path"],
    )
    
    test_idx = np.random.choice(
        adata.shape[0],
        size=int(0.1*adata.shape[0]),
        replace=False,
    )

    adata.obs["data_split"] = "train"
    adata.obs.loc[adata.obs_names[test_idx], "data_split"] = "test"
    
    # infer species
    if ("Gapdh" in adata.var_names) or ("Rpl7" in adata.var_names):
        species = "mouse"
    elif ("GAPDH" in adata.var_names) or ("MALAT1" in adata.var_names):
        species = "human"
    else:
        msg = "species could not be inferred from gene names."
        raise ValueError(msg)

    adata.obs["species"] = species
    adata.obs["cell_type"] = (
        adata.obs[dataset_map[dataset]["cell_type_col"]]
    )

    return adata
    

def score_identity_specific_grns(
    adata: anndata.AnnData,
    ident_grn_pairs: dict,
    cell_type_col: str="cell_ontology_class",
) -> pd.DataFrame:
    """
    Parameters
    ----------
    adata : anndata.AnnData
        [Obs, Features] where `.var_names` are GRN names.
        Features are GRN activity scores (higher is higher activity).
    ident_grn_pairs : dict
        cell ontology class keys and List[str] values of specific GRNs.
        these pairs will be used to assess the specificity of each GRN
        to indicated identities.
    
    Returns
    -------
    auc_scores : pd.DataFrame
        [pairs, (identity, grn, auc)] AUROC scores for each cell identity
        GRN pair provided.
    """
    # filter `ident_grn_pairs` for unexpressed genes
    grns_in_pairs = np.unique(sum(list(ident_grn_pairs.values()), []))
    shared_grns = np.intersect1d(grns_in_pairs, adata.var_names)
    missing_grns = np.setdiff1d(grns_in_pairs, adata.var_names)
    
    ident_grn_pairs = {
        k: [x for x in ident_grn_pairs[k] if x in shared_grns] for k in ident_grn_pairs.keys()
    }
    if len(missing_grns) > 0:
        n_removed = len(missing_grns)
        msg = f"Removed {n_removed} GRNs not found in data.\n"
        msg += f"\t{missing_grns}"
        logger.warning(msg)
        
    # remove any identities not found in the dataset
    ident_in_grns = np.unique(list(ident_grn_pairs.keys()))
    shared_idents = np.intersect1d(ident_in_grns, np.unique(adata.obs[cell_type_col]))
    missing_idents = np.setdiff1d(ident_in_grns, np.unique(adata.obs[cell_type_col]))
    
    if len(missing_idents) > 0:
        n_removed = len(missing_idents)
        msg = f"Removed {n_removed} identities not found in data.\n"
        msg += f"\t{missing_idents}"
        logger.warning(msg)
    
    ident_grn_pairs = {
        k: ident_grn_pairs[k] for k in shared_idents
    }

    # `roc_auc_score(y_true, y_score,)` where `y_true` is binary.
    auc_scores = pd.DataFrame(
        index=np.arange(sum([len(v) for v in ident_grn_pairs.values()])),
        columns=["identity", "grn", "auc"],
    )
    i = 0
    for identity in ident_grn_pairs.keys():
        y_true = adata.obs[cell_type_col] == identity
        if np.sum(y_true) == 0:
            msg = f"no observations of class: {identity}. skipping."
            logger.warning(msg)
        
        for grn in ident_grn_pairs.get(identity):
            y_score = adata.obs_vector(grn)
            auc = roc_auc_score(
                y_true,
                y_score,
            )
            auc_scores.iloc[i] = (identity, grn, auc)
            i += 1

    return auc_scores


# TODO: Implement a comparison of identity specific GRNs that performs specific queries,
# rather than using a `.transform` method. We can use the GRN ranks as a proxy for their
# activity scores.


def run_bench(
    model_api,
    dataset_map: dict=IDENT_DATASETS,
    ident_grn_pairs: dict=IDENT_GRN_PAIRS,
) -> pd.DataFrame:
    """Run the identity GRN scoring benchmark tasks given a `model_api`
    that supports `sklearn` style methods.
    
    Parameters
    ----------
    model_api
        supports `.fit(X, y)`, `.predict(X)`, and `.trasform(X)` 
        following the `sklearn` API.
        `.transform(X)` maps `X -> Z` GRN activity scores.
        `.predict(X)` maps `X -> Y` cell type labels.
        `model_api.gene_set_names` contains gene set names for `Z`.
    dataset_map : dict
        keys are dataset names, values are paths to `h5ad` objects.
    ident_grn_pairs : dict
        cell ontology class keys and List[str] values of specific GRNs.
        these pairs will be used to assess the specificity of each GRN
        to indicated identities.        
        
    Returns
    -------
    auc_scores : pd.DataFrame
        [pairs, (identity, grn, auc, dataset)] AUROC scores for each 
        cell identity GRN pair provided in each dataset.
    """
    auc_score_dfs = []
    for dataset_name in dataset_map.keys():
        adata = load_dataset(dataset=dataset_name, dataset_map=dataset_map)
        # we set `cell_type` to the relevant cell type column
        # based on dataset_map information
        logger.info("Fitting model...")
        model_api.fit(
            X=adata.X, 
            y=adata.obs["cell_type"],
            adata=adata,
        )
        logger.info("Transforming data...")
        # [cells, regulons]
        Z_tfs = model_api.transform(adata.X)
        if Z_tfs is None:
            # our no-op defaults return `None`, this means we probably
            # passed a model without embedding support
            msg = "`.tranform` returned `None`.\n"
            msg += f"does {type(model_api)} support `.transform`?"
            raise TypeError(msg)
        
        pred_ad = anndata.AnnData(
            X=Z_tfs,
            obs=adata.obs.copy(),
        )
        pred_ad.var_names = model_api.gene_set_names
        
        auc_scores = score_identity_specific_grns(
            adata=pred_ad,
            ident_grn_pairs=ident_grn_pairs,
        )
        auc_scores["dataset"] = dataset_name
        auc_score_dfs.append(auc_scores,)
        
    auc_score_df = pd.concat(auc_score_dfs, axis=0)
    return auc_score_df
