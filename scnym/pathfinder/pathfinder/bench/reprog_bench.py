"""Evaluate recovery of known reprogramming strategies using
various models.

Models will be trained on a given dataset, then queried to rank
GRNs for their effectiveness in reprogramming.

We will implement models using `model_api` classes supporting the following
functions:

`.train(X, y)` will fit the model to a dataset.
`.predict(X)` will predict cell type labels.
`.transform(X)` will embed cells in a GRN latent space.
`.query(adata, class_A, class_B)` will return a ranked list of GRNs for reprogramming
cell class A to cell class B.


TODO:
 
* How do we handle a mismatch in the granularity of cell type labels?
e.g. many recipes induce "neuron" conversion, but Ximerakis data capture a wide variety of neurons.
"""

import anndata
import scanpy as sc
import numpy as np
import pandas as pd
import json
import logging
import os
import torch

CWD = os.path.split(os.path.abspath(__file__))[0]
PRETRAIN_WEIGHTS_BASENAME = "final_model_weights.pt"
logger = logging.getLogger(__name__)


# keys : str dataset names
# values : dict[{"path", "cell_type_col"}] for anndata objects of shape [Cells, Genes]
REPROG_DATASETS = {
    "tabula_muris_multidom": {
        "path": "/group/singlecell/mouse/tabula_muris/droplet2facs/joint_tm.h5ad",
        "cell_type_col": "cell_ontology_class",
        "domain_groupby": "domain",
        "species": "mouse",
    },
    "tabula_muris_senis": {
        "path": "/group/singlecell/mouse/tabula_muris_senis/joint/joint_droplet_log1p_cpm.h5ad",
        "cell_type_col": "cell_ontology_class",
        "species": "mouse",
    },
    "tabula_muris_senis_multidom": {
        "path": "/group/singlecell/mouse/tabula_muris_senis/joint/joint_both_log1p_cpm.h5ad",
        "cell_type_col": "cell_ontology_class",
        "domain_groupby": "domain",
        "species": "mouse",
    },
    "tabula_muris_senis_liver": {
        "path": "/group/singlecell/mouse/tabula_muris_senis/log1p_cpm/Liver_droplet_log1p_cpm.h5ad",
        "cell_type_col": "cell_ontology_class",
        "species": "mouse",
    },    
    "ximerakis_brain": {
        "path": "/group/singlecell/mouse/ximerakis_aging_brain/annotated_log1p_cpm.h5ad",
        "cell_type_col": "cell_ontology_class",
        "species": "mouse",
    },
    "ximerakis_brain_plus_mef": {
        "path": "/group/singlecell/mouse/ximerakis_aging_brain/annotated_log1p_cpm_plus_mef.h5ad",
        "cell_type_col": "cell_ontology_class",
        "species": "mouse",
    },
    "mcfaline_emt": {
        "path": "/group/singlecell/human/mcfaline_2019_emt_cropseq/annotated_log1p_cpm.h5ad",
        "cell_type_col": "position",
        "species": "human",
        "domain_groupby": "treatment",
    },
    "kang_pbmc_multidom": {
        "path": "/group/singlecell/human/kang_2017_stimulated_pbmc/h5ad/singlets_log1p_cpm.h5ad",
        "cell_type_col": "cell_ontology_class",
        "domain_groupby": "domain",
        "species": "human",
    },
    "denisenko_kidney_multidom": {
        "path": "/group/singlecell/mouse/denisenko_2020_kidney_methods/h5ad/annotated_int_domain_log1p_cpm.h5ad",
        "cell_type_col": "cell_ontology_class",
        "domain_groupby": "domain",
        "species": "mouse",
    },
    "ding_brain_multidom": {
        "path": "/group/singlecell/mouse/ding_2019_broad_methods_comparison_mouse_cortex/h5ad/all_int_domain_log1p_cpm_and_raw.h5ad",
        "cell_type_col": "cell_ontology_class",
        "domain_groupby": "domain",
        "species": "mouse",
    },
}


def load_known_conversions(species: str="mouse") -> pd.DataFrame:
    """Load known cell identity conversions table"""
    if species == "mouse":
        df = pd.read_csv(
            os.path.join(CWD, "assets", "conversions2.tsv"),
            delimiter="\t",
        )
    elif species == "human":
        df = pd.read_csv(
            os.path.join(CWD, "assets", "conversions2_hs.tsv"),
            delimiter="\t",
        )
    else:
        msg = f"species {species} is invalid."
        raise ValueError(msg)
    for k in ["source", "target", "factors", "citation"]:
        assert k in df.columns, f"{k} not in columns."
    return df


def load_dataset(
    dataset: str,
    dataset_map: dict=REPROG_DATASETS,
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


def query_trained_model(model_api, adata, conversions, save_path):
    """Query a trained model to extract the transcription factor ranks for our list of
    known cell-cell conversions."""
    os.makedirs(save_path, exist_ok=True)
    cell_types = np.unique(adata.obs["cell_type"]).tolist() + ["rest"]
    reprog_ranks = []
    for idx in conversions.index:
        source = conversions.loc[idx, "source"]
        target = conversions.loc[idx, "target"]
        grns = conversions.loc[idx, "factors"].split(";")
        grns = [x.strip(" ") for x in grns]
        
        if source not in cell_types:
            msg = f"source: {source} not in cell types, skipping."
            logger.warning(msg)
            continue
        if target not in cell_types:
            msg = f"target: {target} not in cell types, skipping."
            logger.warning(msg)
            continue     

        # query the model
        # `ranked_hypotheses` is list
        ranked_hypotheses = model_api.query(
            adata,
            source,
            target,
        )
        source_r = source.replace(" ", "_")
        target_r = target.replace(" ", "_")
        save_path_use = os.path.join(save_path, f"{source_r}-{target_r}")
        print(f"Making save path: {save_path_use}")
        os.makedirs(save_path_use, exist_ok=True)
        model_api.save_query(path=save_path_use)
        np.savetxt(
            os.path.join(save_path_use, "ranked_factors.txt"),
            np.array(ranked_hypotheses),
            fmt="%s",
        )
        
        known_grn_ranks = [
            ranked_hypotheses.index(x) if x in ranked_hypotheses else adata.shape[1] for x in grns
        ]
        strategy_name = str(idx) + "_" + source + "->" + target
        task_df = pd.DataFrame(
            {
                "strategy": strategy_name,
                "grn": grns,
                "rank": np.array(known_grn_ranks, dtype=np.int32)+1,
            },
            index=np.arange(len(grns)),
        )
        reprog_ranks.append(task_df)
        
    reprog_ranks = pd.concat(reprog_ranks, axis=0)
    return reprog_ranks


def get_reprog_strategy_ranks(
    adata: anndata.AnnData,
    model_api,
    conversions: pd.DataFrame,
    save_path: str=None,
    weights_path: str=None,
) -> pd.DataFrame:
    """Rank gene regulatory networks for reprogramming across a set
    of predefined cell types.
    
    Parameters
    ----------
    adata : anndata.AnnData
        [Cells, Genes] experiment.
        `.obs["cell_type"]` contains cell classes.
    model_api
        model API inspired by `sklearn` models.
        implements `.query(adata, type_A, type_B) -> pd.DataFrame` to
        rank GRNs for reprogramming across cell types.
    conversions : pd.DataFrame
        [n_strategies, (source, target, factors, citation)]
        where `source, target` are cell classes, `factors` are the
        names of GRNs to be activated, and `citation` is a human
        readable column to be ignored.
    save_path : str, optional.
        path for saving task-specific model outputs.
    weights_path : str
        path to load model weights from, if different from `model.log_dir`.
        assumes the directory structure created by this function:
            weights_path / dataset_name / final_model_weights.pt
        
    Returns
    -------
    reprog_ranks : pd.DataFrame
        [n_strategies*n_grns, (strategy, grn, rank)]
        where `strategy` is `{row_num}_{source}-{target}` from each row
        of `conversions`, `grn` is the GRN name, and `rank` is the rank
        assigned to this GRN for this conversion by the model.
    """
    dataset_name = adata.obs.iloc[0]["dataset"]
    if model_api._needs_unique_log_dir:
        # change the `log_dir` attribute to match the dataset
        # this is useful for models that save weights on the filesystem after fitting
        if not hasattr(model_api, "_orig_log_dir"):
            # set `_orig_log_dir` so we don't continue to add subdirectories
            # as we loop across datasets
            model_api._orig_log_dir = model_api.log_dir
        os.makedirs(model_api._orig_log_dir, exist_ok=True)
        model_api.log_dir = os.path.join(model_api._orig_log_dir, dataset_name)
        logger.warning(f"Set a unique log_dir for {dataset_name}\n{model_api.log_dir}")       
    if hasattr(model_api, "_load_unique_weights"):
        # TODO: Make this check more robust, decide if we want this attribute as part
        # of the base ModelAPI
        logger.warning(f"Loading pre-trained weights for {dataset_name}")
        weights_path = model_api.log_dir if weights_path is None else weights_path
        # load weights assuming the user saved them using the directory struture above
        dataset_weights_path = os.path.join(
            weights_path, dataset_name, PRETRAIN_WEIGHTS_BASENAME,
        )
        model_api._setup_fit(adata.X, adata.obs["cell_type"], adata=adata)
        model_device = list(model_api.model.parameters())[0].device
        model_api.model.load_state_dict(
            torch.load(dataset_weights_path, map_location=model_device)
        )
        model_api.trained = True

    if model_api._fit_before_query:
        # fit the model
        print("Fitting model before query")
        model_api.fit(adata.X, adata.obs["cell_type"], adata=adata)
    else:
        print("Model doesn't require _fit_before_query. Skipping.")

    if model_api._setup_fit_before_query:
        # perform pre-fit measures
        model_api._setup_fit(adata.X, adata.obs["cell_type"], adata=adata)

    reprog_ranks = query_trained_model(
        model_api=model_api,
        adata=adata,
        conversions=conversions,
        save_path=save_path,
    )
    return reprog_ranks


def get_auc_from_ranks(ranks, max_rank: int=200) -> float:
    """Compute the area under the CDF curve for ranked reprogramming factors
    
    Parameters
    ----------
    ranks : np.ndarray
        ranks for a set of selected factors. rank 1 is the best.
    max_rank : int
        maximum rank to consider.
    
    Returns
    -------
    auc : float
        area under the curve for recall of factors.
        truncated and normalized based on `max_rank`.
    """
    # y coordinate of a "rank by recalled factors" plot
    y = np.zeros(max_rank)
    for r in ranks:
        y[r:] += 1
    # get a normalization constant as the highest possible sum under the curve
    # the first `len(ranks)` positions are an ascending cumsum, the latter positions
    # are the max possible score for the remaining `max_rank` bins
    norm = np.sum(np.arange(1, len(ranks)+1)) + (max_rank-len(ranks))*len(ranks)
    return y.sum()/norm


def ranks2aucs(rank_df, max_rank: int=200) -> pd.DataFrame:
    """Compute AUC rankings for each strategy across methos in `rank_df`"""
    tmp_df = rank_df.copy()
    if "method" not in tmp_df.columns:
        tmp_df["method"] = "dummy"
    
    auc_df = pd.DataFrame(
        columns=tmp_df["method"].unique(),
        index=tmp_df["strategy"].unique()
    )
    for method in tmp_df["method"].unique():
        method_df = tmp_df.loc[tmp_df["method"]==method]
        for strategy in method_df["strategy"].unique():
            ranks = method_df.loc[method_df["strategy"]==strategy, "rank"]
            auc = get_auc_from_ranks(ranks, max_rank=max_rank)
            auc_df.loc[strategy, method] = auc
    return auc_df


def run_bench(
    model_api,
    dataset_map: dict,
    conversions: pd.DataFrame=None,
    weights_path: str=None,
) -> pd.DataFrame:
    """Run reprogramming benchmarks for a given model across datasets.
    
    Parameters
    ----------
    model_api
        model API inspired by `sklearn` models.
        implements `.query(adata, type_A, type_B) -> pd.DataFrame` to
        rank GRNs for reprogramming across cell types.
    dataset_map : dict
        keys are dataset names, values are {"path", "cell_type_col"} 
        for `h5ad` objects.
    conversions : pd.DataFrame
        [n_strategies, (source, target, factors, citation)]
        where `source, target` are cell classes, `factors` are the
        names of GRNs to be activated, and `citation` is a human
        readable column to be ignored.
    weights_path : str
        path to load model weights from, if different from `model_api.log_dir`.
        assumes the directory structure created by this function:
            `weights_path / dataset_name / final_model_weights.pt`    
        
    Returns
    -------
    reprog_ranks : pd.DataFrame
        [n_strategies*n_grns, (strategy, grn, rank, dataset)]
        where `strategy` is `{row_num}_{source}-{target}` from each row
        of `conversions`, `grn` is the GRN name, and `rank` is the rank
        assigned to this GRN for this conversion by the model.
    """
    reprog_ranks_dfs = []
    for dataset_name in dataset_map.keys():
        logger.debug(f"Loaded dataset {dataset_name}")
        adata = load_dataset(dataset=dataset_name, dataset_map=dataset_map)
        adata.obs["dataset"] = dataset_name
        reprog_rank = get_reprog_strategy_ranks(
            adata=adata,
            model_api=model_api,
            conversions=conversions,
            weights_path=weights_path,
        )
        reprog_rank["dataset"] = dataset_name
        reprog_ranks_dfs.append(reprog_rank)
        logger.debug(f"{reprog_rank.head(5)}")

    reprog_ranks_df = pd.concat(reprog_ranks_dfs, axis=0)
    return reprog_ranks_df
