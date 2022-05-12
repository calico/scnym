"""Evalute the ability of reprogramming models to recover TFs that control
EMT, as validated experimentally in McFaline-Figeroa 2019

Notes
-----
McFaline-Figueroa et al. performed a CROP-seq screen to knock down important TFs during
both spontaneous EMT and TGFb-induced EMT in MCF10A cell culture models. They used
"pseudospatial" analysis to compare cells collected from the inner core / outer edges
of the culture dish, which serve as surrogate labels for EMT state. Using this axis,
they computed the significance of changes in the pseudospatial distribution for each TF
knockdown and reported the TFs that had significant effects.

We have made machine readable versions of those results and evaluate models on the 
ability to recover hits here.
"""
import os
from pathlib import Path
import logging
from typing import Tuple

import pandas as pd
import numpy as np
import anndata
import torch

from .reprog_bench import REPROG_DATASETS


PRETRAIN_WEIGHTS_BASENAME = "final_model_weights.pt"
CWD = os.path.split(os.path.abspath(__file__))[0]
logger = logging.getLogger(__name__)


def _load_emt_data() -> anndata.AnnData:
    """Load and format anndata for EMT experiments"""
    adata = anndata.read_h5ad(
        REPROG_DATASETS["mcfaline_emt"]["path"],
    )
    adata.var_names_make_unique()
    adata.obs["cell_type"] = (
        adata.obs_vector(REPROG_DATASETS["mcfaline_emt"]["cell_type_col"])
    )
    return adata    


def _load_exp_results() -> pd.DataFrame:
    """Load the results of the McFaline et al experiments
    
    Returns
    -------
    tf_df : pd.DataFrame
        columns [tf, mesen_enriched, strong_lfc, treatment, sig_enrich]
        tf - str TF name
        mesen_enriched - bool enriched in mesenchymal cells
        strong_lfc - bool large enrichment defined by authors
        treatment - {"spontaneous", "tgfb_induced"}
        sig_enrich - bool significant enrichment in pseudospace
    """
    tf_df = pd.read_csv(
        Path(CWD)/Path("assets")/Path("mcfaline_2019_emt_tfs.csv"),
        index_col=0,
    )
    return tf_df


def get_exp_tf_ranks(
    grns: list,
    tf_df: pd.DataFrame,
) -> pd.DataFrame:
    """Extract the ranks of experimentally perturbed TFs from a ranked list.
    
    Parameters
    ----------
    grns : list
        [str,] ranked list of regulatory molecules.
    tf_df : pd.DataFrame
        experimentally measured effects of TFs in EMT models.
        columns [tf, mesen_enriched, strong_lfc, treatment, sig_enrich]

    Returns
    -------
    rank_df : pd.DataFrame
        experimentally measured effects of TFs in EMT models with model ranks.
        columns [tf, mesen_enriched, strong_lfc, treatment, sig_enrich, model_rank]
    """
    rank_df = tf_df.copy()
    rank_df["model_rank"] = -1
    rank_df["model_rank"] = rank_df["tf"].apply(
        lambda x: grns.index(x) if x in grns else -1
    )
    rank_df = rank_df[rank_df["model_rank"]!=-1]
    return rank_df


def extract_ranks(
    rank_df: pd.DataFrame, 
    treatment: str,
    sig_enrich: bool=True,
) -> np.ndarray:
    """Extract model ranks for a given treatment and regulator set.
    
    Parameters
    ----------
    rank_df : pd.DataFrame
        experimentally measured effects of TFs in EMT models with model ranks.
        columns [tf, mesen_enriched, strong_lfc, treatment, sig_enrich, model_rank]
    treatment : str
        {"spontaneous", "tgfb_induced"} EMT inducing treatment to use for rank
        extraction.
    sig_enrich : bool
        if `True`, extract ranks for significant regulators.
        else, extract ranks for insignificant regulators.

    Returns
    -------
    ranks : np.ndarray    
        int ranks of pro-mesenchymal regulatory genes.
    """
    bidx = (
        (rank_df["mesen_enriched"]) 
        & 
        (rank_df["sig_enrich"]==sig_enrich)
        &
        (rank_df["treatment"]==treatment)
    )
    ranks = np.array(rank_df.loc[bidx, "model_rank"], dtype=np.int32)
    return ranks


def get_treatment_condition_ranks(
    model_api,
    adata: anndata.AnnData,
    tf_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Get ranks for a single treatment condition
    
    Parameters
    ----------

    Returns
    -------
    rank_df : pd.DataFrame
        experimentally measured effects of TFs in EMT models with model ranks.
        columns [tf, mesen_enriched, strong_lfc, treatment, sig_enrich, model_rank]
    metric_df : pd.DataFrame
        [1, (sum_pos, sum_log_pos, diff_rank, diff_log_rank)].
    """
    if model_api._fit_before_query:
        model_api.fit(X=adata.X, y=adata.obs["cell_type"], adata=adata)
    
    grns = model_api.query(
        adata=adata,
        source="inner",
        target="outer",
    )

    rank_df = get_exp_tf_ranks(grns=grns, tf_df=tf_df)
    pos_ranks = extract_ranks(
        rank_df=rank_df,
        treatment=tf_df.iloc[0]["treatment"],
        sig_enrich=True,
    )
    neg_ranks = extract_ranks(
        rank_df=rank_df,
        treatment=tf_df.iloc[0]["treatment"],
        sig_enrich=False,
    )

    metric_df = pd.DataFrame(
        {
            "sum_pos": np.sum(pos_ranks),
            "sum_log_pos": np.sum(np.log(pos_ranks)),
            "diff_rank" : np.sum(neg_ranks) - np.sum(pos_ranks),
            "diff_log_rank" : np.sum(np.log(neg_ranks)) - np.sum(np.log(pos_ranks)),
        },
        index=[tf_df.iloc[0]["treatment"]],
    )

    return rank_df, metric_df


def _check_weight_saving_loading(
    model_api,
    weights_path,
    dataset_name,
    adata,
):
    """Check if model weight saving directories need to be adjusted or if model weights
    need to be loaded"""
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
        model_api._setup_fit(
            adata.X, 
            adata.obs["cell_type"], 
            adata=adata,
        )
        model_device = list(model_api.model.parameters())[0].device
        model_api.model.load_state_dict(
            torch.load(dataset_weights_path, map_location=model_device)
        )
        model_api.trained = True
    return model_api


def get_model_emt_ranks(
    model_api,
    weights_path: str=None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Get the ranks of EMT regulators predicted by a `model_api` in both spontaneous
    EMT conditions and TGFb-induced EMT conditions.

    Parameters
    ----------
    model_api
        model API inspired by `sklearn` models.
        implements `.query(adata, type_A, type_B) -> pd.DataFrame` to
        rank GRNs for reprogramming across cell types.
    weights_path : str
        path to pre-trained model weights. expected directory structure:
            weights_path / {spontaneous, tgfb_induced} / final_model_weights.pt

    Returns
    -------
    results : pd.DataFrame
        summary statistics for model performance on this task.
        index {"spontaneous", "tgfb_induced"}
        columns ["sum_ranks", "sum_log_ranks", "diff_ranks", "diff_log_ranks"]
    spon_rank : pd.DataFrame
        model ranks relative to experimental data for the spontaneous condition.
    tgfb_rank : pd.DataFrame
        model ranks relative to experimental data for the TGFb condition.

    Notes
    -----
    Computed results based on:

    1. Sum of ranks of positive regulators (lower is better)
    2. Sum of log(ranks) of positive regulators (lower is better)
    3. Difference sum(ranks[negative_tfs]) - sum(ranks[positive_tfs]) (higher is better)
    4. Difference sum(log(ranks[negative_tfs])) - sum(log(ranks[positive_tfs]))
    """
    # load the dataset to use
    adata = _load_emt_data()
    # get treatment condition indices
    spon_bidx = adata.obs_vector("treatment") != "tgfb"
    tgfb_bidx = ~spon_bidx
    # load experimental TF data
    tf_df = _load_exp_results()
    # get ranks for each condition
    model_api = _check_weight_saving_loading(
        model_api=model_api,
        weights_path=weights_path,
        dataset_name="spontaneous",
        adata=adata[spon_bidx].copy(),
    )

    spon_rank, spon_metric = get_treatment_condition_ranks(
        model_api=model_api,
        adata=adata[spon_bidx].copy(),
        tf_df=tf_df[tf_df["treatment"]=="spontaneous"],
    )

    model_api = _check_weight_saving_loading(
        model_api=model_api,
        weights_path=weights_path,
        dataset_name="tgfb_induced",
        adata=adata[tgfb_bidx].copy(),
    )

    tgfb_rank, tgfb_metric = get_treatment_condition_ranks(
        model_api=model_api,
        adata=adata[tgfb_bidx].copy(),
        tf_df=tf_df[tf_df["treatment"]=="tgfb_induced"],
    )
    results = pd.concat([spon_metric, tgfb_metric], axis=0)
    return results, spon_rank, tgfb_rank
