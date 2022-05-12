"""Measure the correlation between ranked predictions for combinatorial 
reprogramming strategies and experimental results collected for all possible 
combinations of two factors among a broad pool for converting MEFs to neurons.

The underlying dataset collected a set of bHLH TFs and a set of POU domain TFs
and measured the neuronal reprogramming efficiency of each bHLH+POU combination
in MEFs. We've collected those efficiences in a table.

We will rank each of these combinations using predictive models and compare
the rank correlation between predicted and experimentally observed ranks. 

References
----------
Tsunemoto et. al. 2018 Nature
"""
import numpy as np
import pandas as pd
import os
import logging
from typing import Tuple
import itertools
from scipy import stats
import anndata

from .reprog_bench import REPROG_DATASETS


CWD = os.path.split(os.path.abspath(__file__))[0]
logger = logging.getLogger(__name__)


def load_cell_data() -> anndata.AnnData:
    """Load Ximerakis brain + MEF single cell data.
    
    Notes
    -----
    Adds new cell labels under "cell_type2use", changing all neuronal subtype
    labels to "neuron" and labeling MEFS as "MEF". All other cell types are simply
    labeled "None".
    """
    # load the dataset to use
    adata = anndata.read_h5ad(
        REPROG_DATASETS["ximerakis_brain_plus_mef"]["path"],
    )
    # convert diverse neuron labels to a generic "neuron" class
    cell_types = adata.obs_vector(
        REPROG_DATASETS["ximerakis_brain_plus_mef"]["cell_type_col"]
    )
    neuron_bidx = np.array(["neuron" in x for x in cell_types])
    mef_bidx = np.array(["embryonic fibroblast" in x for x in cell_types])
    adata.obs["cell_type2use"] = "None"
    adata.obs.loc[mef_bidx, "cell_type2use"] = "MEF"
    adata.obs.loc[neuron_bidx, "cell_type2use"] = "neuron"
    return adata    


def load_experimental_data() -> Tuple[pd.DataFrame, list, list]:
    """Load experimentally measured efficiencies of reprogramming from
    MEFs to neurons using one bHLH TF and one POU TF.
    
    Parameters
    ----------
    None.
    
    Returns
    -------
    df : pd.DataFrame
        ["bHLH", "POU", "efficiency", "rank"] each row is the result of
        reprogramming with the indicated bHLH::POU pair. combinations
        are ordered by efficiency, with `1` as the most efficient rank.
    bhlh_genes : list
        list of bHLH TFs used in the experiment.
    pou_genes : list
        list of POU TFs used in the experiment.
    """
    df = pd.read_csv(
        os.path.join(CWD, "assets", "tsunemoto_2018_diverse_neural_codes_combinations_ranked.csv"),
        index_col=0,
    )
    for k in ["bHLH", "POU", "efficiency", "rank"]:
        assert k in df.columns, f"{k} not in columns."

    df = df.sort_values("efficiency", ascending=False)
    df["rank"] = np.arange(1, df.shape[0]+1)
    max_rank = np.sum(df["efficiency"]>0) + 1
    df.loc[df["efficiency"]==0.0, "rank"] = max_rank
        
    bhlh_genes = np.loadtxt(
        os.path.join(CWD, "assets", "tsunemoto_bhlh_tfs.txt"),
        dtype="str",
    ).tolist()
    pou_genes = np.loadtxt(
        os.path.join(CWD, "assets", "tsunemoto_pou_tfs.txt"),
        dtype="str",
    ).tolist()
    return df, bhlh_genes, pou_genes


def rank_combinations_from_singletons(
    grns: list, 
    k: int=2,
) -> list:
    """Rank the effectiveness of combinations based on the simple sum of the ranks
    of their constituent components.

    Parameters
    ----------
    grns : list
        str elements, ranked list of GRNs for a given task, `0` is the best.
    k : int
        combinatorial complexity to score.

    Returns
    -------
    combination_preds : list
        Tuple[str] elements, ranked list of GRN combinations for a given task.
    
    Notes
    -----
    The logic of this approach is fundamentally flawed. Every combination of factors
    inherently has a less-likely rank than a single highly ranked factor. This statistic
    is only useful if (1) the number of factors is defined *a priori* and (2) we're 
    forced to convert a singleton GRN ranking model into a combination ranking model.
    """
    combinations = list(itertools.combinations(grns, r=2))
    def score_combo(c):
        return sum([grns.index(x) if x in grns else len(grns) for x in c])

    scores = list(map(score_combo, combinations))
    # rank combinations
    sidx = np.argsort(scores)
    combination_preds = np.array(combinations)[sidx].tolist()
    return combination_preds


def rank_bhlh_pou_from_singletons(
    grns: list,
    bhlh_genes: list,
    pou_genes: list,
) -> pd.DataFrame:
    """Rank all combinations of bHLH TFs and POU TFs based on the singleton rankings
    provided in `grns`
    
    Parameters
    ----------
    grns : list
        str elements, ranked list of GRNs for a given task, `0` is the best.    
    bhlh_genes : list
        list of bHLH TFs used in the experiment.
    pou_genes : list
        list of POU TFs used in the experiment.   
        
    Returns
    -------
    comb_df : pd.DataFrame
        ranked combinations, "model_rank" stores the ranked combination.
        cols ["bHLH", "POU", "bHLH_rank", "POU_rank", "sum_single_ranks", "model_rank"].
    """
    bhlh_pou = list(itertools.product(bhlh_genes, pou_genes))

    comb_df = pd.DataFrame(bhlh_pou, columns=["bHLH", "POU"])
    comb_df["bHLH_rank"] = comb_df["bHLH"].apply(lambda x: grns.index(x) if x in grns else len(grns))
    comb_df["POU_rank"] = comb_df["POU"].apply(lambda x: grns.index(x) if x in grns else len(grns))
    comb_df["sum_single_ranks"] = comb_df.loc[:, ["bHLH_rank", "POU_rank"]].sum(axis=1)

    comb_df = comb_df.sort_values("sum_single_ranks", ascending=True)
    comb_df["model_rank"] = np.arange(1, comb_df.shape[0]+1)
    return comb_df


def get_rank_correlation(
    comb_df: pd.DataFrame,
    exp_scores: pd.DataFrame,
) -> Tuple[pd.DataFrame, list, list]:
    """Compute the rank correlation between predictions for the efficiency of 
    reprogramming combinations and their measured performance.
    
    Parameters
    ----------
    comb_df : pd.DataFrame
        model ranked combinations, "model_rank" stores the ranked combination.
        cols ["bHLH", "POU", "bHLH_rank", "POU_rank", "sum_single_ranks", "model_rank"].
    exp_scores : pd.DataFrame
        columns `("bHLH", "POU", "efficiency", "rank")` capturing the experimentally
        measured performance of various TF combinations for reprogramming.
        each row is a single doublet combination and the associated performance.

    Returns
    -------
    results : pd.DataFrame
        columns `("rho", "p_val")` with index `0`.
    ranks : pd.DataFrame
        comparison of model ranks and experimental ranks with columns
        ["bHLH", "POU", "experiment_rank", "experiment_efficiency", "model_rank"]
    """
    exp_scores = exp_scores.rename(
        columns={"rank": "experiment_rank", "efficiency": "experiment_efficiency"},
    )

    # merge exp_scores and comb_df on the bHLH_POU combination
    exp_scores["bHLH_POU"] = (
        exp_scores["bHLH"].astype(str) + "_" + exp_scores["POU"].astype(str)
    )
    comb_df["bHLH_POU"] = (
        comb_df["bHLH"].astype(str) + "_" + comb_df["POU"].astype(str)
    )
    exp_scores = exp_scores.set_index("bHLH_POU")
    comb_df = comb_df.set_index("bHLH_POU")

    ranks = pd.merge(
        comb_df,
        exp_scores,
        how="left",
    )
    ranks.index = ranks["bHLH"].astype(str) + "_" + ranks["POU"].astype(str)

    all_rho, all_p_val = stats.spearmanr(
        ranks["model_rank"],
        ranks["experiment_rank"],
    )
    nz_rho, nz_p_val = stats.spearmanr(
        ranks.loc[ranks["experiment_efficiency"]>0, "model_rank"],
        ranks.loc[ranks["experiment_efficiency"]>0, "experiment_rank"],
    )

    results = pd.DataFrame(
        {"rho": [all_rho, nz_rho], "p_value": [all_p_val, nz_p_val]}, 
        index=["all", "non_zero"],
    )
    return results, ranks


def get_model_rank_correlation(
    model_api,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Train a model on a brain + MEF single cell dataset, rank reprogramming 
    factors, then compare the rank correlation to experimental data.
    
    Parameters
    ----------
    model_api
        model API inspired by `sklearn` models.
        implements `.query(adata, type_A, type_B) -> pd.DataFrame` to
        rank GRNs for reprogramming across cell types.
        
    Returns
    -------
    ranks : pd.DataFrame
        [bHLH, POU, experiment_rank, model_rank] comparison of experimental
        ranks and predicted ranks from the model.
    results : pd.DataFrame
        [rho, p_value] rank correlation as a one-row dataframe.
    """
    adata = load_cell_data()
    
    # get the experimental data
    exp_scores, bhlh_genes, pou_genes = load_experimental_data()
    # check to see which genes are actually "rankable" for the model
    n_original = (len(bhlh_genes), len(pou_genes),)
    bhlh_genes = [x for x in bhlh_genes if x in model_api.gene_sets.keys()]
    pou_genes = [x for x in pou_genes if x in model_api.gene_sets.keys()]
    if len(bhlh_genes) < n_original[0]:
        d = n_original[0] - len(bhlh_genes)
        logger.warn(f"Removed {d} bHLH TFs not in `model_api.gene_sets.keys()`")
    if len(pou_genes) < n_original[1]:
        d = n_original[1] - len(pou_genes)
        logger.warn(f"Removed {d} POU TFs not in `model_api.gene_sets.keys()`")

    # train the model
    model_api.cell_type_col = "cell_type2use"
    if model_api._fit_before_query:
        model_api.fit(
            X=adata.X, 
            y=adata.obs["cell_type2use"], 
            adata=adata,
        )
    if model_api._setup_fit_before_query:
        # perform pre-fit measures
        model_api._setup_fit(
            adata.X, 
            adata.obs["cell_type2use"], 
            adata=adata,
        )
    
    grns = model_api.query(
        adata=adata,
        source="MEF",
        target="neuron",
    )
    comb_df = rank_bhlh_pou_from_singletons(
        grns=grns,
        bhlh_genes=bhlh_genes,
        pou_genes=pou_genes,
    )
    logger.info(f"comb_df\n{comb_df.head()}")
    
    results, comp_ranks_df = get_rank_correlation(
        comb_df=comb_df,
        exp_scores=exp_scores,
    )
    logger.info(f"comp_ranks\n{comp_ranks_df.head()}")

    return comp_ranks_df, results
