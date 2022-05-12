"""
Evaluate ability of reprogramming models to recover a perturbed TF 
when identifying cell identity. 
User is bringing predefined models. 
Evaluating TF overexpression study from Parekh, et al 2018 where 
cells are treated with a single TF overexpression in ESC. 
We would expect a good reprogramming model to recover the perturbed 
TF as the best method for reprogramming between two cell states.

Workflow
--------
1. Fit a model to dataset
2. Query each TF overexpression vs source (starting state) for that model.
3. Rank the recovery of each TF overexpression, store in a dictionary for each model. 
4. Shows rankings per model for recovery of TF overexpression in a table with other 
models for comparison.
5. Calculate performance metrics for each model based on the ranks for TF recovery.
"""
import pandas as pd
import numpy as np
import anndata as ad
from typing import Tuple
import logging
import time

# define a logger object to use for debugging
# we set the logger name to __name__ so it's unique for each module
logger = logging.getLogger(__name__)

import scipy.stats as stats
from scipy.stats import gmean

# Define constants in all capitilzation
# centrally code all the hardcoded constants so all in one place when need to change it

TF_OX_DATASETS = {
    "parekh_hs": {
        "path": "/group/singlecell/human/parekh_2018_hpsc_tf_reprogramming/annotated_log1p_cpm.h5ad",
        "cell_type_col": "tf",
        "source_key": "mCherry",
    }
}


####################################################
# Data loading preliminaries
####################################################


def load_data():
    adata = ad.read_h5ad(TF_OX_DATASETS["parekh_hs"]["path"])
    return adata




####################################################
# Model training, querying, and TF rank extration
####################################################


def rank_perturbed_factors(model, adata, source: str, target_groupby: str) -> dict:
    """
    Rank a perturbed factor's ability to 'recover' a perturbed gene regulatory network
    based on gene expression, relative to a source (control) state.

    Parameters
    ----------
    model : pathfinder method Model API
    adata : adata
        anndata object with perturbed factors annotated in adata.obs
    source : str
        label for cells that are the starting state prior to perturbation
    target_groupby : str
        label for cells that are the perturbed or reprogrammed state
        assumes every target in that column isn't a source

    Returns
    -------
    results : dict
        dictionary with perturbed factors as keys and their ranks as values
        ranks will be ordered based on model's method

    """
    results = {}

    # fit models:
    if model._fit_before_query:
        model.fit(
            adata=adata,
            X=adata.X,  # data
            y=adata.obs[
                target_groupby
            ],  # needs to have matching labels for cell types across data inputs
        )

    # extract unique names to know if source and target perturbations are in here
    # returns an array of unique targets, sorted lexigraphically
    unique_pert = np.unique(adata.obs[target_groupby])

    # warn user if source state not included in perturbed factors
    if source not in unique_pert:
        msg = f"source {source} is not in the set of unique perturbations"
        raise ValueError(msg)

    # turn dictionary of perturbed factors into a list
    unique_pert = unique_pert.tolist()
    # remove source from the perturbations list
    idx = unique_pert.index(source)
    unique_pert.pop(idx)  # can also utilize remove
    # now have unique_pert without source in the list
    # this will now act as the target list

    # need to check that every factor in unique pertubations/target list is present in gene_sets
    # first keep track of factors excluded from rank because don't match gene_sets
    # excluded_factors = [x for x in unique_pert if x not in gene_sets.keys()]
    # create print statement to show excluded factors

    # trim list of unique_pert to match the factors present in the model's gene_sets.keys()
    # use list comprehension, happens inside brackets
    unique_pert = [x for x in unique_pert if x in model.gene_sets.keys()]
    # insure that all genes from unique_pert factors are detected in adata
    # use a restrictive assumption
    unique_pert = [x for x in unique_pert if x in adata.var_names]
    logger.debug(f"unique_pert\n{unique_pert}")
    # model query
    # here t is for overexpressed TF/perturbed factors
    for t in unique_pert:
        logger.info(f"\tranking factor: {t}")
        start = time.time()
        logger.debug(f"target: {t}, source: {source}")
        rank_tfs = model.query(
            adata=adata,
            source=source,  # just a string to look at adata
            target=t,  # is a unique perturbation, one factor at a time
        )
        tf_idx = rank_tfs.index(t) if (t in rank_tfs) else len(model.gene_sets.keys())
        results[t] = (
            tf_idx + 1
        )  # add 1 to rankings so don't start with zero, which would impede interpretations of metrics later
        end = time.time()
        elapsed = end - start
        logger.info(f"\t>>>time elapsed: {elapsed:.3f}")     

    return results


def calc_perturb_factor_ranks(
    models_dict,
    adata,
    source: str,
    target_groupby: str,
) -> pd.DataFrame:
    """
    Creates a dataframe of perturbed factors by model names.
    Shows rankings per model for each perturbed factor's ability to 'recover' a
    perturbed gene regulatory network, or reprogramming strategy, based on gene expression
    relative to a source (control) state.

    Parameters
    ----------
    models_dict : a dictionary of pathfinder methods
    adata : adata
        anndata object with tfs in adata.obs
    source : str
        label for cells that are the starting state prior to perturbation or reprogramming
    target_groupby : str
        label for cells that are the perturbed or reprogrammed state
        assumes every target in that column isn't a source

    Returns
    -------
    model_ranks : pandas dataframe
        perturbed factors as rows X model names as columns
        showing rank for how each model recovers a perturbed factor

    """
    # make a list to store scores per model in
    score_col = []
    # for each model class in the dictionary of models
    for model_name in models_dict.keys():
        logger.info(f"running model: {model_name}")
        # run ranking perturbed factor function for each model
        factor_ranks = rank_perturbed_factors(
            model=models_dict[model_name],
            adata=adata,
            source=source,
            target_groupby=target_groupby,
        )


        # create a dataframe of factor_ranks with model name as index, transpose so factors are row titles
        score_df = pd.DataFrame(factor_ranks, index=[model_name]).T
        # append each score_df into the score columns list
        score_col.append(
            score_df
        )  # will prevent newest score from overwriting previous model's scores

    # concatenate all score_col by stitching together along the second dimensions = columns axis
    model_ranks = pd.concat(score_col, axis=1)

    return model_ranks


####################################################
# Computing performance metrics given ranks
####################################################


def calc_sums(rank_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates sum of all ranks per model.

    Parameters
    ----------
    rank_df : pandas dataframe
        perturbed factors as rows X model names as columns
        showing rank of perturbed factor per each model in a pandas dataframe

    Returns
    -------
    sums_df : pandas dataframe
        models as index and rows and sum of ranks as columns
    """
    sums = []
    # extract model names from rank_df
    for model_name in rank_df.columns:
        # sum of all ranks per model
        model_sum = rank_df[model_name].sum()
        # create an array of model sums
        sum_array = [(model_name, model_sum)]
        # store to data frame
        sum_df = pd.DataFrame(sum_array)
        # append each sum_df into the dataframe
        sums.append(sum_df)

    # concatenate all sums by stitching together along the first dimension (rows)
    sums_df = pd.concat(sums)
    # rename column
    sums_df.columns = ["model", "sum ranks"]
    sums_df = sums_df.set_index("model").copy()

    return sums_df


def calc_sumlogs(rank_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates sum of logs of ranks per model.

    Parameters
    ----------
    rank_df : pandas dataframe
        perturbed factors as rows X model names as columns
        showing rank of perturbed factor per each model in a pandas dataframe

    Returns
    -------
    sumlogs_df : pandas dataframe
        models as index rows and sum of log ranks as columns
    """
    sumlogs = []
    # extract model_names from rank_df
    for model_name in rank_df.columns:
        # log of all ranks per model
        model_log_rank = np.log(rank_df[model_name])
        # sum all log ranks per model
        model_sum_log_ranks = model_log_rank.sum()
        # create an array of model sums
        sum_log_array = [(model_name, model_sum_log_ranks)]
        # store to data frame
        sumlog_df = pd.DataFrame(sum_log_array)
        # append each sum_df into the dataframe
        sumlogs.append(sumlog_df)

    # concatenate all sums by stitching together along the first dimension (rows)
    sumlogs_df = pd.concat(sumlogs)
    # rename column
    sumlogs_df.columns = ["model", "sum of log ranks"]
    sumlogs_df = sumlogs_df.set_index("model").copy()

    return sumlogs_df


def calc_geomeans(rank_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates geometric mean of ranks per model.

    Parameters
    ----------
    rank_df : pandas dataframe
        perturbed factors as rows X model names as columns
        showing rank of perturbed factor per each model in a pandas dataframe

    Returns
    -------
    geomeans_df : pandas dataframe
        models as index rows and geometric mean of ranks as columns
    """
    geomeans = []
    # extract model_name from rank df
    for model_name in rank_df.columns:
        # geometric mean of all ranks per model
        model_geomean_rank = gmean(rank_df[model_name])
        # create an array of model geometric means
        geomean_array = [(model_name, model_geomean_rank)]
        # store to data frame
        geomean_df = pd.DataFrame(geomean_array)
        # append each sum_df into the dataframe
        geomeans.append(geomean_df)

    # concatenate all sums by stitching together along the first dimension (rows)
    geomeans_df = pd.concat(geomeans)
    # rename column
    geomeans_df.columns = ["model", "geomean ranks"]
    geomeans_df = geomeans_df.set_index("model").copy()

    return geomeans_df


def calc_auc(rank_df: pd.DataFrame, rank_cutoff: int) -> pd.DataFrame:
    """
    Calculates area under the curve (AUC) for ranks per model.

    Parameters
    ----------
    rank_df : pandas dataframe
        perturbed factors as rows X model names as columns
        showing rank of perturbed factor per each model in a pandas dataframe
    rank_cutoff : int
        hyperparameter for maximum threshold of ranks to include in AUC calculation

    Returns
    -------
    auc_df : pandas dataframe
        models as index rows and auc of ranks as columns
    """
    curve_heights = {}
    # calc_auc
    # get ranks from table within the cutoff range
    for model_name in rank_df.columns:
        rankings = rank_df[model_name]
        # keep_rankings = all_rankings[all_rankings < rank_cutoff]

        # make list to store rank 'heights'
        y_list = []
        # for each rank in rankings, calculate how many TFs hold that rank
        # y represents the "height" for total possible TFs at a rank
        for r in range(
            1, rank_cutoff + 1
        ):  # here is where might want to use rank cutoff because only care up until a point
            y = np.sum(
                rankings <= r
            )  # do I want to do this from all_rankings or just from those kept under cutoff? should be same, right?
            y_list.append(y)  # keeps track of rank steps

        # don't ovrwrite y_list from last model
        curve_heights[
            model_name
        ] = y_list  # if don't give unique keys you will overwrite models with key collision

    # max score possible is number TFs X rank cutoff (all TFs score 1)
    max_auc = rank_df.shape[0] * rank_cutoff
    # don't actually store the sum values of each of the model curves

    # compute fraction of model auc normalized to max_auc
    auc_df = pd.DataFrame(
        index=list(rank_df.columns), columns=["auc ranks"]
    )  # need a string in an iterable form, pandas needs a container of names
    # loop through model_names in dictionary or dataframe and re-set NaN values to aucs
    for model_name in curve_heights.keys():
        auc_df.loc[model_name, "auc ranks"] = sum(curve_heights[model_name]) / max_auc

    return auc_df


def calc_metrics(
    rank_df: pd.DataFrame,
    rank_cutoff: int,
) -> pd.DataFrame:
    """
    Calculate performance metrics for pathfinder models from a dataframe of ranked perturbed factors and the models which computed them.

    Parameters
    ----------
    rank_df : pandas dataframe
        perturbed factors as rows X model names as columns
        showing rank of perturbed factor per each model in a pandas dataframe
    rank_cutoff : int
        hyperparameter for maximum threshold of ranks to include in AUC calculation

    Returns
    -------
    metrics_df : pandas dataframe
        models as rows and sum of ranks, log of sum of ranks, area under the curve of ranks, geometric mean as columns
    """
    metrics_df = []

    # calculate sum of ranks per model
    sums_df = calc_sums(rank_df=rank_df)

    # calculate sum of log ranks per model
    sumlogs_df = calc_sumlogs(rank_df=rank_df)

    # calculate geometric mean of ranks per model
    geomeans_df = calc_geomeans(rank_df=rank_df)

    auc_df = calc_auc(rank_df=rank_df, rank_cutoff=rank_cutoff)

    metrics = [
        sums_df,
        sumlogs_df,
        geomeans_df,
        auc_df,
    ]

    metrics_df = pd.concat(metrics, axis=1)

    return metrics_df


####################################################
# Optional plotting utilities
####################################################


####################################################
# User-facing API
####################################################


def calc_parekh_tf_ox_bench(
    models_dict: dict,
    rank_cutoff: int = 96,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculating performance metrics for how well pathfinder models recover TF overexpression
    strategies for reprogramming.
    Rank recovery of each TF for each model.

    Parameters
    ----------
    models_dict : dictionary of pathfinder reprogramming methods
    rank_cutoff : int
        hyperparameter for maximum threshold of ranks to include in AUC calculation
        default is 96 for a 96 well platee

    Returns
    -------
    perf_metrics: pandas dataframe
        models as rows and sum of ranks, log of sum of ranks, area under the curve of ranks, geometric mean as columns
    model_ranks : pandas dataframe
        each TF as row, model as columns
    """
    # Load data from some path.
    adata = load_data()
    
    # Change the cell type column of supplied models to reflect the data we're fitting to
    for model_name in models_dict.keys():
        models_dict[model_name].cell_type_col = (
            TF_OX_DATASETS["parekh_hs"]["cell_type_col"]
        )

    # Train and query our model, rank TFs for ability to recover
    # celltypes.
    model_ranks = calc_perturb_factor_ranks(
        models_dict=models_dict,
        adata=adata,
        target_groupby=TF_OX_DATASETS["parekh_hs"]["cell_type_col"],
        source=TF_OX_DATASETS["parekh_hs"]["source_key"],
    )
    # Calculate a performance metrics table across all models.
    perf_metrics = calc_metrics(
        rank_df=model_ranks,
        rank_cutoff=rank_cutoff,
    )
    # Give user back metrics table and table of ranks for each model.

    return perf_metrics, model_ranks
