import anndata
import scanpy as sc
import numpy as np
import pandas as pd
import typing


def name_gradients_from_gene_sets(
    gradients,
    gene_sets,
) -> pd.DataFrame:
    n_cols = gradients.shape[1]
    columns = sorted(list(gene_sets.keys())) + (n_cols - len(gene_sets.keys()))*["other"]
    gradients.columns = columns
    return gradients


def ad_groupby(
    adata: anndata.AnnData,
    groupby: str,
    npop: typing.Callable=np.mean,
    use_rep: str='X',
) -> pd.DataFrame:
    '''Generate a data frame of grouped expression values.
    
    Parameters
    ----------
    adata : anndata.AnnData
        [Cells, Genes]
    groupby : str
        categorical variable in `adata.obs` to use for grouping
        cells.
    npop : Callable
        numpy operation to perform along the cell axis.
    
    Returns
    -------
    df : pd.DataFrame
        indexed by groups, columns are variables in `adata.var_names`.
        values are the result of npop on the given value for all cells
        in the index group.
    '''
    if groupby not in adata.obs.columns:
        msg = f'{groupby} is not a variable in adata.obs'
        raise ValueError(msg)
    
    if adata.obs[groupby].dtype.name == 'category':
        groups = adata.obs[groupby].cat.categories
    else:
        groups = np.unique(adata.obs[groupby])
    
    if use_rep == 'X':
        X = adata.X
        var_names = np.array(adata.var_names)
    elif use_rep in adata.obsm.keys():
        X = adata.obsm[use_rep]
        # create dummy variable names
        var_names = [
            str(x) for x in np.arange(X.shape[1])
        ]
    elif use_rep in adata.layers.keys():
        X = adata.layers[use_rep]
        var_names = np.array(adata.var_names)
    else:
        msg = f'`use_rep` {use_rep} is not valid.'
        raise ValueError(msg)
        
    df = pd.DataFrame(
        np.zeros((len(groups), X.shape[1])),
        index=groups,
        columns=var_names,
    )        
    
    for group in groups:
        bidx = np.array(adata.obs[groupby]==group, dtype=np.bool)
        xg = X[bidx, :]
        df.loc[group, :] = np.array(npop(xg, axis=0)).flatten()
    return df


def get_lv_saliency(
    adata: anndata.AnnData,
    gradients: pd.DataFrame,
    groupby: str,
    target_class: str,
    scale_means: str="group",
) -> pd.DataFrame:
    """Get saliency score equivalents by taking the dot product 
    of regulator mRNA expression with int. gradient for the regulatory program.
    
    e.g. saliency = tf_mrna_expression * tf_grn_intgrad
    
    Parameters
    ----------
    adata : anndata.AnnData
        [Cells, Features]
    gradients : pd.DataFrame
        [target_class_cells, LatentVariables]
    groupby : str
        grouping variable in `adata.obs` for the target classes.
    target_class : str
        target class in `groupby` for saliency analysis.
    scale_means : str
        method for scaling gene expression mean values before saliency scoring.
        `"group"` scales [0, 1] across means of each class.
        `"cells"` scales [0, 1] across cells then computes means for each class.
        `None` does not scale and uses raw mean expression values in `adata.X`.
    
    Returns
    -------
    set_saliency : pd.DataFrame
        [programs, (int_grad, mean_expr, saliency)]
    """
    detected_regulators = np.intersect1d(
        gradients.columns,
        adata.var_names,
    )
    if scale_means == "cells":
        adata = sc.pp.scale(
            adata,
            copy=True,
            zero_center=False,
            max_value=1.0,
        )
    
    target_ad_means = ad_groupby(
        adata,
        groupby=groupby,
        npop=np.mean
    )
    
    if scale_means == "group":
        X = np.array(target_ad_means)
        X -= X.min(0)
        X /= X.max(0)
        target_ad_means = pd.DataFrame(
            X,
            columns=target_ad_means.columns,
            index=target_ad_means.index
        )
    
    index = gradients.columns.tolist()
    index = [x for x in index if x!="other"]
    set_saliency = pd.DataFrame(
        index=index,
        columns=["int_grad", "mean_expr", "saliency"],
    )
    set_saliency["mean_expr"] = 0.
    
    int_grad = gradients.loc[:, index].mean(0)
    mean_expr = np.array(
        target_ad_means.loc[target_class, detected_regulators],
    )
    mean_expr = mean_expr.flatten()
    
    set_saliency["int_grad"] = int_grad
    set_saliency.loc[detected_regulators, "mean_expr"] = mean_expr

    set_saliency["saliency"] = (
        set_saliency["int_grad"] * set_saliency["mean_expr"]
    )
    set_saliency = set_saliency.sort_values("saliency", ascending=False)
    set_saliency["rank"] = np.arange(set_saliency.shape[0])+1
    return set_saliency
