import anndata
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class ModelAPI(object):
    """
    Abstract model API wrapper for predictive reprogramming experiments.
    
    Attributes
    ----------
    _fit_before_query : bool
        fit the model with `.fit` before calling query.
        default = False.
    _setup_fit_before_query : bool
        model requires calling `._setup_fit()` before running `.query()`.
    _needs_unique_log_dir : bool
        model requires a unique `.log_dir` attribute to save some outputs
        for each call of `.fit(...)`.
    cell_type_call : str
        column in `adata.obs` containing relevant cell type annotations.
    ranks_combinations : bool
        model ranks combinations of factors. 
        if `False`, model ranks individual factors.
    """
    
    def __init__(self, *args, **kwargs) -> None:
        """Initialization sets default attributes and accepts
        arbitrary args and kwargs to allow for multiple inheritance
        with `super`.
        """
        self._fit_before_query = False
        self._setup_fit_before_query = False
        self._needs_unique_log_dir = False
        self.cell_type_col = "cell_type"
        self.ranks_combinations = False
        return
    
    def fit(self, X, y, **kwargs) -> None:
        """Train model. Default no-op."""
        return
    
    def predict(self, X, **kwargs) -> None:
        """Predict cell types. Default no-op."""
        return
    
    def transform(self, X, **kwargs) -> None:
        """GRN embeddings. Default no-op."""
        return    
    
    def query(self, adata, source, target) -> None:
        """Reprogramming query. Must be implemented."""
        raise NotImplementedError()
        
    def save_query(self, path: str) -> None:
        """Save intermediary representiations generated 
        during a `query`"""
        return
        
    def _check_source_target(self, adata, source, target) -> None:
        """Validate that `source` and `target` classes are in `adata`"""
        cell_types = np.unique(adata.obs[self.cell_type_col]).tolist() + ["rest"]
        if source not in cell_types:
            msg = f"source: {source} not in cell types."
            raise ValueError(msg)
        if target not in cell_types:
            msg = f"target: {target} not in cell types."
            raise ValueError(msg)
        if source == target:
            msg = f"source and target cannot both be {source}"
            raise ValueError(msg)
        return
    
    def _prune_gene_sets(self, adata: anndata.AnnData) -> None:
        """Prune gene sets to remove genes not present in `adata`"""
        if not hasattr(self, "original_gene_sets"):
            # store the original gene sets
            self.original_gene_sets = self.gene_sets.copy()
        new_gene_sets = {}
        for k in self.gene_sets.keys():
            new_gene_sets[k] = [
                x for x in self.original_gene_sets[k] if x in adata.var_names
            ]
        # filter out any gene sets with no matching genes
        new_gene_sets = {k:v for k, v in new_gene_sets.items() if len(v)>0}
        self.gene_sets = new_gene_sets

    def _filter_tf_target_correlation(
        self, 
        adata: anndata.AnnData,
        corr_min: float=0.,
        corr_max: float=None,
    ) -> None:
        """Filter gene sets based on the Spearman correlation of regulatory molecule 
        mRNAs and their targets to infer activating and repressive relationships.
        """
        from scipy import stats
        from scipy import sparse

        self._corr_gene_sets = {}
        gene_set_mrnas = [x for x in self.gene_sets.keys() if x in adata.var_names]

        for reg in gene_set_mrnas:
            x_reg = np.array(adata.obs_vector(reg)).reshape(-1)
            y_targets = adata[:, self.gene_sets[reg]].X
            y_targets = y_targets.toarray() if sparse.issparse(y_targets) else y_targets

            x_reg = pd.Series(x_reg, index=adata.obs_names)
            y_targets = pd.DataFrame(
                y_targets, index=adata.obs_names, columns=self.gene_sets[reg]
            )
            # runs in parallel on all available CPUs, returns
            # pd.core.series.Series, indices are gene names
            corrs = y_targets.corrwith(x_reg, method="spearman")

            if corr_min is not None:
                bidx = (corrs > corr_min)
            else:
                bidx = np.ones_like(corrs).astype(bool)
            if corr_max is not None:
                bidx = bidx & (corrs < corr_max)

            targets2keep = np.array(corrs.index)
            targets_removed = targets2keep[~bidx].tolist()
            logger.info(f"Set {reg} removed targets:\n\t{targets_removed}")
            targets2keep = targets2keep[bidx].tolist()
            self.gene_sets[reg] = targets2keep
            self._corr_gene_sets[reg] = corrs
            if len(targets2keep)==0:
                # remove the gene set altogether
                del self.gene_sets[reg]
        return


    def _filter_frequent_targets(
        self, adata: anndata.AnnData, max_parents: int=50,
    ) -> None:
        """Filter target genes with many parental regulators
        
        Notes
        -----
        Inspired by filtration of frequent target genes in Knowledge-Primed NNs.
        """
        # [n_gene_sets, n_genes]
        self._gene_set_matrix = np.zeros(
            (len(self.gene_sets.keys()), adata.shape[1],)
        )

        gene_names = adata.var_names.tolist()
        gene_set_names = list(self.gene_sets.keys())
        for i, k in enumerate(gene_set_names):
            genes = set(self.gene_sets[k]) # set speeds up lookup op below
            bidx = np.array(
                [x in genes for x in gene_names],
                dtype=np.bool,
            )
            self._gene_set_matrix[i, :] = bidx
        
        # compute the number of TFs regulating each target
        self._gene_set_matrix = pd.DataFrame(
            self._gene_set_matrix,
            index=gene_set_names,
            columns=gene_names,
        )
        msg = self._gene_set_matrix.sum(0).sort_values(ascending=False).head(40)
        logger.info(f"Top targets:\n{msg}\n")
        n_regs_per_target = np.array(self._gene_set_matrix.sum(0))
        targets = np.array(adata.var_names)
        targets2rm = targets[n_regs_per_target > max_parents].tolist()
        n = len(targets2rm)
        logger.info(f"Removing {n} freq. targets from gene sets:\n\t{targets2rm}")
        targets2rm = set(targets2rm)
        for k in gene_set_names:
            self.gene_sets[k] = [x for x in self.gene_sets[k] if x not in targets2rm]
        return
        


from .baseline import DEReprog, GSEAReprog, AUCellReprog, Fauxgrify
from .pfnd_scbasset import pscBasset
from .pathfinder import Pathfinder, PathFauxgrify
from .pyscenic import PySCENIC
from . import attributionpriors