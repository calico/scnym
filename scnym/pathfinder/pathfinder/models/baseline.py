"""

Notes
-----
# create a function where all shared arguments are fixed, so only the first argument
# varies across workers
pfxn = functools.partial(fxn, arg1=arg1, arg2=arg2)
# launch a "pool" of workers
pool = multiprocessing.Pool()
# launch jobs on the pool of workers, passing the variable argument
results = pool.map(pfxn, variable_args_passed_to_workers)
# get a list of the results
results = list(results)
"""
import pandas as pd
import numpy as np
import scanpy as sc
import functools
import multiprocessing
from pathlib import Path
from typing import Callable
import anndata
import networkx as nx
from .gsea import gsea
from . import ModelAPI

import sys
sys.path += ["/home/jacob/src/scutil"]
import auc_score_ad

import logging
logger = logging.getLogger(__name__)


class DEReprog(ModelAPI):
    
    def __init__(
        self,
        gene_sets: dict,
        cell_type_col: str="cell_type",
    ) -> None:
        """Rank gene regulatory networks for reprogramming based on
        differential expression heuristics
        
        Parameters
        ----------
        gene_sets : dict
            keys are GRN names, values are lists of gene names.
        cell_type_col : str
            column in `adata.obs` that will hold cell type classes.
            
        Returns
        -------
        None.
        """
        super(DEReprog, self).__init__()
        self.gene_sets = gene_sets
        self.gene_set_names = sorted(list(gene_sets.keys()))
        self.cell_type_col = cell_type_col
        return

    def transform(
        self,
        X: np.ndarray,
        adata: anndata.AnnData,
    ) -> np.ndarray:
        """Extract mRNAs for each TF in `gene_sets`. Use `0` filling for genes where
        no mRNA is present."""
        var_names = adata.var_names.tolist()
        gs_with_features = [x for x in self.gene_set_names if x in var_names]
        gs_with_features_idx = [self.gene_set_names.index(x) for x in gs_with_features]

        de_X = np.zeros((adata.shape[0], len(self.gene_set_names)))
        de_X[:, gs_with_features_idx] = adata[:, gs_with_features].X.toarray()
        return de_X
    
    def query(
        self, 
        adata: anndata.AnnData,
        source: str, 
        target: str,
    ) -> list:
        """Find reprogramming GRNs by differential expression
        
        Parameters
        ----------
        adata : anndata.AnnData
            [Cells, Genes]
        source : str
            class name for source class.
        target : str
            class name for target class.
        
        Returns
        -------
        grns : list
            list of GRNs ranked by importance [high -> low].        
        """
        # validate inputs
        self._check_source_target(adata, source, target)
        # perform DE
        sc.tl.rank_genes_groups(
            adata=adata,
            groupby=self.cell_type_col,
            groups=[target],
            reference=source,
            use_raw=False,
            n_genes=adata.shape[1],
            method="t-test",
        )
        dex = sc.get.rank_genes_groups_df(adata, group=target)
        dex = dex.sort_values("scores", ascending=False)
        
        genes = dex["names"].tolist()
        grns = [x for x in genes if x in self.gene_set_names]
        return grns
    
    
class GSEAReprog(ModelAPI):
    
    def __init__(
        self,
        gene_sets: dict,
        cell_type_col: str="cell_type",
        random_sets: int=1000,
    ) -> None:
        """Rank gene regulatory networks for reprogramming based on
        gene set enrichments.
        
        Parameters
        ----------
        gene_sets : dict
            keys are GRN names, values are lists of gene names.
        cell_type_col : str
            column in `adata.obs` that will hold cell type classes.
        random_sets : int
            number of random permutations to perform for the GSEA
            empirical null distribution.
            
        Returns
        -------
        None.
        """
        super(GSEAReprog, self).__init__()
        self.gene_sets = gene_sets
        self.gene_set_names = sorted(list(gene_sets.keys()))
        self.cell_type_col = cell_type_col
        
        self.random_sets = random_sets
        return
    
    def query(
        self, 
        adata: anndata.AnnData,
        source: str, 
        target: str,
    ) -> list:
        """Find reprogramming GRNs by GSEA.
        
        Parameters
        ----------
        adata : anndata.AnnData
            [Cells, Genes]
        source : str
            class name for source class.
        target : str
            class name for target class.
        
        Returns
        -------
        grns : list
            list of GRNs ranked by importance [high -> low].        
        """
        # validate inputs
        self._check_source_target(adata, source, target)
        # # prune gene sets to observed genes
        # self._prune_gene_sets(adata=adata)
        
        # perform GSEA
        # gsea(
        #     D: np.ndarray, 
        #     C: list[{0,1}], 
        #     S_sets: List[List[int]],
        # ) -> order, NES, p_values
        # for gene sets in `S_sets`
        
        # subset to source vs. target cells
        cell_bidx = np.array(
            adata.obs[self.cell_type_col].isin([source, target]).astype(bool),
            dtype=np.bool,
        )
        # get the genes still expressed in the subset
        gene_bidx = np.array(adata[cell_bidx, :].X.sum(0) > 0).flatten().astype(np.bool)
        # extract the matrix of source & target cells, with only genes expressed within
        # this subpopulation present
        D = adata.X[cell_bidx]
        D = D[:, gene_bidx]
        # .X might be a sparse array, might not be. check and densify.
        D = D.toarray() if type(D)!=np.ndarray else D
        # must be [Genes, Samples], transpose for GSEA
        D = D.T
        
        # setup class labels, 0: source, 1: target
        C = np.array(adata.obs[self.cell_type_col] == target, dtype=np.int)
        C = C[cell_bidx]
        # setup `S_sets`, each value is a list of int indices for genes
        # in a given set
        # here, we order `S_sets` to match `self.gene_set_names`.
        var_names = adata.var_names[gene_bidx].tolist()
        S_sets = [
            [var_names.index(x) for x in self.gene_sets[k] if x in var_names] 
            for k in self.gene_set_names
        ]
        
        # order is a list of gene indices, ranked top NES to
        # lowest NES
        order, NES, p_values = gsea(
            D=D,
            C=C,
            S_sets=S_sets,
            random_sets=self.random_sets,
            n_jobs=-1,
        )
        print(order)
        print(NES)
        print(p_values)
        gsea_df = pd.DataFrame({
            "gene": [self.gene_set_names[x] for x in order],
            "nes": NES,
            "p_val": p_values,
        })
        gsea_df = gsea_df.sort_values("nes", ascending=False)
        self.gsea_df = gsea_df
        grns = gsea_df["gene"].tolist()
        return grns

    
class AUCellReprog(ModelAPI):
    
    MAX_DENSE_SIZE = 800e6
    
    def __init__(
        self,
        gene_sets: dict,
        cell_type_col: str="cell_type",
        max_genes: int=2000,
        multiprocess: bool=True,
        densify: bool=False,  
        refit_per_query: bool=False,      
    ) -> None:
        """Rank gene regulatory networks for reprogramming based on
        AUCell scores for gene regulatory networks extracted from
        prior knowledge.
        
        Parameters
        ----------
        gene_sets : dict
            keys are GRN names, values are lists of gene names.
        cell_type_col : str
            column in `adata.obs` that will hold cell type classes.
        max_genes : int
            maximum number of genes to use for AUCell score computation.
        multiprocess : bool
            parallelize AUCell scoring. Can lead to OOM errors for 
            large datasets.      
        densify : bool
            convert matrices to dense before AUCell scoring.
            uses more memory, but runs faster. will only densify
            if the number of elements in the source vs. target
            matrix is `< self.MAX_DENSE_SIZE`.
        refit_per_query : bool
            refit the AUC scoring on a new data subset for each additional query,
            even if a fit already exists.              
        
        Returns
        -------
        None.
        """
        super(AUCellReprog, self).__init__()
        self.gene_sets = gene_sets
        self.gene_set_names = sorted(list(gene_sets.keys()))
        self.cell_type_col = cell_type_col
        
        self.max_genes = max_genes
        self.multiprocess = multiprocess
        self.densify = densify
        self.refit_per_query = refit_per_query
        self.trained = False
        return
    
    def _make_gene_programs_table(self,) -> None:
        """Generate a gene programs table to allow for AUCell scoring"""
        dfs = []
        for gene_set in self.gene_set_names:
            df = pd.DataFrame({
                "Program": gene_set,
                "Gene": self.gene_sets[gene_set],
            })
            dfs.append(df)
        joint_df = pd.concat(dfs, axis=0)
        # remove any programs with only one gene
        # this avoids type-cast errors in `auc_score_ad`
        gene_counts = joint_df.groupby("Program").count()["Gene"]

        keep_programs = gene_counts.index[gene_counts > 2]
        joint_df = joint_df.loc[joint_df["Program"].isin(keep_programs)]
        self.gene_program_df = joint_df.set_index("Program")
        return
    
    def fit(self, X=None, y=None, adata: anndata.AnnData=None) -> None:
        """Estimate AUCell scores for provided gene sets
        
        Parameters
        ----------
        X, y
            dummy variables, ignored for this model.
        adata : anndata.AnnData
            [Cells, Genes] anndata experiment for AUCell scoring.
            
        Returns
        -------
        None.
        
        Notes
        -----
        Most likely called with `.query` to subset to relevant cell types first.
        Fitting to the whole dataset is expensive and likely unnecessary for most
        tasks.
        """
        if self.trained and not self.refit_per_query:
            # no-op
            return
        if adata is None:
            msg = "`adata` must be provided for AUCell. Got `None`."
            raise TypeError(msg)

        # prune gene sets to observed genes
        logger.debug("Pruning gene sets")
        self._prune_gene_sets(adata=adata)
        # make gene programs table
        logger.debug("Making gene programs table")
        self._make_gene_programs_table()

        if np.prod(adata.shape) > self.MAX_DENSE_SIZE:
            # do not densify if it's likely to cause OOM
            densify = False
        else:
            densify = self.densify
        
        # auc_scores is [Cells, GRNs] where GRNs are ordered 
        # lexographically by the result of `np.unique(self.gene_program_df)`.
        logger.debug("Scoring program activity")
        auc_scores = auc_score_ad.score_program_activity(
            adata=adata,
            gene_groups=self.gene_program_df,
            max_genes=self.max_genes,
            multiprocess=self.multiprocess,
            densify=densify,
        )
        
        # store auc scores as part of the model object
        self.auc_ad = anndata.AnnData(
            X=np.array(auc_scores),
            obs=adata.obs.copy(),
        )
        self.auc_ad.var_names = np.unique(self.gene_program_df.index)
        self.trained = True

        msg = "auc_ad cell_type_col\n"
        x = np.unique(self.auc_ad.obs[self.cell_type_col]).tolist()
        msg += f"{x}"
        logger.debug(msg)
        return
    
    def transform(self, X, adata, **kwargs) -> np.ndarray:
        """Transform to AUCell scores"""
        if not hasattr(self, "auc_ad"):
            msg = "Fitting the model to `adata`..."
            print(msg)
            self.fit(X=X, y=adata.obs[self.cell_type_col], adata=adata)
        
        return self.auc_ad.X
    
    def query(
        self, 
        adata: anndata.AnnData,
        source: str, 
        target: str,
    ) -> list:
        """Find reprogramming GRNs based on the top differential
        gene set scores from AUCell
        
        Parameters
        ----------
        adata : anndata.AnnData
            [Cells, Genes]
        source : str
            class name for source class.
        target : str
            class name for target class.
        
        Returns
        -------
        grns : list
            list of GRNs ranked by importance [high -> low].
        """
        # validate arguments
        self._check_source_target(adata, source, target)
        
        # subset to relevant cells
        logger.info(f"{adata.shape[0]} in initial adata for AUCell")
        bidx = adata.obs[self.cell_type_col].isin([source, target])
        subset_adata = adata[bidx, :].copy()
        logger.info(f"{subset_adata.shape[0]} in subset adata for AUCell")
        
        # fit to the subset
        self.fit(X=None, y=None, adata=subset_adata)
        
        sc.tl.rank_genes_groups(
            adata=self.auc_ad,
            groupby=self.cell_type_col,
            groups=[target],
            reference=source,
            use_raw=False,
            n_genes=self.auc_ad.shape[1],
            method="t-test",
        )
        de_auc = sc.get.rank_genes_groups_df(self.auc_ad, group=target)
        de_auc = de_auc.sort_values("scores", ascending=False)
        self.de_auc = de_auc
        grns = de_auc["names"].tolist()
        return grns

  
class Fauxgrify(ModelAPI):
    
    def __init__(
        self,
        gene_sets: dict,
        cell_type_col: str="cell_type",
        network_degree: int=3,
        degenerate_thresh: float=0.98,
        max_rank_considered: int=100,
        max_source_expression: float=3.,
        multiprocess: bool=False,
        net_score_preprocess: Callable=np.abs,
        **kwargs,
    ) -> None:
        """Implement a method based on `Mogrify`.
        
        Parameters
        ----------
        gene_sets : dict
            keys are GRN names, values are lists of gene names.
        cell_type_col : str
            column in `adata.obs` that will hold cell type classes.
        network_degree : int
            max number of degrees of separation between a regulator
            and a target to consider.
            [Default = 3]
        degenerate_thresh : float
            (0, 1] maximum fraction of overlap between the target sets
            of two regulators to consider them degenerate.
            if two TFs are degenerate, only the top ranked TF is retained
            in suggestions.
            `degenerate_thresh=1.0` does not prune degenerate TFs.
        max_rank_considered : int
            TFs with a rank higher than this value will not be considered.
        max_source_expression : float
            maximum expression in the source population for considered TFs.
        net_score_preprocess : Callable
            preprocessing function applied to gene scores before integrating from gene
            scores to network scores. [default = np.abs] per Mogrify. 
            PathFauxgrify default is a ReLU. if `None`, uses a no-op.
            
        Returns
        -------
        None.

        Notes
        -----
        Mogrify heuristics do *not* offer a `.transform` method, as there is no simple
        way to score the GRN activity of individual cells. Rather, Mogrify requires
        differential expression statistics for each sample, which only make sense in the
        context of a cell population, not a single observation.
        """
        super(Fauxgrify, self).__init__(gene_sets=gene_sets, **kwargs)
        
        self.gene_sets = gene_sets
        self.gene_set_names = sorted(list(gene_sets.keys()))
        self.multiprocess = multiprocess
        
        self.cell_type_col = cell_type_col
        self.network_degree = network_degree
        self.degenerate_thresh = degenerate_thresh
        self.max_rank_considered = min(max_rank_considered, len(self.gene_set_names))
        self.max_source_expression = max_source_expression
        self.net_score_preprocess = net_score_preprocess

        # set up the directed graph of regulators and genes
        self._construct_regulatory_graph()
        self.precomp_target_genes = {}
        return
    
    def _construct_regulatory_graph(self,) -> None:
        """Construct a regulatory graph from the provided gene sets"""
        gene_sets = self.gene_sets
        # instantiate an empty graph
        G = nx.DiGraph()
        
        # populate the graph with a node for each gene
        # and each regulator
        targets = sum(list(gene_sets.values()), [])
        regulators = list(gene_sets.keys())
        gene_names = set(targets + regulators)
        G.add_nodes_from(gene_names)
        
        # add edges from regulators to targets
        for reg in gene_sets.keys():
            for target in gene_sets[reg]:
                G.add_edge(reg, target)
        
        # G.successors(node_name) returns targets of `node_name`
        # G.predecessors(node_name) returns regulators of `node_name`
        # both functions return an iterator over the other nodes
        self.reg_graph = G
        return
    
    def _construct_redundancy_matrix(self,) -> None:
        """Construct a matrix with the fraction of genes in each
        pair of gene sets that are shared
        
        Returns
        -------
        None. Sets `.regulon_overlap_matrix` pd.DataFrame with indices
        and columns matching gene set names.
        `regulon_overlap_matrix.loc[i, j]` is the fraction of genes in `i`
        that are shared with the program `j`.
        """
        if hasattr(self, "regulon_overlap_matrix"):
            # no-op
            return
        # M_i,j is the fraction of i's genes that are contained within j
        # NOTE: This is not symmetric.
        logger.info("Constructing regulon target overlap matrix")
        self.regulon_overlap_matrix = pd.DataFrame(
            index=self.gene_set_names,
            columns=self.gene_set_names,
        )
        for i in self.gene_set_names:
            logger.info(f"Creating redundnacy matrix at gene_set_i {i}")
            for j in self.gene_set_names:
                i_g = set(self.gene_sets[i])
                j_g = set(self.gene_sets[j])
                ij_g = i_g.intersection(j_g)
                # number of genes that are shared as a fraction of
                # total genes in the `i` program
                self.regulon_overlap_matrix.loc[i, j] = len(ij_g)/len(i_g)
        return
    
    def _get_target_genes(self, regulator: str, order: int) -> pd.DataFrame:
        """Get target genes with `order` levels of indirect effects
        
        Parameters
        ----------
        regulator : str
            regulator to query targets for.
        order : int
            number of indirect regulatory steps to consider when searching
            for targets.
            e.g. `order=1` only gets direct targets, `order=2` gets one
            layer of indirection.
        
        Returns
        -------
        target_df : pd.DataFrame
            [targets, (parent_regulator, n_parent_children, indirect_level)]
        """
        x = self.precomp_target_genes.get(regulator, None)
        if x is not None:
            # use precomputed target set
            return x
        # initialize a dict with order of regulation 0: List[str: regulator]
        # each successive integer key will be populated by a list of tuples 
        # of the form:
        #      ([gene_names, ...], parent_regulator, n_parent_children)
        target_genes = {0: [([regulator], None, None)]}
        level = 1
        while level <= order:
            # get all the regulators that are targets of the previous level
            # of regulation
            
            # get all the targets from the previous level
            indirect_regs = sum(
                [target_genes[level-1][i][0] for i in range(len(target_genes[level-1]))], []
            )
            indirect_regs2use = np.intersect1d(indirect_regs, self.gene_set_names)
            logger.debug(
                f"indirect regulators at level {level}\n\t{indirect_regs}"
            )
            
            target_genes[level] = []
            # for each indirect regulator at this level, get all of its targets
            # and store them in a growing list
            for indirect_reg in indirect_regs2use:
                targets = list(self.reg_graph.successors(indirect_reg))
                indirect_reg_targets = (
                    targets, indirect_reg, len(targets),
                )
                target_genes[level].append(indirect_reg_targets)
            
            level += 1
        
        # convert our dict tuple data structure to dataframe
        dfs = []
        for level in range(1, order+1):
            
            for regulon in target_genes[level]:
                # each regulon is tuple from above
                # set index to genes
                tmp_df = pd.DataFrame(
                    {
                        "parent_regulator": regulon[1],
                        "n_parent_children": regulon[2],
                        "indirect_level": level,
                    },
                    index=regulon[0],
                )
                dfs.append(tmp_df)
        
        df = pd.concat(dfs, axis=0)
        # remove duplicates and priorize the first occurence which
        # is the most direct regulatory relationship
        # .duplicated() marks duplicates `True` after the first 
        # occurence by default
        df = df[~df.index.duplicated(keep="first")]
        # store for later
        # TODO: consider not storing the shared dictionary in case 
        # it conflicts with shared memory
        # self.precomp_target_genes[regulator] = df
        return df
    
    def _get_deg_score(
        self, 
        adata: anndata.AnnData, 
        source: str, 
        target: str,
    ) -> pd.DataFrame:
        """Compute the Mogrify differentially expressed gene score
        heuristic across source and target classes
        
        Parameters
        ----------
        adata : anndata.AnnData
            [Cells, Genes]
        source : str
            class name for source class.
        target : str
            class name for target class.

        Returns
        -------
        gene_scores : pd.DataFrame
            indexed by gene name, only values are float in `"gene_score"`.
        
        Notes
        -----
        Mogrify heuristic for a gene score `G_x` where `x` is a gene,
        `T_x` is the level in the target population and `S_x` in the
        source, and `p_x` is the p_value.
        
        .. math::
        
            G_x = |\log_2(T_x) - \log_2(S_x)|(-\log_{10}(p_x))
        """
        # perform DE
        sc.tl.rank_genes_groups(
            adata=adata,
            groupby=self.cell_type_col,
            groups=[target],
            reference=source,
            use_raw=False,
            n_genes=adata.shape[1],
            method="t-test",
        )
        dex = sc.get.rank_genes_groups_df(adata, group=target)
        # clip p-values to the minimum observed non-zero value
        clip_min = np.min(dex.loc[dex["pvals"]>0, "pvals"])
        # we drop the absolute values used in the original Mogrify implementation
        # so that we don't need hacky heuristics to eliminate genes that are more
        # highly expressed in the source cell type
        dex["gene_score"] = (
            np.array(dex["logfoldchanges"]) * (-1 * np.log10(np.clip(dex["pvals"], clip_min, 1.)))
        )
        # add mean expr level in source and target
        source_mean = pd.DataFrame(
            {
                "source_mean": np.array(
                    adata[adata.obs[self.cell_type_col]==source].X.mean(0)).flatten()
            },
            index=adata.var_names,
        )
        target_mean = pd.DataFrame(
            {
                "target_mean": np.array(
                    adata[adata.obs[self.cell_type_col]==target].X.mean(0)).flatten()
            },
            index=adata.var_names,
        )        
        dex["source_mean"] = np.array(source_mean.loc[dex["names"], "source_mean"])
        dex["target_mean"] = np.array(target_mean.loc[dex["names"], "target_mean"])
        return dex.set_index("names")

    def _calc_single_net_score(
        self, network_name: str, gene_scores: pd.DataFrame,
    ) -> float:
        """Compute the `net_score` for a single `network_name` in `gene_sets`
        
        Parameters
        ----------
        network_name : str
            single key in `gene_sets`. net score for this `network_name` is computed.
        gene_scores : pd.DataFrame
            indexed by gene name, only values are float in `"gene_score"`.

        Returns
        -------
        net_score : float
            network score for this `network_name`.
        target_genes : pd.DataFrame
            [targets, (parent_regulator, n_parent_children, indirect_level)]
        """
        # get target genes from the regulatory graph
        # pd.DataFrame, indexed by gene name
        target_genes = self._get_target_genes(
            regulator=network_name, 
            order=self.network_degree,
        )
        logger.debug(f"network_name: {network_name}")
        logger.debug(f"target_genes\n{target_genes.head(10)}")
        
        # get gene scores for computation
        # set gene scores to 0. by default if a gene
        # is not detected as a DEG
        target_genes["gene_score"] = 0.
        target_in_dex = np.intersect1d(
            target_genes.index,
            gene_scores.index,
        )
        target_genes.loc[target_in_dex, "gene_score"] = np.array(
            gene_scores.loc[target_in_dex, "gene_score"]
        )
        
        # compute network score
        # use absolute value for network score computations
        no_op = lambda x: x
        preproc = (
            no_op if self.net_score_preprocess is None else self.net_score_preprocess
        )
        net_score = np.sum(
            preproc(target_genes["gene_score"])
            * (1./target_genes["indirect_level"])
            * (1./target_genes["n_parent_children"])
        )
        return net_score, target_genes
    
    def _get_network_score(self, gene_scores: pd.DataFrame) -> pd.DataFrame:
        """Get network scores using Mogrify heuristics
        
        Notes
        -----
        Mogrify computes the network score as the sum of target gene scores
        with a geometric penalty for the regulatory distance of indirect 
        targets and a penalty for indirect targets whose direct parents have
        a large number of target genes. The latter penalty is an attempt to
        reduce the number of promiscuous regulators (e.g. Fos/Jun).
        
        .. math::
        
            N_x = \sum_{r \in V_x} G_r (L_r O_r)^{-1}
            
        where :math:`N_x` is the network score `V_x` is the set of all target
        genes for regulator :math:`x`, :math:`r` is a target gene indicator
        :math:`G_r` is the gene score, :math:`L_r` is the number of regulatory
        steps between a regulator and target starting at 1 for direct targets,
        and :math:`O_r` is the number of target genes of the parent TF for 
        each gene.
        """
        logger.info("Created partial function")
        partial_get_net_score = functools.partial(
            self._calc_single_net_score,
            gene_scores=gene_scores,
        )
        if self.multiprocess:
            logger.info("Created parallel pool")
            pool = multiprocessing.Pool()
            # each result element is (net_score: float, target_genes: pd.DataFrame)
            results = list(pool.map(partial_get_net_score, self.gene_set_names))
            pool.close()
            logger.info("Closed parallel pool")
        else:
            logger.info("Computing network scores in serial")
            results = []
            for gs in self.gene_set_names:
                logger.info(f"\tComputing net score for {gs}")
                r = self._calc_single_net_score(gs, gene_scores=gene_scores)
                results.append(r)
            logger.info("Network scores computed.")

        network_scores = pd.DataFrame(
            index=self.gene_set_names,
            columns=["network_score",],
        )
        for i, network_name in enumerate(self.gene_set_names):
            network_scores.loc[network_name, "network_score"] = results[i][0]
            self.precomp_target_genes[network_name] = results[i][1]
        
        network_scores["gene_score"] = 0.
        
        # add gene scores for regulator molecules
        regs_in_dex = np.intersect1d(
            self.gene_set_names,
            gene_scores.index,
        )
        for k in ("gene_score", "source_mean", "target_mean",):
            network_scores.loc[regs_in_dex, k] = np.array(
                gene_scores.loc[regs_in_dex, k],
                dtype=np.float64,
            )
        return network_scores
    
    def _rank_from_scores(
        self,
        network_scores: pd.DataFrame,
    ) -> pd.DataFrame:
        """Get regulator ranks from network and gene scores.
        
        Notes
        -----
        Mogrify uses a simple heuristic and takes the final
        network rank to be the sum of the network score rank
        and the gene score rank.
        """
        logger.info("Getting network ranks from scores")
        # rank network scores
        network_scores = network_scores.sort_values("network_score", ascending=False)
        network_scores["network_score_rank"] = np.arange(network_scores.shape[0])+1
        # rank gene scores
        network_scores = network_scores.sort_values("gene_score", ascending=False)
        network_scores["gene_score_rank"] = np.arange(network_scores.shape[0])+1
        # mogrify uses the sum of the network and regulator molecule gene rank
        # score as the final rank
        network_scores["sum_of_ranks"] = np.sum(
            network_scores.loc[:, ["network_score_rank", "gene_score_rank"]],
            axis=1,
        )
        # sort with top rank in `.iloc[0]`.
        network_scores = network_scores.sort_values("sum_of_ranks")
        network_scores["rank"] = np.arange(network_scores.shape[0])+1
        return network_scores
    
    def _prune_network_results(
        self,
        network_scores: pd.DataFrame,
    ) -> pd.DataFrame:
        """Prune the suggested TFs based on Mogrify heuristics
        
        Notes
        -----
        Mogrify prunes networks that have a rank above a threshold
        and networks that share many target genes. It also prunes
        regulators expressed above a threshold in the source dataset.
        Their original threshold was "20 TPM", which is roughly equivalent
        to `20 ~= exp(3)-1` in log1p(CPM) units.
        """
        logger.info("Pruning network rank results")
        # remove TFs expressed in source
        bidx = (network_scores["source_mean"] <= self.max_source_expression)
        network_scores = network_scores.loc[bidx].copy()
        # remove low ranking TFs
        bidx = (network_scores["rank"] <= self.max_rank_considered)
        network_scores = network_scores[bidx].copy()
        # rewrite ranks
        network_scores = network_scores.sort_values("rank", ascending=True)
        network_scores["rank"] = np.arange(network_scores.shape[0])+1
        max_rank = int(np.max(network_scores["rank"]))
        # starting from the final TF, consider whether
        # any TFs above it are highly redudant.
        # if so, remove this TF.
        regs2rm = []
        for rank in range(max_rank, 0, -1):
            reg = network_scores.loc[network_scores["rank"]==rank].index[0]
            # find regulons that are highly redundant with this regulon
            reg_bidx = (
                self.regulon_overlap_matrix.loc[reg, :] >= self.degenerate_thresh
            )
            overlapping_regs = self.regulon_overlap_matrix.columns[reg_bidx]
            # check if any of the overlapping regulons appear earlier in the
            # network score rankings
            overlapping_regs2search = np.intersect1d(overlapping_regs, network_scores.index)
            better_than_bidx = (network_scores.loc[overlapping_regs2search, "rank"] < rank)
            if np.sum(better_than_bidx) > 0:
                # at least one regulon is overlapping and better than this one.
                # remove it.
                regs2rm.append(reg)
        logger.info("Removing regulons that have superior degenerate analogs:")
        logger.info(f"\t{regs2rm}")
        
        network_scores = network_scores.drop(regs2rm, axis=0)
        return network_scores
            
    def query(
        self, 
        adata: anndata.AnnData,
        source: str, 
        target: str,
    ) -> list:
        """Find reprogramming GRNs using Mogrify-inspired heuristics.
        
        Parameters
        ----------
        adata : anndata.AnnData
            [Cells, Genes]
        source : str
            class name for source class.
        target : str
            class name for target class.
            
        Returns
        -------
        grns : list
            ranked GRNs, high to low.
        """
        # validate inputs
        self._check_source_target(adata, source, target)
        # get gene scores
        gene_scores = self._get_deg_score(adata, source, target)
        self.gene_scores = gene_scores
        # get network scores
        network_scores = self._get_network_score(gene_scores=gene_scores)
        # rank network scores
        network_scores = self._rank_from_scores(network_scores=network_scores)
        self.network_scores = network_scores.copy()
        # set up a matrix quantifying overlap of regulons
        self._construct_redundancy_matrix()
        # prune redundant and low scoring GRNs
        network_scores = self._prune_network_results(network_scores=network_scores)
        self.network_scores_pruned = network_scores.copy()
        logger.info(f"network_scores\n{network_scores.head(15)}")
        grns = network_scores.index.tolist()
        return grns

    def save_query(self, path: str):
        """Save intermediate representations from a query call"""
        if path is None:
            return
        # save query outputs
        gene_score_path = str(Path(path) / Path("gene_scores.csv"))
        self.gene_scores.to_csv(gene_score_path)
        network_scores_path = str(Path(path) / Path("network_scores.csv"))
        self.network_scores.to_csv(network_scores_path)
        network_scores_pruned_path = str(Path(path) / Path("network_scores_pruned.csv"))
        self.network_scores_pruned.to_csv(network_scores_pruned_path)
        return
