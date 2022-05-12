"""Implement PySCENIC as a baseline model"""
from . import ModelAPI
import numpy as np
import pandas as pd 
import scanpy as sc
import anndata
import os
import os.path as osp
import pickle
import logging
# scenic specific
from arboreto.utils import load_tf_names
from arboreto.algo import grnboost2

from ctxcore.rnkdb import FeatherRankingDatabase as RankingDatabase
from pyscenic.utils import modules_from_adjacencies, load_motifs
from pyscenic.prune import prune2df, df2regulons
from pyscenic.aucell import aucell
from dask.diagnostics import ProgressBar


logger = logging.getLogger(__name__)


RESOURCES = {
    "mouse" : {
        "cistarget": "/home/jacob/data/references/mouse/cisTargetDB/mm10__refseq-r80__10kb_up_and_down_tss.mc9nr.feather",
        "motif_annotations" : "/home/jacob/data/references/mouse/motif_annotation/motifs-v9-nr.mgi-m0.001-o0.0.tbl",
    },
    "human": {
        "cistarget": "/home/jacob/data/references/human/cisTargetDB/hg38__refseq-r80__10kb_up_and_down_tss.mc9nr.feather",
        "motif_annotations": "/home/jacob/data/references/human/motif_annotation/motifs-v9-nr.hgnc-m0.001-o0.0.tbl",
    },
}


def get_tf_names(species: str="mouse") -> list:
    """Get the TF names from a species-specific annotations file"""
    df_motifs = pd.read_csv(RESOURCES[species]["motif_annotations"], sep='\t')
    tfs = np.unique(df_motifs.gene_name)
    return tfs.tolist()


def load_databases(species: str="mouse") -> list:
    """Load ranking databases"""
    def name(fname):
        return os.path.splitext(os.path.basename(fname))[0]    
    db_fnames = [RESOURCES[species]["cistarget"]]
    dbs = [RankingDatabase(fname=fname, name=name(fname)) for fname in db_fnames]
    return dbs


def get_ex_matrix(adata, n_cells: int=1000) -> pd.DataFrame:
    # if there are too many cells, downsample following pyscenic tutorial
    if adata.shape[0] > n_cells:
        idx = np.random.choice(adata.shape[0], size=n_cells, replace=False)
    else:
        idx = np.arange(adata.shape[0])
    
    X = adata.X[idx, :] if type(adata.X)==np.ndarray else adata.X[idx, :].toarray()
    ex_matrix = pd.DataFrame(
        X, columns=adata.var_names,
    )
    return ex_matrix    


def get_adjacencies(ex_matrix, tf_names,):
    adjacencies = grnboost2(ex_matrix, tf_names=tf_names, verbose=True)
    return adjacencies


def prune_modules(adjacencies, ex_matrix, dbs, species: str="mouse"):
    modules = list(modules_from_adjacencies(adjacencies, ex_matrix))
    with ProgressBar():
        df = prune2df(dbs, modules, RESOURCES[species]["motif_annotations"])
    return df


class PySCENIC(ModelAPI):

    def __init__(
        self,
        gene_sets: dict,
        species: str="mouse",
        n_cells: int=3000,
        out_path: str="./",
        cell_type_col: str="cell_type",
    ) -> None:
        super(PySCENIC, self).__init__()
        self.species = species
        self.n_cells = n_cells
        # only used for ordering the outputs
        self.gene_set_names = sorted(list(gene_sets.keys()))
        logger.info("Getting TF names")
        self.tf_names = get_tf_names(species=self.species)
        logger.info("Loading databases")
        self.dbs = load_databases(species=self.species)
        self.out_path = out_path
        if out_path is not None:
            os.makedirs(out_path, exist_ok=True)
        return

    def fit(
        self,
        *args,
        adata: anndata.AnnData,
        num_workers: int=8,
        **kwargs,
    ) -> None:
        """Fit SCENIC modules"""
        logger.info("Generating expression matrix")
        ex_matrix_sub = get_ex_matrix(adata=adata, n_cells=self.n_cells)

        logger.info("Generating adjacency graph")
        adjacenies = get_adjacencies(ex_matrix=ex_matrix_sub, tf_names=self.tf_names)

        logger.info("Pruning modules")
        df = prune_modules(
            adjacencies=adjacenies, 
            ex_matrix=ex_matrix_sub, 
            dbs=self.dbs, 
            species=self.species,
        )

        logger.info("Generating regulons")
        regulons = df2regulons(df)
        # save
        self.module_df = df
        self.regulons = regulons
        if self.out_path is not None:
            df.to_csv(osp.join(self.out_path, "module_df.csv"))
            with open(osp.join(self.out_path, "regulons.pkl"), "wb") as f:
                pickle.dump(regulons, f)

        logger.info("Generating AUCell matrix")
        # num_workers kwarg defaults to using total cpu_count() from `multiprocessing`
        ex_matrix = get_ex_matrix(adata=adata, n_cells=adata.shape[0])
        auc_mtx = aucell(ex_matrix, self.regulons,) # output is pd.DataFrame
        if self.out_path is not None:
            auc_mtx.to_csv(osp.join(self.out_path, "auc_mtx.csv"))
        self.auc_mtx = auc_mtx
        return

    def transform(
        self,
        *args,
        adata: anndata.AnnData,
        **kwargs,
    ) -> pd.DataFrame:
        if not hasattr(self, "auc_mtx"):
            print("Fitting before transformation")
            self.fit(adata=adata)
        df = self.auc_mtx.loc[:, [x for x in self.auc_mtx.columns if "+" in x]].copy()
        # rename columns and rearrange to match gene sets
        df.columns = [
            x.replace("(+)", "") for x in df.columns
        ]
        for k in self.gene_set_names:
            if k not in df.columns:
                df[k] = 0.

        df = df.loc[:, self.gene_set_names]
        return np.array(df)

    def query(
        self,
        adata: anndata.AnnData,
        source: str,
        target: str,
    ) -> list:
        self._check_source_target(adata, source, target)

        grn_mat = self.transform(adata=adata)

        tmp_ad = anndata.AnnData(
            X=grn_mat,
            obs=adata.obs.copy(),
        )
        tmp_ad.var_names = np.array(self.gene_set_names)

        sc.tl.rank_genes_groups(
            adata=tmp_ad,
            groupby=self.cell_type_col,
            groups=[target],
            reference=source,
            use_raw=False,
            n_genes=tmp_ad.shape[1],
            method="t-test",
        )
        de_auc = sc.get.rank_genes_groups_df(tmp_ad, group=target)
        de_auc = de_auc.sort_values("scores", ascending=False)
        self.de_auc = de_auc
        grns = de_auc["names"].tolist()
        return grns