import pytest
import os
import os.path as osp
import sys
import numpy as np
import anndata
from typing import Tuple


sys.path.append(osp.abspath('./'))


def _load_data_and_gene_sets() -> Tuple[anndata.AnnData, dict]:
    from pathfinder.bench.identity_bench import load_gene_sets, IDENT_DATASETS
    from pathfinder.bench.reprog_bench import load_dataset, REPROG_DATASETS
    
    adata = load_dataset(
        dataset="tabula_muris_senis_liver",
        dataset_map=REPROG_DATASETS,
    )
    adata.obs["cell_type"] = np.array(adata.obs["cell_ontology_class"])
    gene_sets = load_gene_sets(
        gene_sets_path=IDENT_DATASETS["gene_sets"]["mouse"]["dorothea_tfs"],
    )
    
    return adata, gene_sets


def subsample_by_type(adata):
    cts = np.unique(adata.obs["cell_type"])
    idx = []
    for ct in cts:
        idx.append(
            adata.obs.loc[adata.obs["cell_type"]==ct].index[:25].tolist()
        )
    idx = sum(idx, [])
    return adata[idx].copy()

def test_pyscenic():
    import scanpy as sc
    from pathfinder.models import PySCENIC
    import time
    import pandas as pd
    import pickle

    adata, gene_sets = _load_data_and_gene_sets()
    # subsample
    print("initial data size: ", adata.shape)
    adata = subsample_by_type(adata)
    print("subsampled size: ", adata.shape)

    ps = PySCENIC(
        gene_sets=gene_sets,
        species="mouse",
        n_cells=100,
        out_path="./tmp",
    )

    if os.path.exists("./tmp/auc_mtx.csv"):
        print("Loading existing PySCENIC results for testing.")
        with open(osp.join(ps.out_path, "regulons.pkl"), "rb") as f:
            regulons = pickle.load(f)
        ps.regulons = regulons
        ps.auc_mtx = pd.read_csv("./tmp/auc_mtx.csv", index_col=0)
        print(ps.auc_mtx.head(3))
        print()
    else:
        print("Fitting PySCENIC")
        start = time.time()
        ps.fit(
            adata=adata,
        )
        end = time.time()
        elap = (end-start)/60
        print(f"ELAPSED: {elap} mins")
        print()
    
    X_tfs = ps.transform(adata=adata)
    print("Gata4 scores")
    print(X_tfs[:10, ps.gene_set_names.index("Gata4")])
    print()

    grns = ps.query(
        adata=adata,
        source="hepatic stellate cell",
        target="hepatocyte",
    )
    print("GRns")
    print(grns[:10])
    print("GRN scores")
    print(ps.de_auc.head(15))
    print()
    return


def main():
    import logging
    logging.basicConfig(level=logging.INFO)
    logging.info("INFO statements activated")

    test_pyscenic()
    return


if __name__ == "__main__":
    main()
