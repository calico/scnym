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
    
    gene_sets = load_gene_sets(
        gene_sets_path=IDENT_DATASETS["gene_sets"]["mouse"]["dorothea_tfs"],
    )
    
    return adata, gene_sets


def test_gene_set_corr_filtering():
    from pathfinder.models import DEReprog
    import time
    adata, gene_sets = _load_data_and_gene_sets()
    print("Loaded data.")
    print("%d cells, %d genes." % adata.shape)
    print(f"{len(gene_sets.keys())} gene sets.")
    print()
    model = DEReprog(
        gene_sets=gene_sets,
    )

    # should be no-ops
    assert model.fit(1, 2) is None
    assert model.predict(1,) is None
    assert model.transform(1,) is None

    print("Filtering gene sets by correlation")
    start = time.time()
    model._prune_gene_sets(adata=adata)
    model._filter_tf_target_correlation(
        adata=adata,
        corr_min=0.0,
        corr_max=None,
    )

    print("\nFiltering complete.\n")
    end = time.time()
    elapsed = (end - start)/60
    print(f"Time: {elapsed} minutes")
    print("Gene set sizes:")
    for k in model.gene_sets.keys():
        print(f"{k}\t\t{len(model.gene_sets[k])}")
    return


def test_gene_set_freq_filtering():
    from pathfinder.models import DEReprog
    import time
    adata, gene_sets = _load_data_and_gene_sets()
    print("Loaded data.")
    print("%d cells, %d genes." % adata.shape)
    print(f"{len(gene_sets.keys())} gene sets.")
    print()
    model = DEReprog(
        gene_sets=gene_sets,
    )

    # should be no-ops
    assert model.fit(1, 2) is None
    assert model.predict(1,) is None
    assert model.transform(1,) is None

    print("Filtering gene sets for frequent targets")
    start = time.time()
    model._prune_gene_sets(adata=adata)
    model._filter_frequent_targets(
        adata=adata,
    )

    print("\nFiltering complete.\n")
    end = time.time()
    elapsed = (end - start)/60
    print(f"Time: {elapsed} minutes")
    print("Gene set sizes:")
    return    


def test_de():
    from pathfinder.models import DEReprog
    
    adata, gene_sets = _load_data_and_gene_sets()
    print("Loaded data.")
    print("%d cells, %d genes." % adata.shape)
    print(f"{len(gene_sets.keys())} gene sets.")
    print()
    model = DEReprog(
        gene_sets=gene_sets,
    )

    # should be no-ops
    assert model.fit(1, 2) is None
    assert model.predict(1,) is None
    assert model.transform(1,) is None

    print("Querying model...")
    # test a simple conversion
    grns = model.query(
        adata=adata,
        source="hepatic stellate cell",
        target="hepatocyte",
    )
    # should have an index lower than 100
    idx = grns.index("Hnf1a")
    assert idx < 100
    print(f"stellate -> hepatocyte Hnf1a rank: {idx+1}")
    print("Done.")
    return


def test_aucell():
    from pathfinder.models import AUCellReprog
    from scipy import sparse
    
    adata, gene_sets = _load_data_and_gene_sets()
    print("Loaded data.")
    print("%d cells, %d genes." % adata.shape)
    print(f"{len(gene_sets.keys())} gene sets.")
    print()

    # test with a sparse matrix
    adata.X = sparse.csr_matrix(adata.X)
    
    print("Querying model...")
    # use a subsample of gene sets to reduce computation time
    model = AUCellReprog(
        gene_sets={
            k: gene_sets[k] for k in ["Hnf1a", "Pax5", "Pax7", "Foxa2"]
        },
    )

    # should be no-ops
    assert model.predict(1,) is None

    model.fit(X=adata.X, y=adata.obs["cell_ontology_class"], adata=adata)

    # should fit model to a subset of data
    grns = model.query(
        adata=adata,
        source="hepatic stellate cell",
        target="hepatocyte",    
    )

    # should have an index lower than 100
    idx = grns.index("Hnf1a")
    print(f"stellate -> hepatocyte Hnf1a rank: {idx+1}")

    # should transform data
    print("Embedding cells to TF space...")
    X_tfs = model.transform(X=None,)
    print("%d cells, %d GRN scores." % X_tfs.shape)
    print("\tn_cells should be smaller than the original data.")
    print("\tbecause we subset to source/target classes.")
    print("Done.")
    return


def test_gsea():
    from pathfinder.models import GSEAReprog
    
    adata, gene_sets = _load_data_and_gene_sets()
    print("Loaded data.")
    print("%d cells, %d genes." % adata.shape)
    print(f"{len(gene_sets.keys())} gene sets.")
    print()
    # subset to a few gene sets to speed up computation
    # set `random_sets` to a lower value than usual to
    # use a lest robust empritical null -> faster computation
    # NOTE: This test will use our multithreaded version of gsea.py
    model = GSEAReprog(
        gene_sets={
            k: gene_sets[k] for k in ["Hnf1a", "Neurod1", "Foxa2", "Pax7"]
        },
        random_sets=100,
    )

    # should be no-ops
    assert model.fit(1, 2) is None
    assert model.predict(1,) is None
    assert model.transform(1,) is None

    print("Querying model...")
    # test a simple conversion
    # load #s of hepatic stellate cells lead to errors
    # when using few permutations
    # switch source to endothelial
    grns = model.query(
        adata=adata,
        source="endothelial cell of hepatic sinusoid",
        target="hepatocyte",
    )
    # should have an index lower than 100
    idx = grns.index("Hnf1a")
    assert idx < 100
    print(f"endothelial -> hepatocyte Hnf1a rank: {idx+1}")
    print("Done.")
    return


def test_fauxgrify():
    from pathfinder.models.baseline import Fauxgrify
    
    adata, gene_sets = _load_data_and_gene_sets()
    print("Loaded data.")
    print("%d cells, %d genes." % adata.shape)
    print(f"{len(gene_sets.keys())} gene sets.")
    print()    

    gene_sets = {
        k: gene_sets[k] for k in ["Hnf1a", "Neurod1", "Foxa2", "Pax7", "Sox2", "Hoxc10"]
    } 
    
    model = Fauxgrify(
        gene_sets=gene_sets,
    )
    
    # all should be no-ops
    assert model.predict(adata.X) is None
    assert model.transform(adata.X) is None
    assert model.fit(adata.X, adata.obs["cell_type"]) is None
    
    source = "Kupffer cell"
    target = "hepatocyte"
    
    grns = model.query(
        adata=adata,
        source=source,
        target=target,
    )
    idx = grns.index("Hnf1a")
    print(f"endothelial -> hepatocyte Hnf1a rank: {idx+1}")
    print("Done.")
    assert idx < 3
    
    print("\ngene_scores")
    print(model.gene_scores.head(15))
    print()
    return


def test_scbasset():
    from pathfinder.models import DATASETS, pscBasset

    model = pscBasset(DATASETS["mm_brain"])
    adata = anndata.read_h5ad(DATASETS["mm_brain"]["path"]+"/h5ad/rna_raw.h5ad")
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.log1p(adata)


def main():
    import logging
    logging.basicConfig(level=logging.INFO)
    print("sys.path")
    print(sys.path)
    import pathfinder
    print(
        "pathfinder path:",
        pathfinder.__file__,
    )
    print()
    # test_gene_set_corr_filtering()
    test_gene_set_freq_filtering()
    # test_de()
    # test_aucell()
    # test_gsea()
    # test_fauxgrify()
    return


if __name__ == "__main__":
    main()
