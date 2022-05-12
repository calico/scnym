import pytest
import os
import os.path as osp
import sys
import numpy as np
import anndata
import logging
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
        gene_sets_path=IDENT_DATASETS["gene_sets"]["mouse"]["enrichr_tfs"],
    )
    
    return adata, gene_sets


def test_identity_bench():
    """Test benchmarks for recovery of GRNs based on enrichment 
    in specific cell identities"""
    from pathfinder.bench.identity_bench import run_bench, IDENT_DATASETS, IDENT_GRN_PAIRS
    from pathfinder.models import AUCellReprog
    
    adata, gene_sets = _load_data_and_gene_sets()
    print("Loaded data.")
    print("%d cells, %d genes." % adata.shape)
    print(f"{len(gene_sets.keys())} gene sets.")
    print()
    
    # setup model
    # use a subsample of gene sets to reduce computation time
    model = AUCellReprog(
        gene_sets={
            k: gene_sets[k] for k in ["Hnf1a", "Pax5", "Pax7",]
        },
    )
    
    # use only a single dataset
    dataset_map = {k: IDENT_DATASETS[k] for k in ["tabula_muris_senis_liver"]}
    auc_scores = run_bench(
        model_api=model,
        dataset_map=dataset_map,
        ident_grn_pairs=IDENT_GRN_PAIRS,
    )
    
    print("AUC scores")
    print(auc_scores.head())
    return


def test_get_reprog_strategy_ranks():
    """Test ranking of reprogramming GRNs for known conversions"""
    from pathfinder.bench.reprog_bench import (
        run_bench, load_known_conversions, REPROG_DATASETS, get_reprog_strategy_ranks
    )
    from pathfinder.models import DEReprog
    
    adata, gene_sets = _load_data_and_gene_sets()
    print("Loaded data.")
    print("%d cells, %d genes." % adata.shape)
    print(f"{len(gene_sets.keys())} gene sets.")
    print()
    
    conversions = load_known_conversions()
    print("Loaded conversions.")
    print("%d conversions listed." % conversions.shape[0])
    
    # use the DE model since it's the fastest to compute
    model = DEReprog(
        gene_sets=gene_sets,
    )
    
    # test reprogramming ranking
    ranks = get_reprog_strategy_ranks(
        adata=adata,
        model_api=model,
        conversions=conversions,
    )
    print("Ranks")
    print(ranks.head(10))
    return
    

def test_reprog_bench():
    """Test benchmarks for recovering known reprogramming routes"""
    from pathfinder.bench.reprog_bench import (
        run_bench, load_known_conversions, REPROG_DATASETS, run_bench
    )
    from pathfinder.models import DEReprog
    
    adata, gene_sets = _load_data_and_gene_sets()
    print("Loaded data.")
    print("%d cells, %d genes." % adata.shape)
    print(f"{len(gene_sets.keys())} gene sets.")
    print()
    
    conversions = load_known_conversions()
    print("Loaded conversions.")
    print("%d conversions listed." % conversions.shape[0])
    
    # use the DE model since it's the fastest to compute
    model = DEReprog(
        gene_sets=gene_sets,
    )
    # use a two datasets to speed computation
    dataset_map = {
        k: REPROG_DATASETS[k] for k in ("tabula_muris_senis_liver", "ximerakis_brain",)
    }
    ranks = run_bench(
        model_api=model,
        dataset_map=dataset_map,
        conversions=conversions,
    )
    print("Ranks")
    print(ranks.head(30))    
    return
    
    
def test_mef2neuro_bench():
    from pathfinder.bench.mef2neuro_bench import (
        get_model_rank_correlation
    )
    from pathfinder.bench.identity_bench import load_gene_sets, IDENT_DATASETS
    from pathfinder.models import DEReprog, Fauxgrify
    
    gene_sets = load_gene_sets(
        gene_sets_path=IDENT_DATASETS["gene_sets"]["mouse"]["dorothea_tfs"],
    )    
    # use the DE model since it's the fastest to compute
    model = DEReprog(
        gene_sets=gene_sets,
    )
    
    ranks, results = get_model_rank_correlation(
        model_api=model,
    )
    print("DE")
    print("ranks")
    print(ranks.head(20))
    print("ranks (by eff)")
    print(ranks.sort_values("experiment_efficiency", ascending=False).head(20))
    print("results")
    print(results.head(10))

    # also test Fauxgrify
    model = Fauxgrify(
        gene_sets=gene_sets,
        max_source_exprssion=3.,
        degenerate_thresh=1.1,
        max_rank_considered=len(gene_sets.keys())+1,
    )
    
    ranks, results = get_model_rank_correlation(
        model_api=model,
    )
    print("Fauxgrify")
    print("ranks")
    print(ranks.head(20))
    print("ranks (by eff)")
    print(ranks.sort_values("experiment_efficiency", ascending=False).head(20))
    print("results")
    print(results.head(10))
 
    return

def test_emt_bench():
    from pathfinder.bench.emt_bench import (
        get_model_emt_ranks
    )
    from pathfinder.bench.identity_bench import load_gene_sets, IDENT_DATASETS
    from pathfinder.models import DEReprog, Fauxgrify

    gene_sets = load_gene_sets(
        gene_sets_path=IDENT_DATASETS["gene_sets"]["human"]["dorothea_mod_tfs"],
    )    
    # use the DE model since it's the fastest to compute
    model = DEReprog(
        gene_sets=gene_sets,
    )

    results, spon_rank, tgfb_rank = get_model_emt_ranks(
        model_api=model,
    )
    print("DE")
    print("results")
    print(results)
    print("spon_rank")
    print(spon_rank.head(5))
    print("tgfb_rank")
    print(tgfb_rank.head(5))    

    # model = Fauxgrify(
    #     gene_sets=gene_sets,
    #     max_source_exprssion=20.,
    #     degenerate_thresh=1.1,
    #     max_rank_considered=len(gene_sets.keys())+1,
    # )
    # results, spon_rank, tgfb_rank = get_model_emt_ranks(
    #     model_api=model,
    # )
    # print("Fauxgrify")
    # print("results")
    # print(results)
    # print("spon_rank")
    # print(spon_rank.head(5))
    # print("tgfb_rank")
    # print(tgfb_rank.head(5))
    return


def test_calc_parekh_tf_ox_bench():
    """Test calculation of TF OX recall in the Parekh 2018 dataset"""
    from pathfinder.bench.identity_bench import load_gene_sets, IDENT_DATASETS
    from pathfinder.bench.tf_ox_bench import (
        TF_OVEXP_DATASETS, calc_parekh_tf_ox_bench, load_data
    )
    from pathfinder.models import DEReprog, AUCellReprog
    import pandas as pd
    import numpy as np
    
    print("loading data")
    adata = load_data()
    print(adata)
    print("TF names")
    print(np.unique(adata.obs[TF_OVEXP_DATASETS["parekh_Hs"]["cell_type_col"]]).tolist())
    print()
    print(adata.obs[TF_OVEXP_DATASETS["parekh_Hs"]["cell_type_col"]].cat.categories)
    print()
    print(adata.obs[TF_OVEXP_DATASETS["parekh_Hs"]["cell_type_col"]].cat.categories.values)
    gene_sets = load_gene_sets(
        gene_sets_path=IDENT_DATASETS["gene_sets"]["human"]["dorothea_mod_tfs"],
    )
    
    # test on a few TFs from gene_sets so goes faster
    gene_sets = {
        k: v for k, v in gene_sets.items() if k in ("HAND2", "GATA4", "ASCL1", "POU5F1")
    }
    print(f"Subset to {len(gene_sets.keys())} gene sets.")
    
    # define two toy models
    print("building models")
    de = DEReprog(
        gene_sets=gene_sets,
        cell_type_col=TF_OVEXP_DATASETS["parekh_Hs"]["cell_type_col"],
    )
    auc = AUCellReprog(
        gene_sets=gene_sets,
        cell_type_col=TF_OVEXP_DATASETS["parekh_Hs"]["cell_type_col"],
        refit_per_query=True,
    )
    
    models_dict = {
        "de": de,
        "auc": auc,
    }
    print("running rank calculation")
    metrics, model_ranks = calc_parekh_tf_ox_bench(
        models_dict=models_dict,
        rank_cutoff=96,
    )
    print("metrics")
    print(metrics)
    print("model_ranks")
    print(model_ranks.head())
    assert type(metrics)==pd.DataFrame
    assert type(model_ranks)==pd.DataFrame
    return

    

def main():
    logging.basicConfig(level=logging.INFO)
    # test_identity_bench()
    # test_get_reprog_strategy_ranks()
    # test_reprog_bench()
    test_mef2neuro_bench()
    # test_emt_bench()
    # test_calc_parekh_tf_ox_bench()
    return


if __name__ == "__main__":
    main()
