#!/bin/env python3
"""Benchmark the performance of baseline models on reprogramming tasks"""
import os
import os.path as osp
import json
import sys
import time

import torch
import numpy as np
import pandas as pd
import anndata

sys.path = [osp.abspath("../../lib"),] + sys.path
import pathfinder

# TODO: This is a hack for experiments where we forgot to save names in the params file
DS_NAME_DEFAULT = "tabula_muris"


# utility functions
def get_gene_sets(params: dict, adata: anndata.AnnData):
    gene_sets = {}
    with open(params["data::gene_sets"], "r") as f:
        for line in f:
            values = line.split("\t")
            # add the TF mRNA to the gene set, values[0]
            gene_sets[values[0]] = [
                x for x in values[1:] if x not in ("", " ", "\n")
            ] + [values[0],]
            assert values[0] in gene_sets[values[0]]
            gene_sets[values[0]] = [
                x for x in gene_sets[values[0]] if x in adata.var_names
            ]
            if len(gene_sets[values[0]]) < 10:
                print(values[0], len(gene_sets[values[0]]))

    print(f"# of TF gene sets: {len(gene_sets)}")
    return gene_sets


def _cast_adata_with_genes(adata: anndata.AnnData, gene_names: list):
    import scnym
    X = scnym.utils.build_classification_matrix(
        X=adata.X,
        model_genes=gene_names,
        sample_genes=np.array(adata.var_names),
        gene_batch_size=2048,
    )
    new_adata = anndata.AnnData(X=X, obs=adata.obs)
    new_adata.var_names = gene_names
    return new_adata


def get_data(params: dict) -> anndata.AnnData:
    """
    Parameters
    ----------
    params : dict
        parameters dictionary passed to `run_scnym_sparse`.
        keys are `key_type::key_name` with various value types.
    
    Returns
    -------
    adata : anndata.AnnData
        [Cells, Genes] genes will match the model.
    """
    adata = anndata.read_h5ad(params["data::path"])
    # filter out nan values in either groupby
    rm_bidx = pd.isna(adata.obs[params["data::groupby"]])
    rm_bidx = rm_bidx | pd.isna(adata.obs[params["data::domain_groupby"]])
    rm_bidx = rm_bidx | (adata.obs[params["data::groupby"]]=="nan")
    rm_bidx = rm_bidx | (adata.obs[params["data::domain_groupby"]]=="nan")
    adata = adata[~rm_bidx, :].copy()

    # subset genes if specified
    if params.get("data::gene_names", False):
        gene_names = np.loadtxt(params["data::gene_names"], dtype=np.str)
        adata = _cast_adata_with_genes(adata, gene_names)
    elif params.get("data::unlabeled_path", False):
        # get shared gene names
        unlab = anndata.read_h5ad(params["data::unlabeled_path"])
        print("Subsetting labeled and unlabeled data to shared genes")
        gene_names = [x for x in adata.var_names if x in unlab.var_names]
        adata = adata[:, gene_names].copy()
    else:
        print("No unlabeled data was loaded.")
    # match cell types to those used in the reprogramming benchmarks
    n2n = {
        "astrocyte of the cerebral cortex": "astrocyte",
        "type B pancreatic cell": "pancreatic beta cell",
    }
    adata.obs[params["data::groupby"]] = np.array(adata.obs[params["data::groupby"]])  
    adata.obs[params["data::groupby"]] = adata.obs[params["data::groupby"]].apply(
        lambda x: n2n.get(x, x)
    )
    # some of our reprogramming benchmark tools assume `cell_type` as the col name
    # it's easier to just copy it here than to change it everywhere else
    adata.obs["cell_type"] = np.array(adata.obs[params["data::groupby"]])    
    return adata


def run_ident_bench(bsln_model, adata,):
    X_grn = bsln_model.transform(
        X=adata.X, adata=adata
    )
    if X_grn is None:
        # transform is a no-op, not supported by this model
        return None

    def noop(*args,**kwargs):
        return

    bsln_model.fit = noop

    grn_ad = anndata.AnnData(
        X=X_grn,
        obs=adata.obs.copy(),
    )
    grn_ad.var_names = bsln_model.gene_set_names

    ident_auc_scores = pathfinder.bench.identity_bench.score_identity_specific_grns(
        adata=grn_ad,
        ident_grn_pairs=pathfinder.bench.identity_bench.IDENT_GRN_PAIRS,
    )
    return ident_auc_scores


def run_reprog_bench(bsln_model, adata, args, name: str):
    save_path = osp.join(args.out_path, f"reprog_bench_{name}")
    os.makedirs(save_path, exist_ok=True)
    conversions = pathfinder.bench.reprog_bench.load_known_conversions()
    reprog_ranks = pathfinder.bench.reprog_bench.query_trained_model(
        model_api=bsln_model,
        adata=adata,
        conversions=conversions,
        save_path=save_path,
    )
    return reprog_ranks


def build_parser():
    import argparse
    parser = argparse.ArgumentParser(
        description="Benchmark baseline models."
    )
    parser.add_argument(
        "exp_manifest", 
        type=str, 
        help="path to an experiment manifest file.")
    parser.add_argument(
        "exp_name", 
        type=str, 
        help="experiment to benchmark. must be a key in the manifest."
    )
    parser.add_argument(
        "--out_path", type=str, 
        help="path for outputs. uses params['out_path'] by default.", 
        default=None,
    )
    parser.add_argument(
        "--species", type=str, default="mouse",
        help="species for PySCENIC. {mouse, human}."
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    with open(args.exp_manifest, "r") as f:
        manifest = json.load(f)
    params = manifest[args.exp_name]
    if args.out_path is None:
        args.out_path = params["out_path"]

    # load dataset
    adata = get_data(params=params["params"])
    # get gene sets
    gene_sets = get_gene_sets(params=params["params"], adata=adata)

    baseline_models = {
        "DE": pathfinder.models.DEReprog(
            gene_sets=gene_sets,
            cell_type_col="cell_type",
        ),
        "Fauxgrify": pathfinder.models.Fauxgrify(
            gene_sets=gene_sets,
            max_rank_considered=len(gene_sets)+1,
            degenerate_thresh=0.98,
        ),
        "AUCell": pathfinder.models.AUCellReprog(
            gene_sets=gene_sets,
            refit_per_query=False,
        ),
        "PySCENIC": pathfinder.models.PySCENIC(
            gene_sets=gene_sets,
            n_cells=15000,
            species=args.species,
            out_path=osp.join(args.out_path, "pyscenic"),
            cell_type_col="cell_type",
        ),
    }

    ident_dfs = []
    rprg_dfs = []
    for name, bsln_model in baseline_models.items():

        print(f"Running benchmarks for {name}")
        os.makedirs(osp.join(args.out_path, name), exist_ok=True)
        start = time.time()
        print("\tRunning identity-specific GRN benchmark")
        ident_df = run_ident_bench(bsln_model, adata)
        print("\tRunning reprogramming factor ranking benchmark")
        rprg_df = run_reprog_bench(bsln_model, adata, args, name)

        if ident_df is not None:
            # not all methods support transformation to GRN scores
            ident_df.to_csv(osp.join(args.out_path, name, "ident_bench_auc.csv"))
            ident_df["method"] = name
            ident_dfs.append(ident_df)
        
        # all methods support the reprogramming TF ranking
        rprg_df.to_csv(osp.join(args.out_path, name, "reprog_bench_ranks.csv"))
        rprg_df["method"] = name
        rprg_dfs.append(rprg_df)

        end = time.time()
        dur = (end - start)/60
        print(f"\t duration: {dur} minutes")

    if len(ident_dfs) > 0:
        # we might not always test baselines that can do the identity GRN scoring
        ident_df = pd.concat(ident_dfs, axis=0)
        ident_df.to_csv(osp.join(args.out_path, "baseline_ident_bench_auc.csv"))

    rprg_df = pd.concat(rprg_dfs, axis=0)
    rprg_df.to_csv(osp.join(args.out_path, "baseline_reprog_bench_ranks.csv"))

    return


if __name__ == "__main__":
    main()
