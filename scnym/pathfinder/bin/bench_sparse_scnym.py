#!/bin/env python3
import os
import os.path as osp
import json
import sys

import torch
import numpy as np
import pandas as pd
from scipy import stats
from skimage.filters import threshold_otsu
import anndata

sys.path = [osp.abspath("../../lib"),] + sys.path
import pathfinder

# TODO: This is a hack for experiments where we forgot to save names in the params file
DS_NAME_DEFAULT = "tabula_muris"


############################################################
# utility functions
############################################################


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
    X = scnym.utils.build_classification_matrix(
        X=adata.X,
        model_genes=np.array(gene_names),
        sample_genes=np.array(adata.var_names),
        gene_batch_size=2048,
    )
    new_adata = anndata.AnnData(X=X, obs=adata.obs)
    new_adata.var_names = gene_names
    return new_adata


def get_data(params):
    adata = anndata.read_h5ad(params["data::path"])
    # filter out nan values in either groupby
    rm_bidx = pd.isna(adata.obs[params["data::groupby"]])
    rm_bidx = rm_bidx | pd.isna(adata.obs[params["data::domain_groupby"]])
    rm_bidx = rm_bidx | (adata.obs[params["data::groupby"]]=="nan")
    rm_bidx = rm_bidx | (adata.obs[params["data::domain_groupby"]]=="nan")
    adata = adata[~rm_bidx, :].copy()

    # load unlabeled data if provided
    if params.get("data::unlabeled_path", False):
        unlab = anndata.read_h5ad(params["data::unlabeled_path"])
    else:
        unlab = None
    
    # subset genes if specified
    if params.get("data::gene_names", False):
        gene_names = np.loadtxt(params["data::gene_names"], dtype=np.str)
        adata = _cast_adata_with_genes(adata, gene_names)
    elif unlab is not None and not params.get("data::all_grn_genes", False):
        # get shared gene names
        print("Subsetting labeled and unlabeled data to shared genes")
        gene_names = [x for x in adata.var_names if x in unlab.var_names]
        adata = adata[:, gene_names].copy()
    elif unlab is not None and params.get("data::all_grn_genes", False):
        # keep all genes that match a GRN, even if they are only expressed in the
        # unlabeled data
        print("Subsetting labeled and unlabeled data to shared genes + all GRNs")
        gene_names = [x for x in adata.var_names if x in unlab.var_names]
        print(f"Found {len(gene_names)} genes shared.")
        print(gene_names[:10])
        n_before = len(gene_names)
        # also include TF mRNAs that are present only in the unlabeled data
        gene_set_names = []
        with open(params["data::gene_sets"], "r") as f:
            for l in f:
                gene_set_names.append(l.split("\t")[0])
        print(f"Found {len(gene_set_names)} GRN mRNAs to consider.")
        gene_names += [
            x for x in gene_set_names 
            if ((x in unlab.var_names or x in adata.var_names) and x not in gene_names)
        ]
        print(f"Kept {len(gene_names)-n_before} GRN mRNAs in only one dataset.")
        print(f"Found {len(gene_names)} total genes to keep.")
        adata = _cast_adata_with_genes(adata, gene_names)
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


def get_pfnd_model(params, adata, gene_sets, args):
    chk_name = {
        "final": "01_final_model_weights.pkl",
        "best": "00_best_model_weights.pkl",
    }.get(args.checkpoint, None)
    if chk_name is None:
        chk_name = "model_weights_%03d.pkl" % int(args.checkpoint)

    path_to_weights = osp.join(params["out_path"], chk_name)

    pfnd = pathfinder.models.Pathfinder(
        gene_sets=gene_sets,
        log_dir=params["out_path"],
        cell_type_col=params["params"]["data::groupby"],
        domain_groupby=params["params"]["data::domain_groupby"],
        n_sparse_latent=0,
        n_dense_latent=0,
        scoring_metric=args.scoring_metric,
        grad_activation="first_layer",
        model_kwargs={
            "hidden_init_dropout": params["params"].get("model::hidden_init_dropout", True),
        },
    )
    # load weights
    pfnd.load(
        path_to_weights, 
        X=adata.X,
        y=adata.obs[params["params"]["data::groupby"]], 
        adata=adata,
    )

    pfnd.query_kwargs = {
        "only_source_reference": True,
        "n_cells": 3000,
        "n_batches": 100,
    }

    print("Loaded Pathfinder model")
    print(pfnd.query_kwargs)

    return pfnd

############################################################
# cell identity-specific GRN benchmark
############################################################


def run_ident_bench(pfnd, adata, args):
    X_grn = pfnd.transform(
        X=adata.X, adata=adata
    )
    pfnd._fit_before_query = False
    pfnd._setup_fit_before_query = False

    def noop(*args,**kwargs):
        return

    pfnd.fit = noop

    grn_ad = anndata.AnnData(
        X=X_grn,
        obs=adata.obs.copy(),
    )
    grn_ad.var_names = pfnd.gene_set_names

    ident_auc_scores = pathfinder.bench.identity_bench.score_identity_specific_grns(
        adata=grn_ad,
        ident_grn_pairs=pathfinder.bench.identity_bench.IDENT_GRN_PAIRS,
    )
    ident_auc_scores.to_csv(osp.join(args.out_path, "ident_bench_auc.csv"))

    print("Ident AUC mean:")
    print(ident_auc_scores.mean(0))
    return

############################################################
# reprogramming recall benchmark
############################################################


def run_reprog_bench(pfnd, adata, args):
    save_path = osp.join(args.out_path, "reprog_bench_pfnd")
    os.makedirs(save_path, exist_ok=True)
    conversions = pathfinder.bench.reprog_bench.load_known_conversions()
    reprog_ranks = pathfinder.bench.reprog_bench.query_trained_model(
        model_api=pfnd,
        adata=adata,
        conversions=conversions,
        save_path=save_path,
    )
    reprog_ranks.to_csv(osp.join(args.out_path, "reprog_bench_ranks.csv"))

    reprog_ranks["top_k"] = reprog_ranks["rank"] < 384
    print("Reprog Ranks Mean")
    print(reprog_ranks.mean(0))
    return


############################################################
# fibroblast -> neuron conversion combinations benchmark
############################################################


def px_grns(
    X: np.ndarray, px_sets: list, px_val: float, gene_set_names: list, px_op: np.add
) -> np.ndarray:
    X = X.copy()
    grn_idx = [gene_set_names.index(x) for x in px_sets]
    X[:, grn_idx] = px_op(X[:, grn_idx], px_val)
    X = np.clip(X, a_min=0, a_max=None)
    return X


def get_ct_scores(
    X: np.ndarray, model: torch.nn.Module, source: str, target: str, class_names: list, detach=True,
) -> np.ndarray:

    Xt = torch.from_numpy(X).float() if type(X) == np.ndarray else X

    scores = model(Xt)
    probs = torch.nn.functional.softmax(scores, dim=-1)
    
    s_idx = class_names.index(source)
    t_idx = class_names.index(target)

    if detach:
        return (
            probs[:, [s_idx, t_idx]].detach().cpu().numpy(), 
            scores[:, [s_idx, t_idx]].detach().cpu().numpy(),
        )
    else:
        return probs[:, [s_idx, t_idx]], scores[:, [s_idx, t_idx]]


def finite_diffs(X, model, px_sets, source, target, class_names, gene_set_names):
    """Compute the gradient in the direction of a combinatorial GRN perturbation
    using a series of finite-difference step sizes.
    """
    px_vals = np.linspace(0, 100, 25)
    # accumulate score estimates
    accum_scores = []
    accum_probs = []

    for pxv in px_vals:
        X_px = px_grns(
            X=X, 
            px_sets=px_sets, 
            px_val=pxv, 
            gene_set_names=gene_set_names, 
            px_op=np.add,
        ) 
        # both are [Cells, (source, target)]
        probs, scores = get_ct_scores(
            X=X_px, 
            model=model,
            source=source, 
            target=target, 
            class_names=class_names,
            detach=True,
        )
        accum_scores.append(scores)
        accum_probs.append(probs)
    
    # reshape to [Steps, Cells, (Source, Target)]
    aprobs = np.stack(accum_probs, 0)
    ascores = np.stack(accum_scores, 0)

    # compute gradients
    # now [Steps-1, Cells, (Source, Target)]
    g_p = aprobs[1:] - aprobs[:-1]
    g_s = ascores[1:] - ascores[:-1]

    # now [Steps-1, (source, target)]
    grads_p = g_p.mean(1)
    grads_s = g_s.mean(1)

    return grads_p, grads_s


def score_m2n_combinations(adata, exp_data, px_groups, pfnd) -> pd.DataFrame:
    """Compute the predicted effect of bHLH:POU TF pairs on neuron class scores"""
    # transform data to get `X_grn` embedding
    _ = pfnd.transform(
        X=adata.X,
        adata=adata,
    )

    # get the LV -> classif probs model
    lv2clf = torch.nn.Sequential(
        *pfnd.model.mid_stack,
        *pfnd.model.hidden_layers,
        pfnd.model.classif,
    )
    lv2clf = lv2clf.eval()

    # get the GRN embedding scores
    src_bidx = adata.obs["cell_ontology_class"].isin(["fibroblast",])
    # tgt_bidx = adata.obs["cell_ontology_class"].isin(["neuron"])

    X_grn_src = adata.obsm["X_grn"][src_bidx]
    # X_grn_tgt = adata.obsm["X_grn"][tgt_bidx]

    # score perturbation effects
    results = pd.DataFrame(
        index=px_groups,
        columns=["s_src", "s_tgt", "p_effect", "n_grn", "sum_singles"],
    )
    for k in results.columns:
        results[k] = 0.
        results[k] = results[k].astype(float)

    for px_sets in px_groups:
        px_sets2use = [x for x in px_sets if x!="None"]

        grads_p, grads_s = finite_diffs(
            X=X_grn_src,
            model=lv2clf,
            px_sets=px_sets2use,
            source="fibroblast",
            target="neuron",
            class_names=np.unique(adata.obs["cell_ontology_class"]).tolist(),
            gene_set_names=pfnd.gene_set_names,
        )

        singleton_idx = [(x, "None") if (x, "None") in px_groups else ("None", x) for x in px_sets]
        sum_effect = (
            np.sum(results.loc[singleton_idx, "p_effect"]) 
            if len(px_sets2use) > 1 
            else
            grads_p.sum(0)[-1]
        )
        results.loc[[px_sets], :] = (
            grads_p.sum(0)[0],
            grads_p.sum(0)[-1],
            grads_p.sum(0)[-1],
            len([x for x in px_sets if x!="None"]),
            sum_effect,
        )
    
    # add experiment results
    results["exp_eff"] = np.array(
        exp_data.set_index(["bHLH", "POU"]).loc[results.index, "efficiency"]
    )

    # add post-proc [0, 1] scaled versions of outputs
    # mm := "min max scaled"
    def min_max(x):
        return (x - x.min()) / (x - x.min()).max()

    results["exp_eff_mm"] = min_max(results["exp_eff"])
    results["p_effect_mm"] = min_max(results["p_effect"])
    results["sum_singles_mm"] = min_max(results["sum_singles"])

    results["p_effect_resid"] = np.abs(results["p_effect"] - results["exp_eff"])
    results["sum_singles_resid"] = np.abs(results["sum_singles"] - results["exp_eff"])

    print("f2n pearson for p_effect")
    t = stats.pearsonr(
        results["exp_eff"],
        results["p_effect"]
    )
    print(t)
    print("f2n pearson for sum_singles")
    t = stats.pearsonr(
        results["exp_eff"],
        results["sum_singles"]
    )
    print(t)

    print("Thresholded Spearman for k=2 GRN perturbations")

    a = results.loc[results["n_grn"]==2, "exp_eff"]
    b = np.array(results.loc[results["n_grn"]==2, "p_effect"]).copy()

    b = np.array(b)
    b[b < threshold_otsu(b)] = 0

    b = np.clip(b, 0, None)
    t = stats.spearmanr(
        a, b
    )
    print(t)

    return results


def run_m2n_bench(model_api, adata, args):
    """Run the mef2neuro conversion bencmark
    
    Notes
    -----
    Assumes the model is trained on a dataset with "fibroblast" and "neuron" classes.
    """
    import itertools
    exp_data, bhlh_genes, pou_genes = (
        pathfinder.bench.mef2neuro_bench.load_experimental_data()
    )
    # remove dups
    exp_data = exp_data.set_index(["bHLH", "POU"])
    exp_data = exp_data.loc[~exp_data.index.duplicated()]
    exp_data = exp_data.reset_index()    

    # filter to combinations that are in the gene sets
    bhlh_genes = [x for x in bhlh_genes if x in model_api.gene_sets.keys()]
    pou_genes = [x for x in pou_genes if x in model_api.gene_sets.keys()]
    # get a list of all possible bHLH:POU factor pairs and singletons
    px_groups = list(itertools.product(bhlh_genes+["None"], pou_genes+["None"]))

    grns = model_api.query(
        adata=adata,
        source="fibroblast",
        target="neuron",
    )    

    comb_df = pathfinder.bench.mef2neuro_bench.rank_bhlh_pou_from_singletons(
        grns=grns,
        bhlh_genes=bhlh_genes,
        pou_genes=pou_genes,
    )    
    results, comp_ranks_df = pathfinder.bench.mef2neuro_bench.get_rank_correlation(
        comb_df=comb_df,
        exp_scores=exp_data,
    )
    save_path = osp.join(args.out_path, "mef2neuro_naive_results.csv")
    results.to_csv(save_path)
    save_path = osp.join(args.out_path, "mef2neuro_naive_ranks.csv")
    comp_ranks_df.to_csv(save_path)

    # subset the adata to relevant tissues
    bidx = (
        adata
        .obs["tissue"]
        .isin(["Brain_Non-Myeloid", "Heart", "Heart_and_Aorta", "Lung", "Mammary_Gland"])
    )
    ad_sub = adata[bidx].copy()
    # perform perturbation based rankings
    px_results = score_m2n_combinations(
        adata=ad_sub,
        exp_data=exp_data,
        px_groups=px_groups,
        pfnd=model_api,
    )
    save_path = osp.join(args.out_path, "mef2neuro_px_results.csv")
    px_results.to_csv(save_path)
    return

############################################################
# main
############################################################


def build_parser():
    import argparse
    parser = argparse.ArgumentParser(
        description="Benchmark a set of scNym weights trained on a mouse atlas."
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
        "--scoring_metric", type=str, default="intgrad_lv",
    )
    parser.add_argument(
        "--checkpoint", type=str, default="final",
        help="checkpoint to use. one of (best, final, int epoch number)."
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

    chk_name = {
        "final": "01_final_model_weights.pkl",
        "best": "00_best_model_weights.pkl",
    }.get(args.checkpoint, None)
    if chk_name is None:
        chk_name = "model_weights_%03d.pkl" % int(args.checkpoint)        

    path_to_weights = osp.join(params["out_path"], chk_name)
    if not osp.exists(path_to_weights):
        msg = f"weights not found at {path_to_weights}"
        raise FileNotFoundError(msg)

    # load data
    adata = get_data(params["params"])
    # load gene sets
    gene_sets = get_gene_sets(params["params"], adata=adata)

    # build pathfinder model
    pfnd = get_pfnd_model(params, adata, gene_sets, args)

    # run benchmark based on binary classification of cell identities given TF scores
    run_ident_bench(pfnd=pfnd, adata=adata, args=args)
    # run reprogramming recipe recall benchmark
    run_reprog_bench(pfnd=pfnd, adata=adata, args=args)
    # run mef2neuro benchmark
    run_m2n_bench(model_api=pfnd, adata=adata, args=args)
    return


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    logging.info("INFO statements are active.")
    main()
