import os
import os.path as osp
import time
import json
from collections import defaultdict

import slurm

import numpy as np
import coolname
import itertools
import sys
sys.path += ["/home/jacob/src/jacobkimmel/pathfinder/lib"]
import pathfinder

DATASETS = {
    "tabula_muris_senis": {
        "path": "/group/singlecell/mouse/tabula_muris_senis/joint/joint_both_log1p_cpm.h5ad",
        "groupby": "cell_ontology_class",
        "domain_groupby": "domain",        
    },
    # TODO: Filter out the nan cell types from this object!
    "tabula_muris": {
        "path": "/group/singlecell/mouse/tabula_muris/anndata/tm_joint_log1p_cpm_counts_layer.h5ad",
        "groupby": "cell_ontology_class",
        "domain_groupby": "technology",
    },
    "moca": {
        "path": "/group/singlecell/mouse/seattle_development_atlas/h5ad/log1p_cpm.h5ad",
        "groupby": "Cluster",
        "domain_groupby": "id",
    },    
}
DATE = "SET_ME"
if DATE == "SET_ME":
    raise ValueError("Please set the `DATE` variable in `batch_sparse_scnym.py`")

ROOT_DIR = f"/scratch/jacob/proc_data/pathfinder/{DATE}"
RUN_SCRIPT = "/home/jacob/src/jacobkimmel/pathfinder/lib/bin/run_sparse_scnym.py"

# # Model pre-trained to learn TF latent variables on the Seattle MOCA
# MODEL_CHECKPOINT = "/scratch/jacob/proc_data/pathfinder/20220120/simple-quizzical-zebra/00_best_model_weights.pkl"
# # Genes used for the model checkpoint
# GENE_NAMES = "/scratch/jacob/proc_data/pathfinder/20220120/simple-quizzical-zebra/model_genes.txt"

# default model parameters that work well
camouflaged_enigmatic_robin = {
    "attrprior::attr_prior": "gini_classwise",
    "attrprior::burn_in_epochs": 0,
    "attrprior::max_unsup_weight": 0.1,
    "attrprior::ramp_epochs": 100,
    "ce::burn_in_epochs": 0,
    "ce::max_unsup_weight": 1.0,
    "ce::ramp_epochs": 0,
    "dan::burn_in_epochs": 100,
    "dan::max_unsup_weight": 0.1,
    "dan::ramp_epochs": 20,
    "data::domain_groupby": "technology",
    "data::gene_sets": "/home/jacob/bin/anaconda/envs/guided_lv/lib/python3.8/site-packages/pathfinder/bench/assets/enrichr_tfs_mm.gmt",
    "data::groupby": "cell_ontology_class",
    "data::name": "tabula_muris",
    "data::path": "/group/singlecell/mouse/tabula_muris/anndata/tm_joint_log1p_cpm_counts_layer.h5ad",
    "data::unlabeled_path": "/group/singlecell/mouse/seattle_development_atlas/h5ad/log1p_cpm.h5ad",
    "data::all_grn_genes": False,
    "gsn::burn_in_epochs": 0,
    "gsn::max_unsup_weight": 0.001,
    "gsn::ramp_epochs": 0,
    "latent_var_corr::burn_in_epochs": 0,
    "latent_var_corr::lv_dropout": False,
    "latent_var_corr::max_unsup_weight": 2.0,
    "latent_var_corr::mean_corr_weight": 0.0,
    "latent_var_corr::noise_p": 0.5,
    "latent_var_corr::ramp_epochs": 0,
    "model::hidden_init_dropout": False,
    "optim::batch_size": 512,
    "optim::freeze_lv": False,
    "optim::model_checkpoint": False,
    "optim::n_epochs": 1500,
    "optim::patience": 400,
    "optim::weight_decay": 1e-05,
    "structured_sparse::burn_in_epochs": 200,
    "structured_sparse::max_unsup_weight": 1e-05,
    "structured_sparse::nonnegative": True,
    "structured_sparse::ramp_epochs": 200,
    "structured_sparse::squash_weights": 1,
    "weight_mask::max_unsup_weight": 0,
}

# parameters to scan
params = {
    "latent_var_corr": {
        "ramp_epochs": [0,],
        "burn_in_epochs": [0,],
        "max_unsup_weight": [2.0,],
        "mean_corr_weight": [0., 0.1],
        "lv_dropout": [False,],
        "noise_p": [0.5],
    },
    "dan": {
        "ramp_epochs": [20],
        "burn_in_epochs": [100],
        "max_unsup_weight": [0.1],
    },
    "structured_sparse": {
        "ramp_epochs": [200,],
        "burn_in_epochs": [200,],
        "max_unsup_weight": [1e-6, 1e-5],
        "squash_weights": [1,],
        "nonnegative": [True],
    },
    "gsn": {
        "ramp_epochs": [0,],
        "burn_in_epochs": [0,],
        "max_unsup_weight": [0, 1e-4, 1e-3],
    },
    "ce": {
        "ramp_epochs": [0,],
        "burn_in_epochs": [0,],
        "max_unsup_weight": [1.,],
    },
    "attrprior": {
        "max_unsup_weight": [0., 0.1,],
        "burn_in_epochs": [0, 400],
        "ramp_epochs": [100,],
        "attr_prior": ["gini_classwise"],
    },
    "weight_mask" : {
        # we only need to decide to use it or not
        "max_unsup_weight": [0,],
    },
    "data": {
        **DATASETS["tabula_muris"],
        "name": "tabula_muris",
        "gene_sets": pathfinder.bench.identity_bench.IDENT_DATASETS["gene_sets"]["mouse"]["enrichr_tfs"],
        # "gene_names": GENE_NAMES, # reshape data to use these gene names
    },
    "optim": {
        "weight_decay": [1e-5],
        "n_epochs": [1500],
        "patience": [700],
        "model_checkpoint": [False,],
        "freeze_lv": [False,],
        "batch_size": [256, 512, 1024],
    },
    "model" : {
        "hidden_init_dropout": [False,],
    },
}

params["data"] = {k:[v] for k,v in params["data"].items()}
# add unlabeled data later so we can toggle it on and off
params["data"]["unlabeled_path"] = [
    # False,
    DATASETS["moca"]["path"], # path to unlabeled anndata
]
params["data"]["all_grn_genes"] = [False, True]

def collapse_dicts(d: dict):
    # input is {k0: {k1: [v0, ..., vN]}}
    nd = {}
    for k0 in d.keys():
        for k1 in d[k0].keys():
            nd[k0+"::"+k1] = d[k0][k1]
    return nd


def join_dicts_by_keys(dicts: list):
    new_dict = defaultdict(list)
    for d in dicts:
        for k, v in dicts.items():
            new_dict[k].append(v)
    return new_dict


def get_value_combinations(params: dict):
    # [[(key, val)]_key1, ..., [(key, val)_keyN]]
    key_vals = [[(key, x) if type(val)==list else (key, x) for x in val ] for key, val in params.items()]
    combinations = itertools.product(*key_vals)
    # one key to one value for each parameter
    combinations_dicts = [
        {x[0]:x[1] for x in y} for y in combinations
    ]
    return combinations_dicts


def get_exp_diff(exp0, exp1, diffs=[]):
    for k in exp0.keys():
        v0 = exp0[k]
        v1 = exp1.get(k, None)
        if type(v0)==dict:
            get_exp_diff(v0, v1, diffs=diffs)
        else:
            if v0 == v1:
                continue
            else:
                diffs.append((k, v0, v1))
    return diffs


def count_diffs(d, ref, max_diffs=1):
    diffs = get_exp_diff(exp0=d, exp1=ref, diffs=[])
    # remove some params we allow to change freely
    free_params = ["optim::patience"]
    diffs = [x for x in diffs if x[0] not in free_params]

    n_diff = len(diffs)
    if n_diff > max_diffs:
        print()
        print(diffs)
        print()
    return n_diff


def check_diffs(d, ref, max_diffs=1):
    """Check that the number of differences from a reference experiment is below
    `max_diffs`. This is useful for e.g. running all experiments that shift only a single
    parameter at a time relative to a working set of parameters.
    """    
    n_diff = count_diffs(d, ref, max_diffs)
    return (n_diff <= max_diffs)


def check_sparsity(d):
    # check only ss or wm is enabled
    a = d["structured_sparse::max_unsup_weight"] > 0
    b = d["weight_mask::max_unsup_weight"] > 0
    check_0 = (not (a and b))
    # ensure we only run one set of structured sparse parameters when they have 0 weight
    check_1 = (
        (b==0) 
        or 
        (
            (b>0) 
            and
            d["structured_sparse::burn_in_epochs"]==0. 
            and 
            d["structured_sparse::ramp_epochs"]==20.
        )
    )
    return (check_0 and check_1)


def check_ss_before_ce(d):
    a = d["ce::burn_in_epochs"]
    b = d["structured_sparse::burn_in_epochs"]
    return (b < a) or (a==0)


def check_dan_with_ce(d):
    a = d["ce::burn_in_epochs"]
    b = d["dan::burn_in_epochs"]
    return a == b


def check_dan_after_ss(d):
    a = d["dan::burn_in_epochs"]
    b = d["structured_sparse::burn_in_epochs"]
    return a >= b


def check_ce_burn_and_ramp(d):
    """Remove all burn in weights without a ramp value"""
    a = d["ce::burn_in_epochs"]
    b = d["ce::ramp_epochs"]
    return ((a==0) or ((a>0) and (b>0)))


def check_ss_redundancy(d):
    """Ensure we don't scan multiple parameter sets when SS is deactivated"""
    a = d["structured_sparse::max_unsup_weight"]
    b = d["structured_sparse::burn_in_epochs"]
    c = d["structured_sparse::ramp_epochs"]
    nn = d["structured_sparse::nonnegative"]
    return ((a>0) or (b==200 and c==200 and (not nn)))


def check_dan_redundancy(d):
    """Ensure we don't scan multiple parameter sets when DAN is deactivated"""
    a = d["dan::max_unsup_weight"]
    b = d["dan::burn_in_epochs"]
    c = d["dan::ramp_epochs"]
    return ((a>0) or (b==0 and c==20))


def check_lvg_redundancy(d):
    """Ensure we don't scan multiple parameter sets when LVG is deactivated"""
    a = d["latent_var_corr::max_unsup_weight"]
    b = d["latent_var_corr::burn_in_epochs"]
    c = d["latent_var_corr::ramp_epochs"]
    x0 = d["latent_var_corr::mean_corr_weight"]
    return ((a>0) or (b==0 and c==0 and x0==0.0))


def check_lvg_ramp_and_burn(d):
    """Only ramp the LVG if it has a burn-in"""
    # ramp iff there's a burn in
    a = d["latent_var_corr::burn_in_epochs"]
    b = d["latent_var_corr::ramp_epochs"]
    check = ((a == 0)&(b==0)) | ((a > 0) & (b > 0))
    return check


def check_lvg_and_dan(d):
    """Only try different DAN settings if LVG is active"""
    a = d["dan::burn_in_epochs"]    
    b = d["latent_var_corr::max_unsup_weight"]
    return (b > 0) | ((b==0) & (a==0))


def check_lvg_and_ss(d):
    """Only try different SS settings if LVG is active"""
    a = d["latent_var_corr::max_unsup_weight"]    
    b = d["structured_sparse::max_unsup_weight"]
    c = d["structured_sparse::ramp_epochs"]
    e = d["structured_sparse::burn_in_epochs"]
    return (a > 0) | ((b==0) & (c==20) & (e==0))


def check_pretrain_redundancy(d):
    """Don't freeze latent variables if we're not loading from a checkpoint"""
    a = d["optim::model_checkpoint"]
    b = d["optim::freeze_lv"]
    return ((a == False) and ~b) or (type(a)==str)

def check_ap_redundnacy(d):
    """Don't scan multiple attrprior configurations if the weight is off"""
    a = d["attrprior::max_unsup_weight"]
    b = d["attrprior::burn_in_epochs"]
    c = d["attrprior::ramp_epochs"]
    return (a or ((b==0)&(c==100)))


def filter_combinations(combinations: list):
    # each value in combinations is a dict with keys toplevel::lowerlevel

    # toss any examples that have both weight masking and structured sparsity
    filtered = []
    for i in range(len(combinations)):
        if (
            check_sparsity(combinations[i]) 
            and
            check_ce_burn_and_ramp(combinations[i])
            and
            check_dan_redundancy(combinations[i])
            and
            check_lvg_ramp_and_burn(combinations[i])
            and
            check_lvg_redundancy(combinations[i])
            and
            check_ss_redundancy(combinations[i])
            and
            check_lvg_and_dan(combinations[i])
            and
            check_lvg_and_ss(combinations[i])
            and
            check_pretrain_redundancy(combinations[i])
            and
            check_ap_redundnacy(combinations[i])
            and
            check_diffs(combinations[i], ref=camouflaged_enigmatic_robin, max_diffs=1)
        ):
            filtered.append(combinations[i])
    return filtered


def make_job(cmd: str, i_name: str, out_path: str):
    job = slurm.Job(
        cmd=cmd,
        name=i_name,
        gpu=1,
        queue="gpu",
        mem=64000,
        cpu=4,
        time="96:00:00",
        out_file=osp.join(out_path, "run.out"),
        err_file=osp.join(out_path, "run.err"),
    )
    return job


def main(launch: bool=False):
    """
    Parameters
    ----------
    launch : bool
        if True, creates directories and launches jobs. otherwise prints to stdout.
    """
    # get combinations
    with open("./batch_sweep_params.json", "w") as f:
        json.dump(params, f)
    cparams = collapse_dicts(params)
    combinations = get_value_combinations(cparams)
    # List[dict] of parameter combinations
    filtered = filter_combinations(combinations)
    print(f"{len(combinations)} raw parameters combinations.")
    print(f"{len(filtered)} combinations remain after filtering.")
    # for each set of parameters, create an output directory with a fun name
    os.makedirs(ROOT_DIR, exist_ok=True)

    experiment_log = {}
    jobs = []
    for i in range(len(filtered)):
        i_name = coolname.generate_slug(3)
        if count_diffs(filtered[i], ref=camouflaged_enigmatic_robin, max_diffs=1) == 0:
            i_name = "camouflaged-enigmatic-robin"

        p = filtered[i]
        experiment_log[i_name] = {}
        experiment_log[i_name]["exp_number"] = i
        experiment_log[i_name]["time"] = time.strftime("%H-%M-%S")
        experiment_log[i_name]["date"] = DATE
        experiment_log[i_name]["params"] = p
        out_path = osp.join(ROOT_DIR, i_name)
        experiment_log[i_name]["out_path"] = out_path

        if launch:
            os.makedirs(out_path, exist_ok=True)
        # generate command
        path_to_params = osp.join(experiment_log[i_name]["out_path"], "params.json")
        cmd = f"python {RUN_SCRIPT} {path_to_params} {out_path}"
        experiment_log[i_name]["cmd"] = cmd
        # write parameters
        if launch:
            with open(path_to_params, "w") as f:
                json.dump(p, f)
            print(f"command {i}, {i_name}\n\t{cmd}")
        else:
            print(f"command {i}, {i_name}\n\t{cmd}")

        job = make_job(cmd=cmd, i_name=i_name, out_path=out_path)
        jobs.append(job)

    # save experiments file
    with open("./experiment.json", "w") as f:
        s = json.dumps(experiment_log, indent=4, sort_keys=True,)
        f.writelines(s)

    if launch:
        slurm.multi_run(jobs, max_proc=32, verbose=True)
        print("To the moon!")
    return


if __name__ == "__main__":
    main(launch=True)
