import json
import os.path as osp
import pandas as pd
import numpy as np
import argparse

import sys
sys.path = ["/home/jacob/src/scutil"] + sys.path
import slurm


BENCH_SCNYM = "/home/jacob/src/jacobkimmel/pathfinder/lib/bin/bench_sparse_scnym.py"
BENCH_BASE = "/home/jacob/src/jacobkimmel/pathfinder/lib/bin/bench_baselines.py"


def make_scnym_job(exp_name, manifest, args):

    cmd_args = [
        "python",
        BENCH_SCNYM,
        args.exp_manifest,
        exp_name,
    ]
    print(" ".join(cmd_args))
    job = slurm.Job(
        cmd=" ".join(cmd_args),
        name="sbench_"+exp_name,
        cpu=1,
        gpu=1,
        queue="gpu",
        mem=64000,
        time="72:00:00",
        out_file=osp.join(manifest[exp_name]["out_path"], "scnym_bench.out"),
        err_file=osp.join(manifest[exp_name]["out_path"], "scnym_bench.err"),
    )
    return job


def make_base_job(exp_name, manifest, args):

    cmd_args = [
        "python",
        BENCH_BASE,
        args.exp_manifest,
        exp_name,
    ]
    print(" ".join(cmd_args))
    job = slurm.Job(
        cmd=" ".join(cmd_args),
        name="bbench_"+exp_name,
        cpu=4,
        gpu=0,
        queue="standard",
        mem=128000,
        time="128:00:00",
        out_file=osp.join(manifest[exp_name]["out_path"], "base_bench.out"),
        err_file=osp.join(manifest[exp_name]["out_path"], "base_bench.err"),        
    )
    return job



def build_parser():
    parser = argparse.ArgumentParser(
        description="Benchmark a set of scNym weights trained on a mouse atlas."
    )
    parser.add_argument(
        "exp_manifest", 
        type=str, 
        help="path to an experiment manifest file.",
    )    
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    # load manifest
    with open(args.exp_manifest, "r") as f:
        manifest = json.load(f)
    
    jobs = []

    keys = list(manifest.keys())
    # tmp TODO
    tmp = [
        x for x in keys if osp.exists(osp.join(manifest[x]["out_path"], "01_final_model_weights.pkl"))
    ]
    print(f"Benchmarking {len(tmp)} models.")
    keys = [x for x in keys if x in tmp]
    for i, exp_name in enumerate(keys):
        scnym_job = make_scnym_job(exp_name, manifest, args)
        # tmp TODO
        jobs.append(scnym_job)
        # add base job only for the first model
        if i == 0:
            base_job = make_base_job(exp_name, manifest, args)
            # jobs.append(base_job)
    
    slurm.multi_run(jobs, verbose=True)
    return


if __name__ == "__main__":
    main()
