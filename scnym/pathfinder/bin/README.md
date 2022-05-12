# Pathfinder command line tools

1. `batch_bench.py` : Launch batch jobs for benchmarking on `slurm`. Takes one positional argument `exp_manifest`, a path to a JSON experiment file that is exported by `batch_sparse_scnym.py`.
2. `batch_sparse_scnym.py` : Creates a series of batch jobs to test different parameter sets for Pathfinder. By default, runs an ablation experiment training a full Pathfinder model as described in the paper along with a set of ablated versions. No CLI arguments. Has one hardcoded parameter `DATE` to specify the output path and will throw a helpful error if you forget to set it before running. Creates a JSON file describing all the model parameters for launched jobs: `experiment.json`.
3. `bench_baselines.py` : Run benchmarks from the Pathfinder manuscript for baseline models. PySCENIC is expensive and will require 100GB+ of RAM as specified. Takes two positional CLI arguments: (1) an `experiment.json` file from `batch_sparse_scnym.py`, (2) the name of an experiment in `experiment.json` to use for benchmarking. Also takes an optional `--species` argument for PySCENIC, defaults to mouse.
4. `bench_sparse_scnym.py`: Run benchmarks for a Pathfinder model. Takes two positional CLI arguments: (1) an `experiment.json` file from `batch_sparse_scnym.py`, (2) the name of an experiment in `experiment.json` to use for benchmarking.
5. `experiment2summary.py` : Takes an `experiment.json` file and generates a summary table of training statistics from the Tensorboard outputs. Used at the beginning of Pathfinder development to find a parameter regime that didn't impact classification performance dramatically.
6. `run_sparse_scnym.py`: Experiment level training script. Takes a set of Pathfinder parameters in a JSON file (automatically generated by `batch_sparse_scnym.py`) and trains a Pathfinder model.
7. `default_params.json`: Default parameters for the `run_sparse_scnym.py` script, as described in the manuscript. Requires users to set the proper paths for the `data::path`, `data::unlabeled_path`, and `data::gene_sets` parameters. Current paths are set for the Tabula Muris+MOCA data on the Calico local cluster.