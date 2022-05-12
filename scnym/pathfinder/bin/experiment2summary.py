"""Extract training logs for all experiments in an `experiment.json` manifest"""
import pandas as pd
import numpy as np
import os
import os.path as osp
import glob
import json
import argparse
import traceback

import tqdm
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# keys from the experiment description to pair with the training values
KEYS2EXTRACT = ()

def tflog2pandas(path: str) -> pd.DataFrame:
    """Convert a single tensorflow log file to pandas DataFrame

    Parameters
    ----------
    path : str
        path to tensorflow log file directory

    Returns
    -------
    pd.DataFrame
        converted dataframe

    References
    ----------
    https://bit.ly/31o4fz4
    """
    DEFAULT_SIZE_GUIDANCE = {
        "compressedHistograms": 1,
        "images": 1,
        "scalars": 0,  # 0 means load all
        "histograms": 1,
    }
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    try:
        event_acc = EventAccumulator(path, DEFAULT_SIZE_GUIDANCE)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    return runlog_data


def summ_log(df, params: dict, summ_mode: str="min_val_loss"):
    """Summarize a training run and pair it with experiment parameters"""

    min_epochs = max(
        (
            params["ce::burn_in_epochs"]+params["ce::ramp_epochs"],
            params.get("structured_sparsity::burn_in_epochs", 0.0),
            params.get("dan::burn_in_epochs", 0.0),
        )
    )

    # find summary epoch
    if summ_mode == "min_val_loss":
        val_loss = (
            df.loc[df["metric"]=="Loss/val", ["value", "step"]]
            .sort_values("step")
        ).copy()
        # mask out any values before we initialized the val loss
        val_loss.loc[val_loss["step"]<min_epochs, "value"] = 1e6
        idx = np.argmin(val_loss["value"])
        step = int(val_loss.iloc[idx]["step"])
    else:
        raise NotImplementedError()
    
    # extract summary epoch
    df_summ = df.loc[df["step"]==step].copy()
    df_tab = df_summ.pivot_table(index="step", columns="metric", values="value")
    for p in params:
        df_tab.loc[df_tab.index[0], p] = params[p]
    return df_tab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "exp_manifest", type=str, help="path to experiment manifest JSON"
    )
    parser.add_argument("out_path", type=str, help="path to CSV output file")
    parser.add_argument(
        "--summ_mode", 
        type=str,
        default="min_val_loss",
        help="method for experiment summarization. default: [%default]"
    )
    args = parser.parse_args()
    summ_mode = args.summ_mode

    with open(args.exp_manifest, "r") as f:
        # keys are exp names, paths are values (manifest[exp_name][out_path] + "/tblog")
        manifest = json.load(f)

    logs = []
    for exp in tqdm.tqdm(manifest.keys()):
        # load params that are already only one layer
        path_to_params = osp.join(manifest[exp]["out_path"], "params.json")
        with open(path_to_params, "r") as f:
            params = json.load(f)

        df = tflog2pandas(path=osp.join(manifest[exp]["out_path"], "tblog"))
        df_summ = summ_log(df, params=params, summ_mode=summ_mode)
        df_summ["experiment"] = exp

        logs.append(df_summ)
    
    exp_df = pd.concat(logs, axis=0)
    exp_df.to_csv(args.out_path)
    return


if __name__ == "__main__":
    main()
