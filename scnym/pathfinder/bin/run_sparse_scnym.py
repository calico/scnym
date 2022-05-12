import json
from os import path as osp
import pickle

import torch
import numpy as np
import pandas as pd
import scnym
import anndata


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
        unlab = _cast_adata_with_genes(unlab, gene_names) if unlab is not None else None
    elif unlab is not None and not params.get("data::all_grn_genes", False):
        # get shared gene names
        print("Subsetting labeled and unlabeled data to shared genes")
        gene_names = [x for x in adata.var_names if x in unlab.var_names]
        adata = adata[:, gene_names].copy()
        unlab = unlab[:, gene_names].copy() 
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
        unlab = _cast_adata_with_genes(unlab, gene_names)        
    else:
        print("No unlabeled data was loaded.")
    return adata, unlab


def get_optimizer(params, model):
    if params.get("optim::freeze_lv", False):
        print("Freezing initial LV parameters")
        # turn off gradients to the latent variable weights
        lv_emb = dict(model.named_modules())["embed.0"]
        for p in lv_emb.parameters():
            p.requires_grad = False
    params2opt = list(filter(lambda p: p.requires_grad, model.parameters()))
    # NOTE: if we don't typecast to list first, the `len` below will empty the generator
    # and leave no elements in `params2opt` for the optimizer to use!
    n_p = len(params2opt)
    print(f"Optimizing {n_p} parameters.")
    optimizer = torch.optim.AdamW(
        params2opt,
        weight_decay=params["optim::weight_decay"],
    )
    return optimizer


def get_batch_transformers():
    b_trans = scnym.dataprep.SampleMixUp(alpha=0.3, keep_dominant_obs=True)
    batch_transformers = {"train": b_trans}
    return batch_transformers


def get_dataloaders(params: dict, adata: anndata.AnnData, unlab: anndata.AnnData=None):
    y_label = adata.obs[params["data::groupby"]]
    y = np.array(
        pd.Categorical(y_label, categories=np.unique(y_label)).codes, dtype=np.int32
    )
    domain_label = adata.obs[params["data::domain_groupby"]]
    domain = np.array(
        pd.Categorical(domain_label, categories=np.unique(domain_label)).codes,
        dtype=np.int32,
    )

    train_idx = np.random.choice(
        adata.shape[0], size=int(0.8 * adata.shape[0]), replace=False
    )
    valtest_idx = np.setdiff1d(np.arange(adata.shape[0]), train_idx)
    test_idx = np.random.choice(
        np.arange(len(valtest_idx)), size=int(0.5 * len(valtest_idx)), replace=False
    )
    val_idx = np.setdiff1d(valtest_idx, test_idx)

    indices = {"train": train_idx, "val": val_idx, "test": test_idx}

    train_ds = scnym.dataprep.SingleCellDS(
        X=adata.X[train_idx],
        y=y[train_idx],
        domain=domain[train_idx],
        num_classes=len(np.unique(y)),
        num_domains=len(np.unique(domain)),
    )

    val_ds = scnym.dataprep.SingleCellDS(
        X=adata.X[val_idx],
        y=y[val_idx],
        domain=domain[val_idx],
        num_classes=len(np.unique(y)),
        num_domains=len(np.unique(domain)),
    )

    test_ds = scnym.dataprep.SingleCellDS(
        X=adata.X[test_idx],
        y=y[test_idx],
        domain=domain[test_idx],
        num_classes=len(np.unique(y)),
        num_domains=len(np.unique(domain)),
    )

    batch_size = params.get("optim::batch_size", 256)
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=False, drop_last=True
    )
    val_dl = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False
    )
    test_dl = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, drop_last=False,
    )

    dataloaders = {"train": train_dl, "val": val_dl, "test": test_dl}
    print("# train: ", len(train_ds))
    print("# val  : ", len(val_ds))

    if unlab is not None:
        # build an unlabeled dataloader
        unlab_ds = scnym.dataprep.SingleCellDS(
            X=unlab.X,
            y=np.zeros(unlab.shape[0]),
            domain=None,
        )
        unlab_dl = torch.utils.data.DataLoader(
            unlab_ds, batch_size=batch_size, shuffle=True, drop_last=True,
        )
        dataloaders["unlabeled"] = unlab_dl

    return dataloaders, indices


def get_model(params, adata, gene_sets):
    model = scnym.model.CellTypeCLF(
        n_genes=adata.shape[1],
        n_hidden=256,
        n_hidden_init=len(gene_sets),
        hidden_init_dropout=params.get("model::hidden_init_dropout", False),
        n_cell_types=len(np.unique(adata.obs[params["data::groupby"]])),
    )
    # load a checkpoint if provided
    if params.get("optim::model_checkpoint", False):
        print("Loading pre-trained model checkpoint...")
        prev_state_dict = torch.load(
            params["optim::model_checkpoint"], map_location="cpu"
        )
        # assume we only want the first LV layer
        prev_state_dict = {k:v for k, v in prev_state_dict.items() if "embed.0" in k}
        # by updating and reloading the current dict, we only update a subset of the 
        # parameters
        model_dict = model.state_dict()
        model_dict.update(prev_state_dict)
        model.load_state_dict(model_dict)
        
    return model


def get_ss(params, model, adata, gene_sets):
    ss = scnym.losses.StructuredSparsity(
        n_genes=model.n_genes,
        n_hidden=model.n_hidden_init,
        gene_names=adata.var_names.tolist(),
        gene_sets=gene_sets,
    )
    if params["structured_sparse::squash_weights"] > 0:
        ss.init_model_params(model, scale=1e-3)

    ss_weight = scnym.losses.ICLWeight(
        ramp_epochs=params["structured_sparse::ramp_epochs"],
        burn_in_epochs=params["structured_sparse::burn_in_epochs"],
        max_unsup_weight=params["structured_sparse::max_unsup_weight"],
        sigmoid=True,
    )

    return ss, ss_weight


def get_gsn(params, model, adata, gene_sets):
    gsn = scnym.losses.WithinGeneSetNorm(
        gene_sets=gene_sets,
        gene_names=np.array(adata.var_names),
    )
    gsn_weight = scnym.losses.ICLWeight(
        ramp_epochs=params["gsn::ramp_epochs"],
        burn_in_epochs=params["gsn::burn_in_epochs"],
        max_unsup_weight=params["gsn::max_unsup_weight"],
        sigmoid=True,
    )
    return gsn, gsn_weight


def get_wm(model, adata, gene_sets):
    wm = scnym.losses.WeightMask(
        model=model,
        gene_sets=gene_sets,
        gene_names=np.array(adata.var_names),
        nonnegative=True,
    )
    return wm, 0.0


def get_ce(params, **kwargs):
    ce = scnym.losses.scNymCrossEntropy()
    ce_weight = scnym.losses.ICLWeight(
        ramp_epochs=params["ce::ramp_epochs"],
        burn_in_epochs=params["ce::burn_in_epochs"],
        max_unsup_weight=params["ce::max_unsup_weight"],
        sigmoid=True,
    )
    return ce, ce_weight


def get_dan(params, model, adata):
    dan = scnym.losses.DANLoss(
        dan_criterion=scnym.losses.cross_entropy,
        model=model,
        n_domains=len(np.unique(adata.obs[params["data::domain_groupby"]])),
    )
    dan_weight = scnym.losses.ICLWeight(
        ramp_epochs=params["dan::ramp_epochs"],
        burn_in_epochs=params["dan::burn_in_epochs"],
        max_unsup_weight=params["dan::max_unsup_weight"],
        sigmoid=True,
    )
    return dan, dan_weight


def get_lvg_corr(params, adata, gene_sets):
    lgc = scnym.losses.LatentGeneCorrGuide(
        gene_names=adata.var_names.tolist(),
        latent_var_genes=sorted(list(gene_sets.keys())),
        criterion="pearson",
        mean_corr_weight=params["latent_var_corr::mean_corr_weight"],
        lv_dropout=params.get("latent_var_corr::lv_dropout", False),
        noise_p=params.get("latent_var_corr::noise_p", 0.5),
    )
    return lgc, params["latent_var_corr::max_unsup_weight"]


def get_ap(params, train_ds):
    ap = scnym.losses.AttrPrior(
        reference_dataset=train_ds,
        batch_size=params.get("optim::batch_size", 256),
        attr_prior=params.get("attrprior::attr_prior", "gini_classwise"),
        grad_activation="first_layer",
    )
    ap_weight = scnym.losses.ICLWeight(
        ramp_epochs=params["attrprior::ramp_epochs"],
        burn_in_epochs=params["attrprior::burn_in_epochs"],
        max_unsup_weight=params["attrprior::max_unsup_weight"],
        sigmoid=True,
    )    
    return ap, ap_weight


def get_criteria(params, model, adata, gene_sets, dataloaders):

    criteria = []
    # cross entropy
    ce, ce_weight = get_ce(params)
    criteria.append(
        {"name": "ce", "function": ce, "weight": ce_weight, "validation": True},
    )
    # sparsity
    if params["structured_sparse::max_unsup_weight"] > 0:
        ss, ss_weight = get_ss(params, model, adata, gene_sets)
        criteria.append(
            {"name": "ss", "function": ss, "weight": ss_weight, "validation": False}
        )
    # weight mask
    if params["weight_mask::max_unsup_weight"] > 0:
        wm, wm_weight = get_wm(model, adata, gene_sets)
        criteria.append(
            {"name": "wm", "function": wm, "weight": wm_weight, "validation": False},
        )
    # adversary
    if params["dan::max_unsup_weight"] > 0:
        dan, dan_weight = get_dan(params=params, model=model, adata=adata)
        criteria.append(
            {"name":"dan", "function": dan, "weight": dan_weight, "validation": False}
        )
    # attribution prior
    if params["attrprior::max_unsup_weight"] > 0:
        ap, ap_weight = get_ap(params=params, train_ds=dataloaders["train"].dataset)
        criteria.append(
            {"name":"ap", "function": ap, "weight": ap_weight, "validation": False,},
        )
    # gene set norm
    gsn, gsn_weight = get_gsn(params, model, adata, gene_sets)
    criteria.append(
        {"name":"gsn", "function": gsn, "weight": gsn_weight, "validation": False},
    )
    # latent var corr
    lvg, lvg_weight = get_lvg_corr(params, adata, gene_sets)
    criteria.append(
        {"name":"lvg_corr", "function": lvg, "weight": lvg_weight, "validation": False},
    )
    return criteria


def get_min_epochs(params):
    min_epochs = max(
        (
            params["ce::burn_in_epochs"]+params["ce::ramp_epochs"],
            params.get("structured_sparsity::burn_in_epochs", 0.0)+params.get("structured_sparsity::ramp_epochs", 0.0),
            params.get("dan::burn_in_epochs", 0.0),
            params.get("latent_var_corr::burn_in_epochs", 0.0)+params.get("latent_var_corr::ramp_epochs", 0.0),
        )
    )
    return min_epochs


def get_trainer(
    *, 
    params, 
    model, 
    dataloaders, 
    optimizer, 
    criteria, 
    batch_transformers, 
    out_path,
):

    min_epochs = get_min_epochs(params)
    trainer = scnym.trainer.MultiTaskTrainer(
        model=model,
        dataloaders=dataloaders,
        unsup_dataloader=dataloaders.get("unlabeled", None),
        optimizer=optimizer,
        criteria=criteria,
        batch_transformers=batch_transformers,
        out_path = out_path,
        n_epochs=params["optim::n_epochs"],
        min_epochs=min_epochs,
        patience=params["optim::patience"],
        use_gpu=True,
        tb_writer=osp.join(out_path, "tblog"),
        save_freq=params.get("optim::save_freq", 200),
    )
    return trainer


def build_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("params", type=str, help="path to parameters")
    parser.add_argument("out_path", type=str, help="path for outputs")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # load params
    with open(args.params, "r") as f:
        params = json.load(f)

    print(json.dumps(params, indent = 4))
    
    # load data
    adata, unlab = get_data(params)
    # make datasets
    dataloaders, indices = get_dataloaders(params=params, adata=adata, unlab=unlab)
    print("Built dataloaders, saving indices.")
    with open(osp.join(args.out_path, "data_split_indices.pkl"), "wb") as f:
        pickle.dump(indices, f)

    print("Loading gene sets.")
    gene_sets = get_gene_sets(params=params, adata=adata)
    print("Setting up model.")
    model = get_model(params=params, adata=adata, gene_sets=gene_sets)
    if torch.cuda.is_available():
        model = model.cuda()
    # setup the training criteria
    criteria = get_criteria(
        params=params, 
        model=model, 
        adata=adata, 
        gene_sets=gene_sets, 
        dataloaders=dataloaders,
    )
    # setup the optimizer
    optimizer = get_optimizer(params=params, model=model)
    # add any non-model parameters that are required
    for c in criteria:
        if c.get("name", "not_dan") == "dan":
            fxn = c["function"]
            print("adding dan parameters", fxn)
            for adv in fxn.dann:
                # dann is a nn.ModuleList of adversaries
                optimizer.add_param_group(
                    {"params": adv.domain_clf.parameters()}
                )
    # get batch transformers
    batch_transformers = get_batch_transformers()
    # setup the trainer
    trainer = get_trainer(
        params=params,
        model=model,
        dataloaders=dataloaders,
        criteria=criteria,
        optimizer=optimizer,
        batch_transformers=batch_transformers,
        out_path=args.out_path,
    )
    # train!
    trainer.train()

    print("Fitting complete.")
    return


if __name__ == "__main__":
    seed = 397
    np.random.seed(seed)
    torch.manual_seed(seed)
    main()
