"""Test `MultiTaskTrainer` and associated loss functions"""

import pytest
import os
import os.path as osp
import sys
import numpy as np
import torch

sys.path.append("../")


def test_multitask_mixmatch():
    """Test the multitask MixMatchWrapper"""
    import scanpy as sc
    import scnym
    from scnym.dataprep import SingleCellDS

    torch.manual_seed(1)
    np.random.seed(1)

    # load 10x human PBMC data as a sample
    adata = sc.datasets.pbmc3k()
    sc.pp.filter_cells(adata, min_counts=100)
    sc.pp.filter_genes(adata, min_cells=100)
    sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e6)
    sc.pp.log1p(adata)

    # generate fake class labels
    adata.obs["class"] = np.random.randint(
        0,
        3,
        adata.shape[0],
    )

    # create dataloaders
    d_ds = SingleCellDS(
        X=adata.X.toarray(),
        y=np.array(adata.obs["class"]),
    )
    u_ds = SingleCellDS(
        X=adata.X.toarray(),
        y=np.zeros(adata.shape[0]),
    )

    d_dl = torch.utils.data.DataLoader(d_ds, batch_size=16)
    u_dl = torch.utils.data.DataLoader(u_ds, batch_size=16)

    # draw a batch from each
    d_batch = next(iter(d_dl))
    u_batch = next(iter(u_dl))

    # initialize mixmatch using a dummy model
    unsup_criterion = torch.nn.MSELoss(reduction="none")
    sup_criterion = scnym.trainer.cross_entropy

    augment = scnym.dataprep.AUGMENTATION_SCHEMES["log1p_drop"]

    model = scnym.model.CellTypeCLF(
        n_layers=1,
        n_hidden=128,
        residual=False,
        init_dropout=0.0,
        n_genes=adata.shape[1],
        n_cell_types=3,
    )

    K = 2
    mixmatch = scnym.losses.MixMatchLoss(
        unsup_criterion=unsup_criterion,
        sup_criterion=sup_criterion,
        augment=augment,
        alpha=0.3,
        n_augmentations=K,
    )

    # wrap the mixmatch loss for `MultiTaskTrainer`
    wrapped = scnym.losses.MultiTaskMixMatchWrapper(
        mixmatch_loss=mixmatch,
        sup_weight=1.0,
        unsup_weight=0.1,
    )

    # forward pass through the wrapped loss
    loss = wrapped(
        labeled_sample=d_batch,
        unlabeled_sample=u_batch,
        model=model,
        weight=1000.0,  # dummy value to confirm API compat.
    )
    msg = "loss was a tuple. this should only output one value."
    assert type(loss) != tuple, msg

    # test operations on a CUDA device if it's available
    # in the testing environment
    if torch.cuda.is_available():
        model = model.cuda()
        u_batch["input"] = u_batch["input"].cuda()
        d_batch["input"] = d_batch["input"].cuda()
        d_batch["output"] = d_batch["output"].cuda()

        # remove the old teacher because the CPU parameters
        # can't be updated with the CUDA parameters
        wrapped.mixmatch_loss.teacher = None

        loss = wrapped(
            model=model,
            unlabeled_sample=u_batch,
            labeled_sample=d_batch,
        )

    # test that the wrapped loss is compatible with the
    # MultiTask API
    wrapped.train(True)
    wrapped.train(False)
    wrapped.eval()
    assert hasattr(wrapped, "epoch")
    return


def test_multitask_dan():
    """
    # TODO finish
    if True:
        return
    import scanpy as sc
    import scnym
    from scnym.dataprep import SingleCellDS
    torch.manual_seed(1)
    np.random.seed(1)

    # load 10x human PBMC data as a sample
    adata = sc.datasets.pbmc3k()
    sc.pp.filter_cells(adata, min_counts=100)
    sc.pp.filter_genes(adata, min_cells=100)
    sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e6)
    sc.pp.log1p(adata)

    # create dataloaders
    d_ds = SingleCellDS(
        X = adata.X.toarray(),
        y = np.array(adata.obs['class']),
    )
    u_ds = SingleCellDS(
        X = adata.X.toarray(),
        y = np.zeros(adata.shape[0]),
    )

    d_dl = torch.utils.data.DataLoader(d_ds, batch_size=16)
    u_dl = torch.utils.data.DataLoader(u_ds, batch_size=16)

    dataloaders = {"train": d_dl, "val": d_dl}

    ce_crit = scnym.losses.scNymCrossEntropy()
    da_crit = scnym.losses.DANLoss(
        model=model,
    )

    criteria = [
        {
            'function': ce_crit,
            'weight': 1.,
            'validation': True,
            'name': 'ce',
        },
        {
            'function': da_crit,
            'weight': 0.1,
            'validation': False,
            'name': 'domain_adv',
        }
    ]
    """
    return


def test_multitask_trainer():
    """Test a multitask model performing supervised classification
    and unsupervised reconstruction
    """
    import scnym
    import scanpy as sc
    import pandas as pd

    torch.manual_seed(1)
    np.random.seed(1)

    #####################################
    # Generate sample data loaders
    #####################################

    # load 10x human PBMC data as a sample
    adata = sc.datasets.pbmc3k()
    # seurat flavor operates on counts
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat_v3")
    sc.pp.pca(adata)
    sc.pp.neighbors(adata, n_neighbors=15)
    # generate clusters to use as class labels
    sc.tl.leiden(adata, resolution=0.5, key_added="leiden")
    adata.obs["class"] = pd.Categorical(
        adata.obs["leiden"],
    ).codes

    X = adata.X
    y = np.array(adata.obs["class"])

    ds = scnym.dataprep.SingleCellDS(
        X=X,
        y=y,
    )

    l_dl = torch.utils.data.DataLoader(ds, batch_size=256)
    t_dl = torch.utils.data.DataLoader(ds, batch_size=256)
    u_dl = torch.utils.data.DataLoader(ds, batch_size=256)

    dataloaders = {
        "train": l_dl,
        "val": t_dl,
    }

    #####################################
    # Setup the model
    #####################################

    model = scnym.model.CellTypeCLF(
        n_genes=X.shape[1],
        n_cell_types=len(np.unique(y)),
        n_hidden=128,
        n_layers=1,
        init_dropout=0.1,
        use_raw_counts=True,
    )
    if torch.cuda.is_available():
        model = model.cuda()

    #####################################
    # Setup multi-task criteria
    #####################################
    unsup_criterion = torch.nn.MSELoss(reduction="none")
    sup_criterion = scnym.trainer.cross_entropy

    augment = scnym.dataprep.AUGMENTATION_SCHEMES["count_poisson"]
    mixmatch_loss = scnym.losses.MixMatchLoss(
        unsup_criterion=unsup_criterion,
        sup_criterion=sup_criterion,
        augment=augment,
        alpha=0.3,
        n_augmentations=2,
    )

    mm_crit = scnym.losses.MultiTaskMixMatchWrapper(
        mixmatch_loss=mixmatch_loss,
        sup_weight=1.0,
        unsup_weight=scnym.losses.ICLWeight(
            ramp_epochs=100,
            max_unsup_weight=1.0,
        ),
    )

    ce_crit = scnym.losses.scNymCrossEntropy()

    rec_crit = scnym.losses.ReconstructionLoss(
        rec_criterion=scnym.trainer.negative_binomial_loss,
        model=model,
    )

    reg_crit = scnym.losses.LatentL2()

    criteria = [
        {
            "function": mm_crit,
            "weight": 1.0,
            "validation": True,
        },
        {
            "function": rec_crit,
            "weight": scnym.losses.ICLWeight(ramp_epochs=100, max_unsup_weight=0.1),
        },
        {
            "function": reg_crit,
            "weight": scnym.losses.ICLWeight(ramp_epochs=100, max_unsup_weight=1e-5),
        },
    ]

    #####################################
    # Setup optimizer
    #####################################

    optimizer = torch.optim.Adadelta(
        [
            {"params": model.parameters(), "name": "clf"},
            {
                "params": rec_crit.rec_model.decoder.parameters(),
                "name": "ae_dec",
                "weight_decay": 1e-6,
            },
            {"params": rec_crit.rec_model.dispersion, "name": "dispersion"},
            {
                "params": rec_crit.rec_model.libenc.parameters(),
                "name": "ae_libenc",
            },
        ],
    )

    trainer_kwgs = {
        "criteria": criteria,
        "unsup_dataloader": u_dl,
        "verbose": True,
        "scheduler": None,
        "exp_name": "tmp",
        "out_path": "./tmp/",
        "model": model,
        "optimizer": optimizer,
        "dataloaders": dataloaders,
        "n_epochs": 2,
    }

    MTT = scnym.trainer.MultiTaskTrainer(
        **trainer_kwgs,
    )
    MTT.train()
    return


def main():
    test_multitask_trainer()
    return


if __name__ == "__main__":
    main()
