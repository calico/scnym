'''Test the model Trainer'''
import pytest
import os
import os.path as osp
import sys
import numpy as np
import torch

sys.path.append('../')


def test_trainer():
    import scnym
    import scanpy as sc
    import pandas as pd

    # load 10x human PBMC data as a sample
    adata = sc.datasets.pbmc3k()
    sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e6)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    sc.pp.pca(adata)
    sc.pp.neighbors(adata, n_neighbors=15)
    # generate clusters to use as class labels
    sc.tl.leiden(adata, resolution=0.5, key_added='leiden')
    
    adata.obs['class'] = pd.Categorical(
        adata.obs['leiden'],
    ).codes

    X = adata.X
    y = np.array(adata.obs['class'])
    
    ds = scnym.dataprep.SingleCellDS(
        X=X,
        y=y,
    )
    
    l_dl = torch.utils.data.DataLoader(ds, batch_size=256)
    t_dl = torch.utils.data.DataLoader(ds, batch_size=256)
    u_dl = torch.utils.data.DataLoader(ds, batch_size=256)
    
    dataloaders = {
        'train': l_dl,
        'val'  : t_dl,
    }
    
    criterion = scnym.trainer.cross_entropy
    
    model = scnym.model.CellTypeCLF(
        n_genes=X.shape[1],
        n_cell_types=len(np.unique(y)),
        n_hidden=128,
        n_layers=1,
        init_dropout=0.1,
    )
    if torch.cuda.is_available():
        model = model.cuda()
    
    optimizer = torch.optim.Adadelta(
        model.parameters(),
    )
    trainer_kwgs = {
        'verbose' : True,
        'scheduler' : None,
        'exp_name' : 'tmp',
        'out_path' : './tmp/',
        'model' : model,
        'optimizer' : optimizer,
        'criterion' : criterion,
        'dataloaders': dataloaders,
        'n_epochs': 2,
    }
    
    
    # initialize a standard data loader
    T = scnym.trainer.Trainer(
        **trainer_kwgs
    )
    T.train()
    
    # initialize a semi-supervised data loader
    model = scnym.model.CellTypeCLF(
        n_genes=X.shape[1],
        n_cell_types=len(np.unique(y)),
        n_hidden=128,
        n_layers=1,
    )
    if torch.cuda.is_available():
        model = model.cuda()
    
    optimizer = torch.optim.Adadelta(
        model.parameters(),
    )    
    
    ssl_kwargs = {}
    unsup_criterion = torch.nn.MSELoss(reduction='none')
    USL = scnym.trainer.MixMatchLoss(
        alpha=0.3,
        unsup_criterion=unsup_criterion,
        sup_criterion=trainer_kwgs['criterion'],
        decay_coef=ssl_kwargs.get('decay_coef', 0.997),
        mean_teacher=False,
        augment=scnym.dataprep.AUGMENTATION_SCHEMES['None'],
        n_augmentations=1,
        T=0.5,
        augment_pseudolabels=False,
        pseudolabel_min_confidence=0.6,
    )
    
    weight_schedule = scnym.trainer.ICLWeight(
        ramp_epochs=1,
        max_unsup_weight=0.,
        burn_in_epochs = 0,
        sigmoid = True,
    )
    
    T = scnym.trainer.SemiSupervisedTrainer(
        unsup_dataloader=u_dl,
        unsup_criterion=USL,
        unsup_weight=weight_schedule,
        **trainer_kwgs,
    )
    T.train()
    assert T.best_loss < 0.1
    
    # train with a DANN
    model = scnym.model.CellTypeCLF(
        n_genes=X.shape[1],
        n_cell_types=len(np.unique(y)),
        n_hidden=128,
        n_layers=1,
    )
    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = torch.optim.Adadelta(
        model.parameters(),
    )    
    
    ssl_kwargs = {}
    unsup_criterion = torch.nn.MSELoss(reduction='none')
    USL = scnym.trainer.MixMatchLoss(
        alpha=0.3,
        unsup_criterion=unsup_criterion,
        sup_criterion=trainer_kwgs['criterion'],
        decay_coef=ssl_kwargs.get('decay_coef', 0.997),
        mean_teacher=False,
        augment=scnym.dataprep.AUGMENTATION_SCHEMES['None'],
        n_augmentations=1,
        T=0.5,
        augment_pseudolabels=False,
        pseudolabel_min_confidence=0.6,
    )
    dan_criterion = scnym.trainer.DANLoss(
        model=model,
        dan_criterion=scnym.trainer.cross_entropy,
        use_conf_pseudolabels=True,
    )
    
    weight_schedule = scnym.trainer.ICLWeight(
        ramp_epochs=1,
        max_unsup_weight=0.1,
        burn_in_epochs = 0,
        sigmoid = True,
    )
    dan_weight = scnym.trainer.ICLWeight(
        ramp_epochs=1,
        max_unsup_weight=0.1,
        burn_in_epochs = 0,
        sigmoid = True,
    )    
    
    T = scnym.trainer.SemiSupervisedTrainer(
        unsup_dataloader=u_dl,
        unsup_criterion=USL,
        unsup_weight=weight_schedule,
        dan_criterion=dan_criterion,
        dan_weight=dan_weight,
        **trainer_kwgs,
    )
    T.train()
    assert T.best_loss < 0.1    
    
    return


def main():
    test_trainer()


if __name__ == '__main__':
    main()

    
    
    