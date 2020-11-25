import pytest
import os
import os.path as osp
import sys
import numpy as np
import torch

sys.path.append('../')

def test_fit_model_sup():
    '''Test the model fitting function for supervised learning.
    '''
    import scnym
    import scanpy as sc
    torch.manual_seed(1)
    np.random.seed(1)     

    # load 10x human PBMC data as a sample
    adata = sc.datasets.pbmc3k()
    # generate fake class labels
    adata.obs['class'] = np.random.randint(
        0,
        3,
        adata.shape[0],
    )
    
    X = adata.X
    y = np.array(adata.obs['class'])
    
    # test training without unlabeled counts    
    os.makedirs('./tmp', exist_ok=True)
    acc, loss = scnym.main.fit_model(
        X=X,
        y=y,
        traintest_idx=np.arange(X.shape[0]//2),
        val_idx=np.arange(X.shape[0]//2, X.shape[0]),
        batch_size=256,
        n_epochs=1,
        lr=1e-5,
        optimizer_name='adadelta',
        weight_decay=1e-4,
        ModelClass=scnym.model.CellTypeCLF,
        out_path='./tmp',
        n_genes=adata.shape[1],
        mixup_alpha=0.3,
        n_hidden=128,
        n_layers=2,
        residual=False,
    )
    
    return

def test_fit_model_ssl():
    '''Test the model fitting function for semi-supervised
    learning.
    '''
    import scnym
    import scanpy as sc
    torch.manual_seed(1)
    np.random.seed(1)     

    # load 10x human PBMC data as a sample
    adata = sc.datasets.pbmc3k()
    # generate fake class labels
    adata.obs['class'] = np.random.randint(
        0,
        3,
        adata.shape[0],
    )
    
    X = adata.X
    y = np.array(adata.obs['class'])
    # test training with unlabeled counts
    os.makedirs('./tmp', exist_ok=True)    
    acc, loss = scnym.main.fit_model(
        X=X,
        y=y,
        traintest_idx=np.arange(X.shape[0]//2),
        val_idx=np.arange(X.shape[0]//2, X.shape[0]),
        batch_size=256,
        n_epochs=2,
        lr=1e-5,
        optimizer_name='adadelta',
        weight_decay=1e-4,
        ModelClass=scnym.model.CellTypeCLF,
        out_path='./tmp',
        n_genes=adata.shape[1],
        mixup_alpha=0.3,
        n_hidden=128,
        n_layers=2,
        residual=False,
        unlabeled_counts=X[:1000],
        unsup_max_weight=1e-5,
        ssl_method='mixmatch',
        ssl_kwargs={
            'augment': 'None',
            'n_augmentations': 1,
            'augment_pseudolabels': False,
            'T': 1.,
        },
    )
    return


def test_fit_model_ssl_ce():
    '''Test the model fitting function for semi-supervised
    learning with cross-entropy loss.
    '''
    import scnym
    import scanpy as sc
    torch.manual_seed(1)
    np.random.seed(1)     

    # load 10x human PBMC data as a sample
    adata = sc.datasets.pbmc3k()
    # generate fake class labels
    adata.obs['class'] = np.random.randint(
        0,
        3,
        adata.shape[0],
    )
    
    X = adata.X
    y = np.array(adata.obs['class'])
    # test training with unlabeled counts
    os.makedirs('./tmp', exist_ok=True)    
    acc, loss = scnym.main.fit_model(
        X=X,
        y=y,
        traintest_idx=np.arange(X.shape[0]//2),
        val_idx=np.arange(X.shape[0]//2, X.shape[0]),
        batch_size=256,
        n_epochs=2,
        lr=1e-5,
        optimizer_name='adadelta',
        weight_decay=1e-4,
        ModelClass=scnym.model.CellTypeCLF,
        out_path='./tmp',
        n_genes=adata.shape[1],
        mixup_alpha=0.3,
        n_hidden=128,
        n_layers=2,
        residual=False,
        unlabeled_counts=X[:1000],
        unsup_max_weight=1e-5,
        ssl_method='mixmatch',
        ssl_kwargs={
            'augment': 'None',
            'n_augmentations': 1,
            'augment_pseudolabels': False,
            'T': 1.,
            'unsup_criterion': 'crossentropy',
        },
    )
    return


def test_fit_model_ssl_dan():
    '''Test the model fitting function for semi-supervised
    learning with domain adaptation.
    '''
    import scnym
    import scanpy as sc
    torch.manual_seed(1)
    np.random.seed(1)    

    # load 10x human PBMC data as a sample
    adata = sc.datasets.pbmc3k()
    # generate fake class labels
    adata.obs['class'] = np.random.randint(
        0,
        3,
        adata.shape[0],
    )
    
    X = adata.X
    y = np.array(adata.obs['class'])
    # test training with unlabeled counts    
    acc, loss = scnym.main.fit_model(
        X=X,
        y=y,
        traintest_idx=np.arange(X.shape[0]//2),
        val_idx=np.arange(X.shape[0]//2, X.shape[0]),
        batch_size=256,
        n_epochs=2,
        lr=1e-5,
        optimizer_name='adadelta',
        weight_decay=1e-4,
        ModelClass=scnym.model.CellTypeCLF,
        out_path='./tmp',
        n_genes=adata.shape[1],
        mixup_alpha=0.3,
        n_hidden=128,
        n_layers=2,
        residual=False,
        unlabeled_counts=X[:1000],
        unsup_max_weight=1e-5,
        ssl_method='mixmatch',
        ssl_kwargs={
            'augment': 'None',
            'n_augmentations': 1,
            'augment_pseudolabels': False,
            'T': 1.,
            'unsup_criterion': 'mse',
            'dan_criterion': True,
            'dan_ramp_epochs': 100,
            'dan_burn_in_epochs': 0,
            'dan_max_weight': 1.,
        },
    )
    return

def test_fit_model_dan_multdom():
    '''Test the model fitting function for semi-supervised
    learning with domain adaptation across >2 domains.
    '''
    import scnym
    import scanpy as sc
    torch.manual_seed(1)
    np.random.seed(1)    

    # load 10x human PBMC data as a sample
    adata = sc.datasets.pbmc3k()
    # generate fake class labels
    adata.obs['class'] = np.random.randint(
        0,
        3,
        adata.shape[0],
    )
    
    X = adata.X
    y = np.array(adata.obs['class'])
    
    input_domain = np.zeros(X.shape[0])
    unlabeled_domain = np.ones(1000)
    unlabeled_domain[500:] += 1
    
    n_domains = int(np.max([
        input_domain.max(), unlabeled_domain.max()
    ])) + 1
    print('Max domain: ')
    # test training with unlabeled counts    
    acc, loss = scnym.main.fit_model(
        X=X,
        y=y,
        traintest_idx=np.arange(X.shape[0]//2),
        val_idx=np.arange(X.shape[0]//2, X.shape[0]),
        batch_size=256,
        n_epochs=2,
        lr=1e-5,
        optimizer_name='adadelta',
        weight_decay=1e-4,
        ModelClass=scnym.model.CellTypeCLF,
        out_path='./tmp',
        n_genes=adata.shape[1],
        mixup_alpha=0.3,
        n_hidden=128,
        n_layers=2,
        residual=False,
        unlabeled_counts=X[:1000],
        unsup_max_weight=1e-5,
        ssl_method='mixmatch',
        ssl_kwargs={
            'augment': 'None',
            'n_augmentations': 1,
            'augment_pseudolabels': False,
            'T': 1.,
            'unsup_criterion': 'mse',
            'dan_criterion': True,
            'dan_ramp_epochs': 100,
            'dan_burn_in_epochs': 0,
            'dan_max_weight': 1.,
        },
        input_domain=input_domain,
        unlabeled_domain=unlabeled_domain,
    )
    return


def main():
    test_fit_model_dan_multdom()
    return


if __name__ == '__main__':
    main()
