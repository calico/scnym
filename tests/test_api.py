import pytest
import numpy as np
import torch
import anndata
import os
import os.path as osp
from pathlib import Path
import scanpy as sc
from sklearn.model_selection import StratifiedKFold

import sys
sys.path = ['../'] + sys.path
import scnym
scnym_api = scnym.api.scnym_api
atlas2target = scnym.api.atlas2target
scnym_tune = scnym.api.scnym_tune
split_data = scnym.api.split_data
TEST_URL = scnym.api.TEST_URL


def cuda_only(func):
    '''Decorator that sets a test to run only if CUDA is available'''
    def wrap_cuda_only():
        if not torch.cuda.is_available():
            print('Test only run if CUDA compute device is available.')
            return
        else:
            func()
    return wrap_cuda_only


def test_scnym():
    '''Test scNym training and prediction'''
    np.random.seed(1)
    # load a testing dataset
    adata = sc.datasets._datasets.read(sc.settings.datasetdir / 'kang17.h5ad', backup_url=TEST_URL)
    target_bidx = adata.obs['stim']=='stim'
    adata.obs['cell'] = np.array(adata.obs['cell'])
    adata.obs['ground_truth'] = np.array(adata.obs['cell'])
    adata.obs.loc[target_bidx, 'cell'] = 'Unlabeled'
    
    # downsample to speed up testing
    ridx = np.random.choice(
        adata.shape[0],
        size=2048,
        replace=False,
    )
    adata = adata[ridx, :].copy()
    
    # train an scNym model
    config = {'n_epochs': 1}
    scnym_api(
        adata=adata,
        task='train',
        groupby='cell',
        out_path=str(sc.settings.datasetdir),
        config=config,
    )
    
    assert 'scNym_train_results' in adata.uns.keys()
    assert osp.exists(osp.join(str(sc.settings.datasetdir), '00_best_model_weights.pkl'))
    
    # predict cell types with an scNym model
    scnym_api(
        adata=adata,
        task='predict',
        key_added='scNym',
        out_path=str(sc.settings.datasetdir),
        trained_model=str(sc.settings.datasetdir),
        config=config,
    )
    
    assert 'X_scnym' in adata.obsm.keys()
    assert 'scNym' in adata.obs.columns
    
    # check accuracy
    target_gt = np.array(
        adata.obs.loc[target_bidx, 'ground_truth']
    )
    pred = np.array(adata.obs.loc[target_bidx, 'scNym'])
    print('Example Truth vs. Prediction')
    for i in range(10):
        print(f'{target_gt[i]}\t|\t{pred[i]}')
    print()
    acc = np.sum(target_gt == pred) / len(pred) * 100
    print(f'Accuracy on target set: {acc}%')
    
    return


def test_assumption_checking():
    '''Test that errors are thrown if assumptions are violated'''
    np.random.seed(1)
    # load a testing dataset
    adata = sc.datasets._datasets.read(sc.settings.datasetdir / 'kang17.h5ad', backup_url=TEST_URL)
    target_bidx = adata.obs['stim']=='stim'
    adata.obs['cell'] = np.array(adata.obs['cell'])
    adata.obs['ground_truth'] = np.array(adata.obs['cell'])
    adata.obs.loc[target_bidx, 'cell'] = 'Unlabeled'
    
    # downsample to speed up testing
    ridx = np.random.choice(
        adata.shape[0],
        size=2048,
        replace=False,
    )
    adata = adata[ridx, :].copy()
    
    
    # test that an input anndata with duplicate genes
    # throws an error
    adata_dup_genes = adata.copy()
    var_names_with_dups = np.array(adata_dup_genes.var_names)
    var_names_with_dups[-1] = var_names_with_dups[-2]
    adata_dup_genes.var_names = var_names_with_dups
    
    with pytest.raises(ValueError, match='Duplicate Genes'):
        # this should throw an error about duplicate genes
        config = {'n_epochs': 1}
        scnym_api(
            adata=adata_dup_genes,
            task='train',
            groupby='cell',
            out_path=str(sc.settings.datasetdir),
            config=config,
        )
        
    # test that an input anndata with `.X` formatted as something
    # other than log1p(CPM) will throw an error
    adata_not_log1p = adata.copy()
    adata_not_log1p.X = np.expm1(adata_not_log1p.X)

    with pytest.raises(ValueError, match='Normalization'):
        # this should throw an error with instructions on how to 
        # normalize data
        scnym_api(
            adata=adata_not_log1p,
            task='train',
            groupby='cell',
            out_path=str(sc.settings.datasetdir),
            config=config,
        )
    return


def test_pretrained():
    '''Test that pretrained models are disabled'''
    np.random.seed(1)    
    # load a testing dataset
    adata = sc.datasets._datasets.read(sc.settings.datasetdir / 'kang17.h5ad', backup_url=TEST_URL)
    target_bidx = adata.obs['stim']=='stim'
    adata.obs['cell'] = np.array(adata.obs['cell'])
    adata.obs['ground_truth'] = np.array(adata.obs['cell'])
    adata.obs.loc[target_bidx, 'cell'] = 'Unlabeled'
    
    sc.pp.normalize_total(adata, target_sum=int(1e6))
    sc.pp.log1p(adata)    
    
    # predict using a pretrained scNym model
    with pytest.raises(NotImplementedError, match='Pretrained'):
        scnym_api(
            adata=adata,
            task='predict',
            groupby='cell',
            out_path=sc.settings.datasetdir,
            trained_model='pretrained_human',
        )
    
#     assert 'X_scnym' in adata.obsm.keys()
#     assert 'scNym' in adata.obs.columns
    
#     # check accuracy
#     # these labels won't match, so we just inspect them manually
#     # to assess sanity
#     target_gt = np.array(
#         adata.obs.loc[target_bidx, 'ground_truth']
#     )
#     pred = np.array(adata.obs.loc[target_bidx, 'scNym'])
#     print('Example Truth vs. Prediction')
#     for i in range(30):
#         print(f'{target_gt[i]}\t|\t{pred[i]}')

    return


def test_atlas2target():
    np.random.seed(1)    
    adata = sc.datasets.paul15()
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.log1p(adata)
    
    joint_adata = atlas2target(
        adata=adata,
        species='mouse',
        key_added='annotations',
    )
    
    assert 'annotations' in joint_adata.obs.keys()
    assert np.sum(joint_adata.obs['annotations']=='Unlabeled')==adata.shape[0]
    
    return


def test_sslwithatlas():
    np.random.seed(1) 
    adata = sc.datasets.paul15()
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.log1p(adata)    
    
    joint_adata = atlas2target(
        adata=adata,
        species='mouse',
        key_added='annotations',
    )
    # downsample to speed up the test
    ridx = np.random.choice(
        joint_adata.shape[0],
        size=5000,
        replace=False,
    )
    joint_adata = joint_adata[ridx, :].copy()
    sc.pp.filter_genes(joint_adata, min_cells=10)
    print('Subsampled `joint_adata`: %d cells, %d genes' % joint_adata.shape)
    print('joint obs columns:')
    print(joint_adata.obs.columns)
    print()
    config = {'n_epochs': 2, 'batch_size': 32}
    scnym_api(
        adata=joint_adata,
        task='train',
        groupby='annotations',
        out_path=str(sc.settings.datasetdir),
        config=config,
    )
    return


@cuda_only
def test_user_indices():
    '''Test scNym training and prediction with user supplied indices
    for the data split.'''
    np.random.seed(1)
    # load a testing dataset
    adata = sc.datasets._datasets.read(sc.settings.datasetdir / 'kang17.h5ad', backup_url=TEST_URL)
    target_bidx = adata.obs['stim']=='stim'
    adata.obs['cell'] = np.array(adata.obs['cell'])
    adata.obs['ground_truth'] = np.array(adata.obs['cell'])
    adata.obs.loc[target_bidx, 'cell'] = 'Unlabeled'
    
    # downsample to speed up testing
    ridx = np.random.choice(
        adata.shape[0],
        size=2048,
        replace=False,
    )
    adata = adata[ridx, :].copy()
    
    target_bidx = adata.obs['stim']=='stim'
    adata.obs['scNym_split'] = np.random.choice(
        ['train', 'test', 'val'],
        size=adata.shape[0],
        p=[0.8, 0.1, 0.1],
        replace=True,
    )
    print('train samples = ', np.sum(adata.obs['scNym_split']=='train'))
    print('test  samples = ', np.sum(adata.obs['scNym_split']=='test'))
    print('val   samples = ', np.sum(adata.obs['scNym_split']=='val'))
    
    # train an scNym model
    config = {'n_epochs': 1}
    scnym_api(
        adata=adata,
        task='train',
        groupby='cell',
        out_path=str(sc.settings.datasetdir),
        config=config,
    )
    
    assert 'scNym_train_results' in adata.uns.keys()
    assert osp.exists(osp.join(str(sc.settings.datasetdir), '00_best_model_weights.pkl'))
    
    # predict cell types with an scNym model
    scnym_api(
        adata=adata,
        task='predict',
        key_added='scNym',
        out_path=str(sc.settings.datasetdir),
        trained_model=str(sc.settings.datasetdir),
        config=config,
    )
    
    assert 'X_scnym' in adata.obsm.keys()
    assert 'scNym' in adata.obs.columns
    
    # check accuracy
    target_gt = np.array(
        adata.obs.loc[target_bidx, 'ground_truth']
    )
    pred = np.array(adata.obs.loc[target_bidx, 'scNym'])
    print('Example Truth vs. Prediction')
    for i in range(10):
        print(f'{target_gt[i]}\t|\t{pred[i]}')
    print()
    acc = np.sum(target_gt == pred) / len(pred) * 100
    print(f'Accuracy on target set: {acc:.03f}%')
    
    return


@cuda_only
def test_user_domains():
    '''Test scNym training and prediction with user supplied domain
    labels for each cell.'''
    np.random.seed(1)
    # load a testing dataset
    adata = sc.datasets._datasets.read(sc.settings.datasetdir / 'kang17.h5ad', backup_url=TEST_URL)
    target_bidx = adata.obs['stim']=='stim'
    adata.obs['cell'] = np.array(adata.obs['cell'])
    adata.obs['ground_truth'] = np.array(adata.obs['cell'])
    adata.obs.loc[target_bidx, 'cell'] = 'Unlabeled'
    
    # downsample to speed up testing
    ridx = np.random.choice(
        adata.shape[0],
        size=2048,
        replace=False,
    )
    adata = adata[ridx, :].copy()
    
    # use individual patients as unique domain labels
    domain_groupby = 'ind'
    
    target_bidx = adata.obs['stim']=='stim'
    adata.obs['scNym_split'] = np.random.choice(
        ['train', 'test', 'val'],
        size=adata.shape[0],
        p=[0.8, 0.1, 0.1],
        replace=True,
    )
    print('train samples = ', np.sum(adata.obs['scNym_split']=='train'))
    print('test  samples = ', np.sum(adata.obs['scNym_split']=='test'))
    print('val   samples = ', np.sum(adata.obs['scNym_split']=='val'))
    
    # train an scNym model
    config = {'n_epochs': 1, 'domain_groupby': domain_groupby}
    scnym_api(
        adata=adata,
        task='train',
        groupby='cell',
        out_path=str(sc.settings.datasetdir),
        config=config,
    )
    
    assert 'scNym_train_results' in adata.uns.keys()
    assert osp.exists(osp.join(str(sc.settings.datasetdir), '00_best_model_weights.pkl'))
    
    # predict cell types with an scNym model
    scnym_api(
        adata=adata,
        task='predict',
        key_added='scNym',
        out_path=str(sc.settings.datasetdir),
        trained_model=str(sc.settings.datasetdir),
        config=config,
    )
    
    assert 'X_scnym' in adata.obsm.keys()
    assert 'scNym' in adata.obs.columns
    
    # check accuracy
    target_gt = np.array(
        adata.obs.loc[target_bidx, 'ground_truth']
    )
    pred = np.array(adata.obs.loc[target_bidx, 'scNym'])
    print('Example Truth vs. Prediction')
    for i in range(10):
        print(f'{target_gt[i]}\t|\t{pred[i]}')
    print()
    acc = np.sum(target_gt == pred) / len(pred) * 100
    print(f'Accuracy on target set: {acc:.03f}%')
    
    return


@cuda_only
def test_tune():
    '''Test a hyperparameter tuning experiment for scNym'''
    np.random.seed(1)
    # load a testing dataset
    adata = sc.datasets._datasets.read(sc.settings.datasetdir / 'kang17.h5ad', backup_url=TEST_URL)
    target_bidx = adata.obs['stim']=='stim'
    adata.obs['cell'] = np.array(adata.obs['cell'])
    adata.obs['ground_truth'] = np.array(adata.obs['cell'])
    adata.obs.loc[target_bidx, 'cell'] = 'Unlabeled'
    
    ridx = np.random.choice(adata.shape[0], size=2000, replace=False)
    adata = adata[ridx, :].copy()
    
    parameters = {
        'n_epochs': [1,],
        'lr': [1e-3, 1.],
        'ssl_kwargs': {
            'dan_ramp_epochs': [20,],
        }
    }
    
    results, best_params = scnym_tune(
        adata=adata,
        groupby='cell',
        parameters=parameters,
        hold_out_only=True,
        groupby_eval='ground_truth',
        n_splits=2,
    )
    
    print('Results')
    print('-'*10)
    print(results)
    print('-'*10)
    print()
    
    return


@cuda_only
def test_tune_provide_splits():
    '''Test a hyperparameter tuning experiment for scNym
    with user provided 5-fold splits.
    '''
    np.random.seed(1)
    # load a testing dataset
    adata = sc.datasets._datasets.read(sc.settings.datasetdir / 'kang17.h5ad', backup_url=TEST_URL)
    target_bidx = adata.obs['stim']=='stim'
    adata.obs['cell'] = np.array(adata.obs['cell'])
    adata.obs['ground_truth'] = np.array(adata.obs['cell'])
    adata.obs.loc[target_bidx, 'cell'] = 'Unlabeled'
    
    ridx = np.random.choice(adata.shape[0], size=2000, replace=False)
    adata = adata[ridx, :].copy()
    
    split_data(
        adata,
        groupby='cell',
        n_splits=2,
    )    
    
    parameters = {
        'n_epochs': [1,],
        'lr': [1e-3, 1.],
        'ssl_kwargs': {
            'dan_ramp_epochs': [20,],
        },
    }
    
    results, best_params = scnym_tune(
        adata=adata,
        groupby='cell',
        parameters=parameters,
        hold_out_only=True,
        groupby_eval='ground_truth',
        n_splits=2,
    )
    
    print('Results')
    print('-'*10)
    print(results)
    print('-'*10)
    print()
    
    return


def main():
    print('<-- test_scnym')
    test_scnym()
    print('<-- test_assumption_checking')
    test_assumption_checking()
    print('<-- test_pretrained')
    test_pretrained()
    print('<-- test_atlas2target')
    test_atlas2target()
    print('<-- test_sslwithatlas')
    test_sslwithatlas()    
    print('<-- test_user_indices')
    test_user_indices()
    print('<-- test_user_domains')
    test_user_domains()
    print('<-- test_tuning')
    test_tune()
    print('<-- test_tune_provide_splits')
    test_tune_provide_splits()
    return


if __name__ == '__main__':
    main()