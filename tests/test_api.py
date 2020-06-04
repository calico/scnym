import pytest
import numpy as np
import anndata
import os
import os.path as osp
from pathlib import Path
import scanpy as sc

import sys
sys.path.append('../')
import scnym
scnym_api = scnym.api.scnym_api
atlas2target = scnym.api.atlas2target
TEST_URL = scnym.api.TEST_URL


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
        size=5000,
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


def test_pretrained():
    np.random.seed(1)    
    # load a testing dataset
    adata = sc.datasets._datasets.read(sc.settings.datasetdir / 'kang17.h5ad', backup_url=TEST_URL)
    target_bidx = adata.obs['stim']=='stim'
    adata.obs['cell'] = np.array(adata.obs['cell'])
    adata.obs['ground_truth'] = np.array(adata.obs['cell'])
    adata.obs.loc[target_bidx, 'cell'] = 'Unlabeled'
    
    # predict using a pretrained scNym model
    scnym_api(
        adata=adata,
        task='predict',
        groupby='cell',
        out_path=sc.settings.datasetdir,
        trained_model='pretrained_human',
    )
    
    assert 'X_scnym' in adata.obsm.keys()
    assert 'scNym' in adata.obs.columns
    
    # check accuracy
    # these labels won't match, so we just inspect them manually
    # to assess sanity
    target_gt = np.array(
        adata.obs.loc[target_bidx, 'ground_truth']
    )
    pred = np.array(adata.obs.loc[target_bidx, 'scNym'])
    print('Example Truth vs. Prediction')
    for i in range(30):
        print(f'{target_gt[i]}\t|\t{pred[i]}')    

    return


def test_atlas2target():
    np.random.seed(1)    
    adata = sc.datasets._datasets.read(sc.settings.datasetdir / 'kang17.h5ad', backup_url=TEST_URL)
    
    joint_adata = atlas2target(
        adata=adata,
        species='human',
        key_added='annotations',
    )
    
    assert 'annotations' in joint_adata.obs.keys()
    assert np.sum(joint_adata.obs['annotations']=='Unlabeled')==adata.shape[0]
    
    return


def test_sslwithatlas():
    np.random.seed(1) 
    adata = sc.datasets._datasets.read(sc.settings.datasetdir / 'kang17.h5ad', backup_url=TEST_URL)
    
    joint_adata = atlas2target(
        adata=adata,
        species='human',
        key_added='annotations',
    )
    # downsample to speed up the test
    ridx = np.random.choice(
        joint_adata.shape[0],
        size=5000,
        replace=False,
    )
    joint_adata = joint_adata[ridx, :]
    config = {'n_epochs': 1, 'lr': 1.0}
    scnym_api(
        adata=joint_adata,
        task='train',
        groupby='annotations',
        out_path=sc.settings.datasetdir,
        config=config,
    )
    return


def main():
    test_scnym()
    return


if __name__ == '__main__':
    main()