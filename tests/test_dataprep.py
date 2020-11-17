import pytest
import os
import os.path as osp
import sys
import numpy as np
import torch

sys.path.append('../')


def test_dataset():
    '''Test loading samples from a dataset.
    Checks dataset creation and index parsing.
    '''
    import scanpy as sc
    from scnym.dataprep import SingleCellDS
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
    # add fake domain labels
    adata.obs['domain'] = np.random.randint(
        0,
        3,
        adata.shape[0],
    )
    
    ds = SingleCellDS(
        X = adata.X.toarray(),
        y = np.array(adata.obs['class']),
        domain = np.array(adata.obs['domain']),
    )
    
    # draw a sample
    idx = 10
    sample = ds[idx]
    sample['input'] = sample['input'].unsqueeze(0)
    # ensure the sample is the correct index
    assert np.all(
        sample['input'].numpy() == adata.X[idx, :].toarray().flatten()
    )
    # ensure the class label is correct
    y_gt = np.array(adata.obs['class'])[idx]
    y_onehot = sample['output'].view(1, -1)
    _, y_ds = torch.max(y_onehot, dim=1)
    assert np.all(y_ds.item() == y_gt)
    # ensure the domain label is correct
    d_gt = np.array(adata.obs['domain'])[idx]
    d_onehot = sample['domain'].view(1, -1)
    _, d_ds = torch.max(d_onehot, dim=1)
    assert np.all(d_ds.item() == d_gt)    
    
    # throw an error by passing an invalid idx
    
    with pytest.raises(TypeError):
        sample = ds['bad_index_type']
    
    return


def test_dataset_sparse():
    '''Test loading samples from a dataset using sparse matrices.
    Checks dataset creation and index parsing.
    '''
    import scanpy as sc
    from scnym.dataprep import SingleCellDS
    from scipy import sparse
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
    
    X = sparse.csr_matrix(adata.X)
    
    ds = SingleCellDS(
        X = X,
        y = np.array(adata.obs['class']),
    )
    
    # draw a sample
    idx = 10
    sample = ds[idx]
    sample['input'] = sample['input'].unsqueeze(0)
    # ensure the sample is the correct index
    assert np.all(
        sample['input'].numpy() == adata.X[idx, :].toarray().flatten()
    )
    
    # throw an error by passing an invalid idx
    
    with pytest.raises(TypeError):
        sample = ds['bad_index_type']
    
    return


def test_normalization():
    '''Test log transformations and library size normalization'''
    import scanpy as sc
    from scnym.dataprep import SingleCellDS, LibrarySizeNormalize, ExpMinusOne
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
    
    ds = SingleCellDS(
        X = adata.X.toarray(),
        y = np.array(adata.obs['class']),
    )
    
    # draw a sample
    idx = 10
    sample = ds[idx]
    sample['input'] = sample['input'].unsqueeze(0)    
    
    # normalize the vector and log1p transform
    norm = LibrarySizeNormalize(
        log1p=True,
    )
    sample_norm = norm(sample)
    
    # exponentiate the vector
    eom = ExpMinusOne()
    sample_expm1 = eom(sample)
    
    # renormalize the vector and check validity
    sample_renorm = norm(sample_expm1)
    
    assert torch.all(
        sample_renorm['input']==sample_norm['input']
    )
    
    return
    

def test_multinomial_sampling():
    '''Test that multinomial sampling works using a large
    sample depth and a permutation test
    '''
    import scanpy as sc
    from scnym.dataprep import SingleCellDS, LibrarySizeNormalize, MultinomialSample
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
    
    ds = SingleCellDS(
        X = adata.X.toarray(),
        y = np.array(adata.obs['class']),
    )
    
    # draw a sample
    idx = 10
    sample = ds[idx]
    sample['input'] = sample['input'].unsqueeze(0)    
    # keep an unmodified copy of the sample
    sample_orig = sample.copy()
    
    # normalize the sample but do not log transform
    norm = LibrarySizeNormalize(
        log1p=False,
    )
    sample_norm = norm(sample)
        
    # get a multinomial vector with random depth
    mn = MultinomialSample(
        depth = (10000, 20000),
    )
    sample_mn = mn(sample_norm)
    # check that the depth is in the expected range
    d = torch.sum(sample_mn['input'], 1)
    lt_bidx = d < 10000
    gt_bidx = d > 20000
    assert torch.sum(lt_bidx) == 0.
    assert torch.sum(gt_bidx) == 0.
    
    # sample deeply from a multinomial
    mn = MultinomialSample(
        depth = (1000000, 1000001),
    )
    sample_mn = mn(sample_norm)
    # check that the depth is in the expected range
    d = torch.sum(sample_mn['input'], 1)
    lt_bidx = d < 1000000
    gt_bidx = d > 1000001
    assert torch.sum(lt_bidx) == 0.
    assert torch.sum(gt_bidx) == 0.    
    
    # sample using a depth ratio
    mn = MultinomialSample(
        depth_ratio = (0.25, 2),
    )
    sample_mn = mn(sample_norm)
    # check that the depth is in the expected range
    d = torch.sum(sample_mn['input'], 1)
    in_sizes = torch.sum(sample_norm['input'], 1)
    l_sz = in_sizes * 0.25
    h_sz = in_sizes * 2
    lt_bidx = d < l_sz
    gt_bidx = d > h_sz    
    assert torch.sum(lt_bidx) == 0.
    assert torch.sum(gt_bidx) == 0.     
    
    # normalize the sample abundance profile
    sample_mn_norm = norm(sample_mn)
    
    # get the l2 norm between the original sample and the
    # counts drawn from a multinomial
    d = torch.log1p(sample_mn_norm['input']) - torch.log1p(sample_orig['input'])
    test_norm = torch.norm(
        d, 2,
    )
    
    # permute genes to generate a null distribution of norms
    n_tests = 1000
    rand_norms = torch.zeros(n_tests,)
    for i in range(n_tests):
        ridx = torch.randperm(sample_orig['input'].size(0))
        d = torch.log1p(sample_mn_norm['input']) - torch.log1p(sample_orig['input'][ridx])
        rand_norms[i] = torch.norm(
            d, 
            2,
        )

    p = torch.sum(test_norm > rand_norms) // n_tests
    
    msg = f'p = {p} suggests something is rotten with multinomial sampling.'
    assert p < 0.05, msg
    
    return


def test_poisson_sampling():
    '''Test the Poisson sampling method.'''
    import scanpy as sc
    from scnym.dataprep import SingleCellDS, LibrarySizeNormalize, PoissonSample
    from torchvision.transforms import Compose
    torch.manual_seed(17)
    np.random.seed(1)    
    
    # load 10x human PBMC data as a sample
    adata = sc.datasets.pbmc3k()
    # generate fake class labels
    adata.obs['class'] = np.random.randint(
        0,
        3,
        adata.shape[0],
    )
    
    ds = SingleCellDS(
        X = adata.X.toarray(),
        y = np.array(adata.obs['class']),
    )
    
    # draw a sample
    idx = 10
    sample = ds[idx]
    sample['input'] = sample['input'].unsqueeze(0)    
    # keep an unmodified copy of the sample
    sample_orig = sample.copy()
    
    # normalize the sample but do not log transform
    norm = LibrarySizeNormalize(
        log1p=False,
    )
    sample_norm = norm(sample)
    sample_norm_orig = sample_norm.copy()
    
    # sample using a depth ratio
    poisson = PoissonSample(depth_range=(0.5, 2.))
    sample_ps = poisson(sample_norm.copy())
    print('initial:')
    print(sample_norm_orig['input'][sample_norm_orig['input']>0][:20])
    print('sampled 00:')
    print(sample_ps['input'][sample_ps['input']>0][:20])
    
    # check that sample depths are in an expected range
    library_sizes = sample_ps['input'].sum(1)
    too_small_bidx = torch.sum(library_sizes < 4.5e5)
    too_large_bidx = torch.sum(library_sizes > 2.1e6)
    if not too_small_bidx == 0:
        raise RuntimeError(f'{too_small_bidx} cells had small libraries.')
    if not too_large_bidx == 0:
        raise RuntimeError(f'{too_large_bidx} cells had large libraries.')    

    # check that a second sample is different    
    sample_ps2 = poisson(sample_norm.copy())
    print('sampled 01:')
    print(sample_ps2['input'][sample_ps2['input']>0][:20])
    
    all_eq = torch.all(
        sample_ps['input'] == sample_ps2['input']
    )
    if all_eq:
        print('sampled 00:')
        print(sample_ps['input'][sample_ps['input']>0][:20])        
        print('sampled 01:')
        print(sample_ps2['input'][sample_ps2['input']>0][:20])        
        msg = f'The two Poisson samples were identitcal: {all_eq}'
        raise RuntimeError(msg)
        
    # test passing a minibatch for sampling
    dl = torch.utils.data.DataLoader(ds, batch_size=16)
    batch = next(iter(dl))
    
    tr = Compose(
        [
            LibrarySizeNormalize(log1p=False),
            PoissonSample(),
            LibrarySizeNormalize(log1p=True),
        ]
    )
    
    batch_sampled = tr(batch)
    
    return

def test_gene_masking():
    '''Test that gene masking augmentations zero out a
    random subset of genes'''
    import scanpy as sc
    from scnym.dataprep import SingleCellDS, GeneMasking
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
    
    ds = SingleCellDS(
        X = adata.X.toarray(),
        y = np.array(adata.obs['class']),
    )
    
    # draw a sample
    idx = 10
    sample = ds[idx]
    sample['input'] = sample['input'].unsqueeze(0)    
    sample_orig = sample.copy()
    
    # initiate a gene masker
    gm = GeneMasking(p_drop = 0.8, p_apply=1.0)
    
    sample_masked = gm(sample.copy())
    
    # check that we masked some genes
    n_genes_original = torch.sum(sample_orig['input'] > 0)
    n_genes_masked = torch.sum(sample_masked['input'] > 0)
    assert n_genes_masked < n_genes_original
    
    # check the masking is randomized
    sample_masked0 = gm(sample.copy())
    sample_masked1 = gm(sample.copy())
    
    bidx0 = sample_masked0['input']==0
    bidx1 = sample_masked1['input']==0
    assert ~torch.all(bidx0 & bidx1)
    
    return


def test_mixup():
    import scanpy as sc
    from scnym.dataprep import SingleCellDS, SampleMixUp
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
    
    # draw a batch from the dataloader
    ds = SingleCellDS(
        X = adata.X.toarray(),
        y = np.array(adata.obs['class']),
    )
    dl = torch.utils.data.DataLoader(ds, batch_size=16)
    
    batch = next(iter(dl))
    
    # mixup the batch
    mixup_op = SampleMixUp(
        alpha = 0.3,
        keep_dominant_obs = True,
    )
    
    mixed_batch = mixup_op(batch)
    
    # test that mixup can be turned off with `alpha = 0.0`
    batch = next(iter(dl))
    mixup_op = SampleMixUp(
        alpha = 0.0,
        keep_dominant_obs = True,
    )
    
    mixed_batch = mixup_op(batch)
    
    assert torch.all(
        mixed_batch['input'] == batch['input']
    )
    
    return


def test_augmentation_schemes():
    '''Test each of the predefined augmentation schemes'''
    import scanpy as sc
    from scnym.dataprep import SingleCellDS, SampleMixUp, AUGMENTATION_SCHEMES
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
    
    # draw a batch from the dataloader
    ds = SingleCellDS(
        X = adata.X.toarray(),
        y = np.array(adata.obs['class']),
    )
    dl = torch.utils.data.DataLoader(ds, batch_size=16)
    
    
    for k in AUGMENTATION_SCHEMES.keys():
        aug = AUGMENTATION_SCHEMES[k]
        batch = next(iter(dl))
        try:
            augmented = aug(batch)
        except RuntimeError:
            print(f'Augmentation scheme {k} failed.')
    return
        


def main():
    test_mixup()
    return


if __name__ == '__main__':
    main()