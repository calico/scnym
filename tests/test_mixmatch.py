import pytest
import os
import os.path as osp
import sys
import numpy as np
import torch

sys.path.append('../')


def test_samplemixup():
    '''Confirms that MixUp is properly saving the
    sample identity of origin for MixMatch.
    '''
    import scnym
    torch.manual_seed(1)
    np.random.seed(1)    
    
    mixup = scnym.dataprep.SampleMixUp(
        alpha=0.3,
        keep_dominant_obs=True,
    )
    
    # create two tensors with two boolean
    # features. One tensor is positive for
    # feature 0, the other for feature 1
    n = 3
    A = torch.zeros((n, 2))
    A[:, 0] = 1
    B = torch.zeros((n, 2))
    B[:, 1] = 1
    
    # create a dummy sample of these
    # two tensors
    C = torch.cat([A, B], dim=0)
    
    sample = {
        'input' : C,
        'output': C,
        'some_other_tensor': C,
    }
    
    # mix the dummy sample
    sample_mixed = mixup(
        sample,
    )
    
    # if our MixUp worked properly, the first `n`
    # examples should be dominated by tensor `A`,
    # and the latter `n` by tensor `B`.
    _, dom_idx = torch.max(sample_mixed['input'], dim=1)
    assert torch.all(dom_idx[:n] == 0)
    assert torch.all(dom_idx[n:] == 1)
    
    # double check that the remaining tensor
    # was also mixed properly
    _, dom_idx = torch.max(sample_mixed['some_other_tensor'], dim=1)
    assert torch.all(dom_idx[:n] == 0), 'other tensor was not mixed'
    assert torch.all(dom_idx[n:] == 1), 'other tensor was not mixed'
    return


def test_mixmatch_forward():
    '''Perform a forward pass through the MixMatch objective
    using a dummy model and resampling augmentations
    
    Notes
    -----
    Does not verify that computations are correct, merely that
    the forward pass completes without error.
    '''
    
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
    adata.obs['class'] = np.random.randint(
        0,
        3,
        adata.shape[0],
    )
    
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
    
    # draw a batch from each
    d_batch = next(iter(d_dl))
    u_batch = next(iter(u_dl))
    
    # initialize mixmatch using a dummy model
    unsup_criterion = torch.nn.MSELoss(reduction='none')
    sup_criterion = scnym.trainer.cross_entropy
    
    augment = scnym.dataprep.AUGMENTATION_SCHEMES['log1p_drop']
    
    model = scnym.model.CellTypeCLF(
        n_layers=1,
        n_hidden=128,
        residual=False,
        init_dropout=0.0,
        n_genes=adata.shape[1],
        n_cell_types=3,
    )
    
    K = 2
    mixmatch = scnym.trainer.MixMatchLoss(
        unsup_criterion=unsup_criterion,
        sup_criterion=sup_criterion,
        augment=augment,
        alpha=0.3,
        n_augmentations=K,
    )

    # mixmatch forward pass
    loss_sup, loss_unsup, sup_outputs = mixmatch(
        model,
        unlabeled_sample=u_batch, 
        labeled_sample=d_batch,
    )
    
    # test operations on a CUDA device if it's available
    # in the testing environment
    if torch.cuda.is_available():
        model = model.cuda()
        u_batch['input'] = u_batch['input'].cuda()
        d_batch['input'] = d_batch['input'].cuda()
        d_batch['output'] = d_batch['output'].cuda()
        
        # remove the old teacher because the CPU parameters
        # can't be updated with the CUDA parameters
        mixmatch.teacher = None
    
        loss_sup, loss_unsup, sup_outputs = mixmatch(
            model,
            unlabeled_sample=u_batch, 
            labeled_sample=d_batch,
        )
    
    return


def test_mixmatch_forward_with_confthresh():
    '''Perform a forward pass through the MixMatch objective
    using a dummy model and a pseudolabel confidence threshold.
    
    Notes
    -----
    Does not verify that computations are correct, merely that
    the forward pass completes without error.
    '''
    import scanpy as sc
    import scnym
    from scnym.dataprep import SingleCellDS
    torch.manual_seed(397)
    
    # load 10x human PBMC data as a sample
    adata = sc.datasets.pbmc3k()
    sc.pp.filter_cells(adata, min_counts=100)
    sc.pp.filter_genes(adata, min_cells=100)
    sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e6)
    sc.pp.log1p(adata)
    
    # generate fake class labels
    adata.obs['class'] = np.random.randint(
        0,
        3,
        adata.shape[0],
    )
    
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
    
    # draw a batch from each
    d_batch = next(iter(d_dl))
    u_batch = next(iter(u_dl))
    
    # initialize mixmatch using a dummy model
    unsup_criterion = torch.nn.MSELoss(reduction='none')
    sup_criterion = scnym.trainer.cross_entropy
    
    augment = scnym.dataprep.AUGMENTATION_SCHEMES['None']
    
    model = scnym.model.CellTypeCLF(
        n_layers=1,
        n_hidden=128,
        residual=False,
        init_dropout=0.0,
        n_genes=adata.shape[1],
        n_cell_types=3,
    )
    
    K = 1
    mixmatch = scnym.trainer.MixMatchLoss(
        unsup_criterion=unsup_criterion,
        sup_criterion=sup_criterion,
        augment=augment,
        alpha=0.3,
        n_augmentations=K,
        T=1.,
        pseudolabel_min_confidence=0.001,
    )

    # mixmatch forward pass
    loss_sup, loss_unsup, sup_outputs = mixmatch(
        model,
        unlabeled_sample=u_batch, 
        labeled_sample=d_batch,
    )
    ls_low_thresh = loss_sup.detach().item()
    lu_low_thresh = loss_unsup.detach().item()
    
    # double check that our running confidence scores
    # are working
    assert mixmatch.running_confidence_scores[0][0].size(0) == 16
    assert torch.sum(
        torch.cat(
            [x[0] for x in mixmatch.running_confidence_scores], 
            dim=0,
        ),
    ) > 0
    
    mixmatch = scnym.trainer.MixMatchLoss(
        unsup_criterion=unsup_criterion,
        sup_criterion=sup_criterion,
        augment=augment,
        alpha=0.3,
        n_augmentations=K,
        T=1.,
        pseudolabel_min_confidence=0.4,
    )

    # mixmatch forward pass
    loss_sup, loss_unsup, sup_outputs = mixmatch(
        model,
        unlabeled_sample=u_batch, 
        labeled_sample=d_batch,
    )
    lu_high_thresh = loss_unsup.detach().item()
    ls_high_thresh = loss_sup.detach().item()
    
    # assert that unsup losses decrease when we set higher thresholds
    print(f'Low  threshold unsup loss: {lu_low_thresh}')
    print(f'Low  threshold sup.  loss: {ls_low_thresh}')
    print(f'High threshold unsup loss: {lu_high_thresh}')
    print(f'High threshold sup.  loss: {ls_high_thresh}')    
    assert lu_high_thresh < lu_low_thresh
    # check that non-confident labels are being zero-ed out
    K = 1
    mixmatch = scnym.trainer.MixMatchLoss(
        unsup_criterion=unsup_criterion,
        sup_criterion=sup_criterion,
        augment=augment,
        alpha=0.3,
        n_augmentations=K,
        T=1.,
        pseudolabel_min_confidence=1.0,
    )

    # mixmatch forward pass
    loss_sup, loss_unsup, sup_outputs = mixmatch(
        model,
        unlabeled_sample=u_batch, 
        labeled_sample=d_batch,
    )
    msg = f'{loss_unsup.detach().item()} was not == 0.'
    assert loss_unsup.detach().item() == 0.
    print(f'Loss with confidence threshold == 1. was {loss_unsup.detach().item()}!')
            
    
    # test operations on a CUDA device if it's available
    # in the testing environment
    if torch.cuda.is_available():
        model = model.cuda()
        u_batch['input'] = u_batch['input'].cuda()
        d_batch['input'] = d_batch['input'].cuda()
        d_batch['output'] = d_batch['output'].cuda()
        
        # remove the old teacher because the CPU parameters
        # can't be updated with the CUDA parameters
        mixmatch.teacher = None        
    
        loss_sup, loss_unsup, sup_outputs = mixmatch(
            model,
            unlabeled_sample=u_batch, 
            labeled_sample=d_batch,
        )
    
    return


def test_mixmatch_forward_with_teacher_bn_runnning_stats():
    '''Perform a forward pass through the MixMatch objective
    using a dummy model and resampling augmentations
    
    Notes
    -----
    Does not verify that computations are correct, merely that
    the forward pass completes without error.
    '''
    
    import scanpy as sc
    import scnym
    from scnym.dataprep import SingleCellDS
    
    # load 10x human PBMC data as a sample
    adata = sc.datasets.pbmc3k()
    sc.pp.filter_cells(adata, min_counts=100)
    sc.pp.filter_genes(adata, min_cells=100)
    sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e6)
    sc.pp.log1p(adata)
    
    # generate fake class labels
    adata.obs['class'] = np.random.randint(
        0,
        3,
        adata.shape[0],
    )
    
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
    
    # draw a batch from each
    d_batch = next(iter(d_dl))
    u_batch = next(iter(u_dl))
    
    # initialize mixmatch using a dummy model
    unsup_criterion = torch.nn.MSELoss(reduction='none')
    sup_criterion = scnym.trainer.cross_entropy
    
    augment = scnym.dataprep.AUGMENTATION_SCHEMES['log1p_drop']
    
    model = scnym.model.CellTypeCLF(
        n_layers=1,
        n_hidden=128,
        residual=False,
        init_dropout=0.0,
        n_genes=adata.shape[1],
        n_cell_types=3,
    )
    
    K = 1
    mixmatch = scnym.trainer.MixMatchLoss(
        unsup_criterion=unsup_criterion,
        sup_criterion=sup_criterion,
        augment=augment,
        alpha=0.3,
        n_augmentations=K,
        teacher_bn_running_stats=False,
    )

    # mixmatch forward pass
    loss_sup, loss_unsup, sup_outputs = mixmatch(
        model,
        unlabeled_sample=u_batch, 
        labeled_sample=d_batch,
    )
    
    # test operations on a CUDA device if it's available
    # in the testing environment
    if torch.cuda.is_available():
        model = model.cuda()
        u_batch['input'] = u_batch['input'].cuda()
        d_batch['input'] = d_batch['input'].cuda()
        d_batch['output'] = d_batch['output'].cuda()
        
        # remove the old teacher because the CPU parameters
        # can't be updated with the CUDA parameters
        mixmatch.teacher = None
    
        loss_sup, loss_unsup, sup_outputs = mixmatch(
            model,
            unlabeled_sample=u_batch, 
            labeled_sample=d_batch,
        )
    
    return


def test_train_mixmatch():
    '''Train a few iterations of a mixmatch loss to ensure we're
    passing gradients properly'''
    import scanpy as sc
    import scnym
    from scnym.dataprep import SingleCellDS
    
    # load 10x human PBMC data as a sample
    adata = sc.datasets.pbmc3k()
    sc.pp.filter_cells(adata, min_counts=100)
    sc.pp.filter_genes(adata, min_cells=100)
    sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e6)
    sc.pp.log1p(adata)
    
    # generate fake class labels
    adata.obs['class'] = np.random.randint(
        0,
        3,
        adata.shape[0],
    )
    
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
    
    # draw a batch from each
    d_batch = next(iter(d_dl))
    u_batch = next(iter(u_dl))
    
    # initialize mixmatch using a dummy model
    unsup_criterion = torch.nn.MSELoss(reduction='none')
    sup_criterion = scnym.trainer.cross_entropy
    
    augment = scnym.dataprep.AUGMENTATION_SCHEMES['None']
    
    model = scnym.model.CellTypeCLF(
        n_layers=1,
        n_hidden=128,
        residual=False,
        init_dropout=0.0,
        n_genes=adata.shape[1],
        n_cell_types=3,
    )
    
    K = 1
    mixmatch = scnym.trainer.MixMatchLoss(
        unsup_criterion=unsup_criterion,
        sup_criterion=sup_criterion,
        augment=augment,
        alpha=0.3,
        n_augmentations=K,
        T=1.,
        pseudolabel_min_confidence=0.001,
    )

    # mixmatch forward pass
    loss_sup, loss_unsup, sup_outputs = mixmatch(
        model,
        unlabeled_sample=u_batch, 
        labeled_sample=d_batch,
    )
    init_sl = loss_sup.detach().item()
        
    # train some iterations to reduce sup loss
    optimizer = torch.optim.Adam(model.parameters())
    for i in range(50):
        optimizer.zero_grad()
        loss_sup, loss_unsup, sup_outputs = mixmatch(
            model,
            unlabeled_sample=u_batch, 
            labeled_sample=d_batch,
        )
        loss_sup.backward()
        
        # check that no gradients are passed to the teacher
        teacher_grad = [
            p.grad is None for p in mixmatch.teacher.parameters()
        ]
        assert sum(teacher_grad) == len(teacher_grad)
        
        optimizer.step()
    final_sl = loss_sup.detach().item()
    assert final_sl < init_sl
    
    # train some iterations to reduce unsup loss
    model = scnym.model.CellTypeCLF(
        n_layers=1,
        n_hidden=128,
        residual=False,
        init_dropout=0.0,
        n_genes=adata.shape[1],
        n_cell_types=3,
    )
    loss_sup, loss_unsup, sup_outputs = mixmatch(
        model,
        unlabeled_sample=u_batch, 
        labeled_sample=d_batch,
    )
    init_ul = loss_unsup.detach().item()    
    
    optimizer = torch.optim.Adam(model.parameters())
    for i in range(50):
        optimizer.zero_grad()
        loss_sup, loss_unsup, sup_outputs = mixmatch(
            model,
            unlabeled_sample=u_batch, 
            labeled_sample=d_batch,
        )
        loss_unsup.backward()
        optimizer.step()
    final_ul = loss_unsup.detach().item()
    assert final_ul < init_ul    
    
    return


def test_sharpen_labels():
    '''Test that label sharpening reduces entropy'''
    from scnym.trainer import sharpen_labels
    
    p = torch.FloatTensor([[0.1, 0.2, 0.7]])
    q = sharpen_labels(p, T=0.5)
    
    H = lambda p: -torch.sum(p * torch.log(p))
    
    assert H(p) > H(q)
    
    return


def test_sharpen_labels_t0():
    '''Test that label sharpening returns argmax one-hot
    labels when temperature is == 0.'''
    from scnym.trainer import sharpen_labels
    
    p = torch.FloatTensor([[0.1, 0.2, 0.7]])
    q = sharpen_labels(p, T=0.0)
    
    assert torch.all(
        q == torch.FloatTensor([0, 0, 1])
    )
    
    return


def test_weight_schedule():
    '''Test semi-supervised weight scheduling'''
    from scnym.trainer import ICLWeight
    
    # test ramping
    w = ICLWeight(
        ramp_epochs = 100,
        sigmoid = False,
        max_unsup_weight = 100.,
    )
    assert w(0) == 0.
    assert w(50) > 0.
    assert w(50) < 100.
    assert w(100) == 100.
    assert w(150) == 100.

    w = ICLWeight(
        ramp_epochs = 100,
        sigmoid = True,
        max_unsup_weight = 100.,
    )
    assert w(0) == np.exp(-5)*100.
    assert w(50) > 0.
    assert w(50) < 100.
    assert w(100) == 100.
    assert w(150) == 100.
    
    # test no ramp
    w = ICLWeight(
        ramp_epochs = 0,
        sigmoid = False,
        max_unsup_weight = 1.,
    )
    assert w(0) == 1.
    assert w(5) == 1.
    
    assert 0. < w(3)
    assert 1. >= w(3)
    
    w = ICLWeight(
        ramp_epochs = 0,
        sigmoid = True,
        max_unsup_weight = 1.,
    )
    assert w(0) == 1.
    assert w(5) == 1.    
    
    # test burn in
    w = ICLWeight(
        burn_in_epochs = 5,
        ramp_epochs = 5,
        sigmoid = False,
        max_unsup_weight = 1.,
    )
    assert w(0) == 0.
    assert w(4) == 0.
    assert w(10) == 1.
    
    # check that we only accept integer epochs
    with pytest.raises(TypeError):
        w('a')
    
    return


def main():
    test_samplemixup()
    test_mixmatch_forward_with_confthresh()
    return


if __name__ == '__main__':
    main()