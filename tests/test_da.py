'''Test domain adaptation tools'''
import pytest
import os
import os.path as osp
import sys
import numpy as np
import torch

sys.path.append('../')


def test_dan_pseudoconf():
    '''Test pseudolabel confidence mechanism for DAN'''
    import scnym
    import copy
    torch.manual_seed(1)
    np.random.seed(1)    
    
    model = scnym.model.CellTypeCLF(
        n_genes=3,
        n_cell_types=2,
        residual=False,
        init_dropout=0.,
        n_layers=1,
        n_hidden=16,
    )
    
    # create fake data from cell type A
    A = torch.randn(size=(32, 3))
    # create fake data from cell type B
    B = torch.randn(size=(32, 3)) + 5
    
    # create datasets and loaders
    # label cell types A/B with [0, 1]
    X_0 = torch.cat([A[:16], B[:16]], dim=0)
    y_0 = torch.zeros(32)
    y_0[16:] = 1
    
    
    # shift X_1 into a new domain
    X_1 = torch.cat([A[16:], B[16:]], dim=0)
    y_1 = torch.zeros(32)
    y_1[16:] = 1
    
    X_1 = X_1 + 10.
    
    # pack into samples
    labeled_sample = {
        'input': X_0,
        'output': y_0.long(),
    }
    unlabeled_sample = {
        'input': X_1,
        'output': y_1.long(),
    }
    
    print('Fake data:')
    print(labeled_sample['input'][:5])
    print(unlabeled_sample['input'][:5])
    
    
    # setup the DAN loss
    dan = scnym.trainer.DANLoss(
        model=model,
        dan_criterion=scnym.trainer.cross_entropy,
        use_conf_pseudolabels=True,
    )
    
    # get pseudolabel confidence
    pseudolabel_min_confidence = 0.9    
    
    model.eval()
    u_out = model(unlabeled_sample['input'])
    u_sm  = torch.nn.functional.softmax(u_out, dim=1)
    u_hc, u_pred = torch.max(u_sm, dim=1)
    
    pseudolabel_confidence = u_hc > pseudolabel_min_confidence
    print(f'# of conf pseudolabels before training: {torch.sum(pseudolabel_confidence)}')
    
    # compute dan loss
    l = dan(
        labeled_sample=labeled_sample,
        unlabeled_sample=unlabeled_sample,
        weight=0.,
        pseudolabel_confidence=pseudolabel_confidence,
    )
    init_l = l.detach().item()
    # how many examples had confident pseudolabels?
    init_n_conf = dan.n_conf_pseudolabels
    
    # we train the model embedding for several iterations
    # to ensure that the pseudolabel confidence increases
    # over time and that this is reflected in the DANN
    optimizer = torch.optim.AdamW(
        [
            {'params':model.parameters()},
        ],
        weight_decay=1e-4,
    )
    
    print('Optimizing model to increase pseudolabel conf...')
    ce_loss = torch.nn.CrossEntropyLoss()
    losses = []
    model.train()
    for i in range(50):
        optimizer.zero_grad()
        
        x_out = model(labeled_sample['input'])
        l = ce_loss(x_out, labeled_sample['output'])
        l.backward()
        
        optimizer.step()
        
        losses.append(l.detach().cpu().item())
    print('Supervised classifier losses:')
    print(f'Initial: {losses[0]}')
    print(f'Final  : {losses[-1]}')
    assert losses[-1] < losses[0]
    
    # check that the pseudolabel confidence is now increased
    model.eval()
    u_out = model(unlabeled_sample['input'])
    u_sm  = torch.nn.functional.softmax(u_out, dim=1)
    u_hc, u_pred = torch.max(u_sm, dim=1)    
    pseudolabel_confidence = u_hc > pseudolabel_min_confidence
    
    l = dan(
        labeled_sample=labeled_sample,
        unlabeled_sample=unlabeled_sample,
        weight=0.,
        pseudolabel_confidence=pseudolabel_confidence,
    )
    final_l = l.detach().item()
    # how many examples had confident pseudolabels?
    final_n_conf = dan.n_conf_pseudolabels
    
    print(f'Initial conf pseudolabels: {init_n_conf}')
    print(f'Final   conf pseudolabels: {final_n_conf}')
    assert final_n_conf > init_n_conf
    
    ######################################################
    # test training the DAN with pseudolabel confidence
    ######################################################
    
    # reset the model parameters
    model = scnym.model.CellTypeCLF(
        n_genes=3,
        n_cell_types=2,
        residual=False,
        init_dropout=0.,
        n_layers=1,
        n_hidden=16,
    )
    # setup a teacher model
    teacher = copy.deepcopy(model)
    teacher.eval()    
    
    dan = scnym.trainer.DANLoss(
        model=model,
        dan_criterion=scnym.trainer.cross_entropy,
        use_conf_pseudolabels=True,
    )
    
    # train the DAN and the model concurrently, but do not pass
    # DAN weigths to the classifier
    # both classification and DAN losses should *decrease*
    print('Optimizing model and DAN...')
    ce_losses = []
    dan_losses = []
    
    optimizer = torch.optim.AdamW(
        [
            {'params':model.parameters()},
            {'params':dan.dann.domain_clf.parameters()},
        ],
        weight_decay=1e-4,
    )
    
    pseudolabel_min_confidence = 0.8
    for i in range(200):
        optimizer.zero_grad()
        
        x_out = model(labeled_sample['input'])
        ce_l = ce_loss(x_out, labeled_sample['output'])
        
        # generate pseudolabels
        teacher.load_state_dict(model.state_dict())
        teacher.eval()
        u_out = teacher(unlabeled_sample['input'])
        u_sm  = torch.nn.functional.softmax(u_out, dim=1)
        u_hc, u_pred = torch.max(u_sm, dim=1)
        pseudolabel_confidence = u_hc > pseudolabel_min_confidence    
        
        dan_l = dan(
            labeled_sample=labeled_sample,
            unlabeled_sample=unlabeled_sample,
            weight=0.,
            pseudolabel_confidence=pseudolabel_confidence,
        )
        
        l = ce_l + dan_l
        l.backward()
        
        ce_losses.append(ce_l.detach().cpu().item())
        dan_losses.append(dan_l.detach().cpu().item())        
        
        # check that gradients are flowing to the model
        model_grad_mag = 0.
        for p in model.classif.parameters():
            if p.grad is not None:
                model_grad_mag += (p.grad**2).sum()
        assert model_grad_mag > 0.
        
        # check that gradients flow to the DANN domain classifier
        dan_grad_mag = 0.
        for p in dan.dann.domain_clf.parameters():
            if p.grad is not None:
                dan_grad_mag += (p.grad**2).sum()
        assert dan_grad_mag > 0.
        if i == 0 or i == 49:
            print(f'Model grad mag: {model_grad_mag}')
            print(f'DANN  grad mag: {dan_grad_mag}')
        
        optimizer.step()
        
        losses.append(l.detach().cpu().item())
        
    # check that losses decreased
    print(f'Initial CE  loss: {ce_losses[0]}')
    print(f'Final   CE  loss: {ce_losses[-1]}')

    print(f'Initial DAN loss: {dan_losses[0]}')
    print(f'Final   DAN loss: {dan_losses[-1]}')
    
    assert ce_losses[-1] < ce_losses[0]
    assert dan_losses[-1] < dan_losses[0]
    
    return


def test_dan_train():
    '''Test training a DANN on fake data that is easy
    to discriminate from three fake domains'''
    import scnym
    torch.manual_seed(1)
    np.random.seed(1)    
        
    model = scnym.model.CellTypeCLF(
        n_genes=3,
        n_cell_types=2,
        residual=False,
        init_dropout=0.,
        n_layers=1,
        n_hidden=16,
    )
    
    # create fake data from cell type A
    A = torch.randn(size=(48, 3))
    # create fake data from cell type B
    B = torch.randn(size=(48, 3)) + 5
    
    # setup the DAN loss
    dan = scnym.trainer.DANLoss(
        model=model,
        dan_criterion=scnym.trainer.cross_entropy,
        n_domains = 3,
    )
    
    # create datasets and loaders
    # label cell types A/B with [0, 1]
    X_0 = torch.cat([A[:16], B[:16]], dim=0)
    y_0 = torch.zeros(32)
    y_0[16:] = 1
    
    
    # shift X_1 into a new domain 1
    X_1 = torch.cat([A[16:32], B[16:32]], dim=0)
    y_1 = torch.zeros(32)
    y_1[16:] = 1
    
    X_1 = X_1 + 1000.
    
    # shift X_2 into new domain 2
    X_2 = torch.cat([A[32:], B[32:]], dim=0)
    y_2 = torch.zeros(32)
    y_2[16:] = 1
    
    X_2 = X_2 - 1000.
    
    # pack into samples
    labeled_domain = torch.cat(
        [torch.zeros(32), torch.ones(32)],
    ).long()
    labeled_domain = torch.nn.functional.one_hot(
        labeled_domain, 
        num_classes=3,
    )
    labeled_sample = {
        'input': torch.cat([X_0, X_2], 0),
        'output': torch.cat([y_0, y_2], 0),
        'domain': labeled_domain,
    }
    unlabeled_domain = (torch.zeros(32) + 2).long()
    unlabeled_domain = torch.nn.functional.one_hot(unlabeled_domain, num_classes=3)
    unlabeled_sample = {
        'input': X_1,
        'output': y_1,
        'domain': unlabeled_domain,
    }
    
    print('Fake data:')
    print(labeled_sample['input'][:5])
    print(unlabeled_sample['input'][:5])
    print('Fake domains:')
    print(labeled_sample['domain'][:5])
    print(unlabeled_sample['domain'][:5])    
    
    # compute dan loss
    l = dan(
        labeled_sample=labeled_sample,
        unlabeled_sample=unlabeled_sample,
        weight=0.,
    )
    init_l = l.detach().item()
    
    # run a few backprop iterations to make sure
    # we can decrease the loss and train the DANN
    optimizer = torch.optim.AdamW(
        [
            {'params':model.parameters()},
            {'params': dan.dann.domain_clf.parameters()},
        ],
        weight_decay=1e-4,
    )
    
    # we scale the reverse gradients passed to the embedding down
    # to 0, so the loss should decrease as we train
    print('Optimizing...')    
    losses = []    
    for i in range(50):
        optimizer.zero_grad()
        l = dan(
            labeled_sample=labeled_sample,
            unlabeled_sample=unlabeled_sample,
            weight=0.,
        )
        l.backward()
        
        # check that gradients are being cut off before
        # flowing back to the model
        model_grad_mag = 0.
        for p in model.classif.parameters():
            if p.grad is not None:
                model_grad_mag += (p.grad**2).sum()
        assert model_grad_mag == 0.
        
        # check that gradients flow to the DANN domain classifier
        dan_grad_mag = 0.
        for p in dan.dann.domain_clf.parameters():
            if p.grad is not None:
                dan_grad_mag += (p.grad**2).sum()
        assert dan_grad_mag > 0.
        if i == 0 or i == 49:
            print(f'Model grad mag: {model_grad_mag}')
            print(f'DANN  grad mag: {dan_grad_mag}')
        
        optimizer.step()
        
        losses.append(l.detach().cpu().item())
        
    final_l = l.detach().item()
    
    if not (init_l > final_l):
        msg = f'Losses did not decrease for domain clf train:\n{losses}'
        raise ValueError(msg)
    else:
        print('Decreasing DA losses by domain clf train')
        print(losses)
        
    # now make sure we are increasing the losses when we only train the
    # embedding model, rather than the DANN itself.
    optimizer = torch.optim.AdamW(
        [
            {'params':model.parameters()},
        ],
        weight_decay=1e-4,
    )
    losses = []

    # get a copy of the DANN parameters to show that 
    # they aren't changing as we optimize the model embedding
    dann_p_t0 = list(dan.dann.domain_clf.parameters())[0]
    for i in range(50):
        optimizer.zero_grad()
        l = dan(
            labeled_sample=labeled_sample,
            unlabeled_sample=unlabeled_sample,
            weight=1.,
        )
        l.backward()
        optimizer.step()
        
        # check that gradients are flowing back to the model
        model_grad_mag = 0.
        for p in model.classif.parameters():
            if p.grad is not None:
                model_grad_mag += (p.grad**2).sum()
        assert model_grad_mag > 0.
        
        # check that gradients are going to the DANN
        # classifier, even if we're not updating them
        dan_grad_mag = 0.
        for p in dan.dann.domain_clf.parameters():
            if p.grad is not None:
                dan_grad_mag += (p.grad**2).sum()
        assert dan_grad_mag > 0.
        if i == 0 or i == 49:
            print(f'Model grad mag: {model_grad_mag}')
            print(f'DANN  grad mag: {dan_grad_mag}')
        
        # check that DANN parameters aren't changing
        dann_p_t1 = list(dan.dann.domain_clf.parameters())[0]
        assert torch.all(dann_p_t0 == dann_p_t1)
        dann_p_t0 = dann_p_t1
        
        losses.append(l.detach().cpu().item())  
        
    if not (losses[0] < losses[-1]):
        msg = f'Losses did not increase for +ive loss:\n{losses}'
        raise ValueError(msg)
    else:
        print('Increasing DA losses with +ive term')
        print(losses)        
        
    return


if __name__ == '__main__':
    #test_dan_pseudoconf()
    #print()
    test_dan_train()