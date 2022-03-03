"""Test guided latent variables using structured sparsity penalties"""

import logging
import pytest
import os
import os.path as osp
import shutil
import tempfile
import sys
import numpy as np
import torch

sys.path = ["../../"] + sys.path


def _get_gene_sets() -> dict:
    """Download an example group of gene sets as a dictionary"""
    from urllib.request import urlretrieve
    HALLMARK_PATH = "https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=MSigDB_Hallmark_2020"
    urlretrieve(HALLMARK_PATH, "./hallmark.gmt")
    gene_sets = {}
    with open("./hallmark.gmt", "r") as f:
        for line in f:
            values = line.split("\t")
            gene_sets[values[0]] = [x for x in values[1:] if x not in ("", " ", "\n")]
    return gene_sets


def _filter_gene_sets(gene_sets, adata) -> dict:
    for k in gene_sets.keys():
        gene_sets[k] = [x for x in gene_sets[k] if x in adata.var_names]
    return gene_sets    


def cuda_only(func):
    """Decorator that sets a test to run only if CUDA is available"""

    def wrap_cuda_only():
        if not torch.cuda.is_available():
            print("Test only run if CUDA compute device is available.")
            return
        else:
            func()

    return wrap_cuda_only


def test_gene_set_to_priors():
    """Test converting a set of gene sets to
    a prior matrix"""
    import scnym
    import scanpy as sc

    torch.manual_seed(1)
    np.random.seed(1)

    # load 10x human PBMC data as a sample
    adata = sc.datasets.pbmc3k()

    # load sample gene sets
    gene_sets = _get_gene_sets()
    gene_sets = _filter_gene_sets(gene_sets, adata)

    print("loaded gene sets")
    print()
    print(gene_sets)
    print()

    SS = scnym.trainer.StructuredSparsity(
        n_genes=adata.shape[1],
        n_hidden=256,
        gene_sets=gene_sets,
        gene_names=np.array(adata.var_names),
    )

    print("Matrix")
    print(SS.prior_matrix[:5, :5])
    print("Program names")
    print(SS.gene_set_names[:5])
    print("Myogenesis idx")
    print(SS.gene_set_names.index("Myogenesis"))
    print("Myogenesis var:")
    print(SS.prior_matrix[SS.gene_set_names.index("Myogenesis"), :].sum())

    # check that variables have the same number of `True` elements as the
    # number of genes in the gene set

    for i, k in enumerate(SS.gene_set_names):
        n_in_program = int(SS.prior_matrix[i, :].sum())
        n_in_set = len(gene_sets[k])

        assert n_in_program == n_in_set, f"{n_in_program} != {n_in_set} for {k}"

    # check that the correct genes are in each program
    for i, k in enumerate(SS.gene_set_names):
        gene_idx = np.array([
            SS.gene_names.index(x) for x in SS.gene_sets[k] if x in SS.gene_names
        ])
        prog_idx = np.where(SS.prior_matrix[i].numpy().flatten())[0]

        itxn_idx = np.intersect1d(gene_idx, prog_idx)
        msg = f"itxn {len(itxn_idx)} | gene {len(gene_idx)} | prog {len(prog_idx)}"
        assert len(itxn_idx) == len(gene_idx) == len(prog_idx), msg

    return


def test_ss_in_and_out_of_set():
    """Confirm that the structured sparsity loss only influences genes that are
    outside the specified gene set"""
    import scnym
    import torch
    import scanpy as sc

    torch.manual_seed(1)
    np.random.seed(1)

    # load 10x human PBMC data as a sample
    adata = sc.datasets.pbmc3k()

    # load sample gene sets
    gene_sets = _get_gene_sets()
    gene_sets = _filter_gene_sets(gene_sets, adata)

    # prepare model and loss
    model = scnym.model.CellTypeCLF(
        n_genes=adata.shape[1],
        n_cell_types=10,
        n_hidden=256,
        n_hidden_init=len(gene_sets),
        hidden_init_dropout=True,
    )

    SS = scnym.trainer.StructuredSparsity(
        n_genes=adata.shape[1],
        n_hidden=len(gene_sets),
        gene_sets=gene_sets,
        gene_names=np.array(adata.var_names),
    )

    # compute the initial loss before perturbing `W`
    loss_0 = SS(model,)
    print("loss before perturbation", loss_0)

    # add weight **within** the gene set
    # this shouldn't change the loss!
    with torch.no_grad():
        # get the weights
        W = dict(model.named_parameters())["embed.0.weight"]
        # get the index of a gene in a gene set
        gene2perturb = gene_sets[SS.gene_set_names[0]][0]
        geneidx2perturb = SS.gene_names.index(gene2perturb)
        W[0, geneidx2perturb] += 10000
    loss_1 = SS(model,)
    print("loss after in-set perturbation", loss_1)
    assert loss_1 == loss_0

    with torch.no_grad():
        # get the weights
        W = dict(model.named_parameters())["embed.0.weight"]
        # get the index of a gene in a gene set
        in_set_idx = [
            SS.gene_names.index(gene_sets[SS.gene_set_names[0]][i])
            for i in range(len(gene_sets[SS.gene_set_names[0]]))
        ]
        out_set_idx = np.setdiff1d(
            np.arange(len(gene_sets[SS.gene_set_names[0]])),
            in_set_idx,
        ).astype(np.int32)
        W[0, out_set_idx] += 1000
    loss_2 = SS(model,)
    print("loss after out-set perturbation", loss_2)
    assert loss_2 > loss_0    
    return



def test_sparsity_loss():
    """Test a forward and backward pass of the sparsity loss"""
    import scnym
    import copy
    from urllib.request import urlretrieve
    import scanpy as sc
    import pandas as pd
    import logging

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    logging.debug("Test debug statement")

    torch.manual_seed(1)
    np.random.seed(1)

    # load 10x human PBMC data as a sample
    adata = sc.datasets.pbmc3k()
    sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e6)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
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
    dl = torch.utils.data.DataLoader(ds, batch_size=256)

    # load sample gene sets
    HALLMARK_PATH = "https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=MSigDB_Hallmark_2020"
    tmpdir = tempfile.mkdtemp()
    urlretrieve(
        HALLMARK_PATH,
        osp.join(tmpdir, "hallmark.gmt"),
    )
    gene_sets = {}
    with open(osp.join(tmpdir, "hallmark.gmt"), "r") as f:
        for line in f:
            values = line.split("\t")
            gene_sets[values[0]] = [x for x in values[1:] if x not in ("", " ", "\n")]
    shutil.rmtree(tmpdir)

    for k in gene_sets.keys():
        gene_sets[k] = [x for x in gene_sets[k] if x in adata.var_names]
    print(f"n_gene_sets: {len(gene_sets.keys())}")

    # init model
    n_hidden_init = len(gene_sets.keys()) + 4
    print(f"n_hidden_init: {n_hidden_init}")
    model = scnym.model.CellTypeCLF(
        n_genes=adata.shape[1],
        n_cell_types=len(np.unique(y)),
        n_hidden=256,
        init_dropout=0.0,
        n_hidden_init=n_hidden_init,
    )

    # init sparsity loss
    SS = scnym.trainer.StructuredSparsity(
        n_genes=adata.shape[1],
        n_hidden=n_hidden_init,
        gene_sets=gene_sets,
        gene_names=np.array(adata.var_names),
    )
    print(SS.prior_matrix.size())
    W = dict(model.named_parameters())["embed.0.weight"]
    print(W.size())
    # forward pass
    loss = SS(model)
    print(f"loss:\t{loss}")

    # forward pass with smaller gene sets list
    gene_sets_small = {k: v for k, v in list(gene_sets.items())[:5]}
    SS = scnym.trainer.StructuredSparsity(
        n_genes=adata.shape[1],
        n_hidden=n_hidden_init,
        gene_sets=gene_sets_small,
        gene_names=np.array(adata.var_names),
    )
    loss_2 = SS(model)
    print(f"loss:\t{loss_2}")
    # smaller gene sets should give us a bigger loss value
    assert loss_2.item() > loss.item(), "smaller gene set didn't yield bigger loss"

    # manually set the prior matrix to 1. everywhere to
    # check that we get 0. loss
    SS.prior_matrix[:, :] = True
    loss = SS(model)
    print(f"loss:\t{loss}")
    assert float(loss.item()) == 0.0, "loss was not 0.0 with empty prior"

    # train for a few epochs and try to reduce losses
    optimizer = torch.optim.Adam(
        model.parameters(),
    )
    CE = scnym.trainer.cross_entropy
    SS = scnym.trainer.StructuredSparsity(
        n_genes=adata.shape[1],
        n_hidden=n_hidden_init,
        gene_sets=gene_sets,
        gene_names=np.array(adata.var_names),
        n_dense_latent=4,
    )

    if torch.cuda.is_available():
        model = model.cuda()
    model_device = list(model.parameters())[0].device

    weight = 1e-3
    init_loss = None
    for epoch in range(40):

        for data in dl:
            input_ = data["input"].to(device=model_device)
            target = data["output"].to(device=model_device)
            outputs = model(input_)

            l_ce = CE(outputs, target)
            l_ss = SS(model)

            loss = l_ce + weight * l_ss

            if init_loss is None:
                init_loss = loss.detach().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 2 == 0:
            print(f"epoch\t{epoch}")
            print(f"CE\t{l_ce}")
            print(f"SS\t{l_ss}")
            print(f"L\t{loss}")

    final_loss = loss.detach().item()
    print(f"Initial loss: {init_loss}")
    print(f"Final loss  : {final_loss}")
    assert final_loss < init_loss, "final loss was bigger than initial"

    # check that values of specified genes in a gene
    # program are higher than other values
    infl_idx = SS.gene_set_names.index("Inflammatory Response")
    W = dict(model.named_parameters())["embed.0.weight"]

    P_j = SS.prior_matrix[infl_idx, :]
    W_j = W[infl_idx, :]

    W_j_inset = W_j[P_j]
    W_j_outset = W_j[~P_j]

    print("W_j in set:")
    print(W_j_inset)
    print("W_j out of set:")
    print(W_j_outset)

    print("Means:")
    print(f"Inset genes : {W_j_inset.mean()}")
    print(f"Outset genes: {W_j_outset.mean()}")

    print("Absolute value Means:")
    print(f"Inset genes : {W_j_inset.abs().mean()}")
    print(f"Outset genes: {W_j_outset.abs().mean()}")

    msg = "in set mean weight was less than out of set mean weight"
    assert W_j_inset.abs().mean() > W_j_outset.abs().mean(), msg

    return


@cuda_only
def test_nonneg_guide():
    import scnym
    import copy
    from urllib.request import urlretrieve
    import scanpy as sc
    import pandas as pd
    import logging

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    logging.debug("Test debug statement")

    torch.manual_seed(1)
    np.random.seed(1)

    # load 10x human PBMC data as a sample
    adata = sc.datasets.pbmc3k()
    sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e6)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
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
    dl = torch.utils.data.DataLoader(ds, batch_size=256)

    # load sample gene sets
    HALLMARK_PATH = "https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=MSigDB_Hallmark_2020"
    tmpdir = tempfile.mkdtemp()
    urlretrieve(
        HALLMARK_PATH,
        osp.join(tmpdir, "hallmark.gmt"),
    )
    gene_sets = {}
    with open(osp.join(tmpdir, "hallmark.gmt"), "r") as f:
        for line in f:
            values = line.split("\t")
            gene_sets[values[0]] = [x for x in values[1:] if x not in ("", " ", "\n")]
    shutil.rmtree(tmpdir)

    for k in gene_sets.keys():
        gene_sets[k] = [x for x in gene_sets[k] if x in adata.var_names]
    print(f"n_gene_sets: {len(gene_sets.keys())}")

    # init model
    n_hidden_init = len(gene_sets.keys()) + 4
    print(f"n_hidden_init: {n_hidden_init}")
    model = scnym.model.CellTypeCLF(
        n_genes=adata.shape[1],
        n_cell_types=len(np.unique(y)),
        n_hidden=256,
        init_dropout=0.0,
        n_hidden_init=n_hidden_init,
    )

    # train for a few epochs and try to reduce losses
    optimizer = torch.optim.Adam(
        model.parameters(),
    )
    CE = scnym.trainer.cross_entropy
    SS = scnym.trainer.StructuredSparsity(
        n_genes=adata.shape[1],
        n_hidden=n_hidden_init,
        gene_sets=gene_sets,
        gene_names=np.array(adata.var_names),
        n_dense_latent=4,
        nonnegative=True,
    )

    if torch.cuda.is_available():
        model = model.cuda()
    model_device = list(model.parameters())[0].device

    weight = 1e-3
    init_loss = None
    for epoch in range(40):

        for data in dl:
            input_ = data["input"].to(device=model_device)
            target = data["output"].to(device=model_device)
            outputs = model(input_)

            l_ce = CE(outputs, target)
            l_ss = SS(model)

            loss = l_ce + weight * l_ss

            if init_loss is None:
                init_loss = loss.detach().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 2 == 0:
            print(f"epoch\t{epoch}")
            print(f"CE\t{l_ce}")
            print(f"SS\t{l_ss}")
            print(f"L\t{loss}")

    final_loss = loss.detach().item()
    print(f"Initial loss: {init_loss}")
    print(f"Final loss  : {final_loss}")
    assert final_loss < init_loss, "final loss was bigger than initial"

    # check that values of specified genes in a gene
    # program are higher than other values
    infl_idx = SS.gene_set_names.index("Inflammatory Response")
    W = dict(model.named_parameters())["embed.0.weight"]

    P_j = SS.prior_matrix[infl_idx, :]
    W_j = W[infl_idx, :]

    W_j_inset = W_j[P_j]
    W_j_outset = W_j[~P_j]

    print("W_j in set:")
    print(W_j_inset)
    print("W_j out of set:")
    print(W_j_outset)

    print("Means:")
    print(f"Inset genes : {W_j_inset.mean()}")
    print(f"Outset genes: {W_j_outset.mean()}")

    print("Absolute value Means:")
    print(f"Inset genes : {W_j_inset.abs().mean()}")
    print(f"Outset genes: {W_j_outset.abs().mean()}")

    msg = "in set mean weight was less than out of set mean weight"
    assert W_j_inset.abs().mean() > W_j_outset.abs().mean(), msg
    return


def test_withingenesetnorm():
    import scnym
    print(scnym.__file__)
    import scanpy as sc
    from urllib.request import urlretrieve

    # load 10x human PBMC data as a sample
    adata = sc.datasets.pbmc3k()

    # load sample gene sets
    HALLMARK_PATH = "https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=MSigDB_Hallmark_2020"
    tmpdir = tempfile.mkdtemp()
    urlretrieve(
        HALLMARK_PATH,
        osp.join(tmpdir, "hallmark.gmt"),
    )
    gene_sets = {}
    with open(osp.join(tmpdir, "hallmark.gmt"), "r") as f:
        for line in f:
            values = line.split("\t")
            gene_sets[values[0]] = [x for x in values[1:] if x not in ("", " ", "\n")]
    shutil.rmtree(tmpdir)

    n_hidden_init = len(gene_sets.keys())
    print(f"n_hidden_init: {n_hidden_init}")
    model = scnym.model.CellTypeCLF(
        n_genes=adata.shape[1],
        n_cell_types=10, # placeholder
        n_hidden=256,
        init_dropout=0.0,
        n_hidden_init=n_hidden_init,
    )

    GSN = scnym.trainer.WithinGeneSetNorm(
        gene_sets=gene_sets,
        gene_names=np.array(adata.var_names),
    )

    ##################################################
    # test forward pass
    ##################################################
    loss_0 = GSN(model=model,)
    print(f"loss_0: {loss_0}")
    # increase some of the within gene weights and ensure the norm goes up
    with torch.no_grad():
        W = dict(model.named_parameters())["embed.0.weight"]
        W[GSN.prior_matrix.bool()] += 16
    loss_1 = GSN(model=model,)
    print("loss_1:", loss_1)
    assert loss_1 >= loss_0
    # out of set genes shouldn't matter
    with torch.no_grad():
        W = dict(model.named_parameters())["embed.0.weight"]
        W[torch.logical_not(GSN.prior_matrix.bool())] += 16
    loss_2 = GSN(model=model,)
    print("loss_2:", loss_2)
    assert loss_1 == loss_2
    return



def test_weight_mask():
    import scnym
    print(scnym.__file__)
    import scanpy as sc
    from urllib.request import urlretrieve

    # check the number of initial positive, negative values
    def get_pos_neg(model):
        W = dict(model.named_parameters())["embed.0.weight"]
        n_pos = torch.sum(W>0.0)
        n_neg = torch.sum(W<0.0)
        return n_pos, n_neg    

    # load 10x human PBMC data as a sample
    adata = sc.datasets.pbmc3k()

    # load sample gene sets
    HALLMARK_PATH = "https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=MSigDB_Hallmark_2020"
    tmpdir = tempfile.mkdtemp()
    urlretrieve(
        HALLMARK_PATH,
        osp.join(tmpdir, "hallmark.gmt"),
    )
    gene_sets = {}
    with open(osp.join(tmpdir, "hallmark.gmt"), "r") as f:
        for line in f:
            values = line.split("\t")
            gene_sets[values[0]] = [x for x in values[1:] if x not in ("", " ", "\n")]
    shutil.rmtree(tmpdir)

    n_hidden_init = len(gene_sets.keys())
    print(f"n_hidden_init: {n_hidden_init}")
    model = scnym.model.CellTypeCLF(
        n_genes=adata.shape[1],
        n_cell_types=10, # placeholder
        n_hidden=256,
        init_dropout=0.0,
        n_hidden_init=n_hidden_init,
    )

    n_pos_init, n_neg_init = get_pos_neg(model)
    print(f"initial pos/neg counts: {n_pos_init}; {n_neg_init}")    

    WM = scnym.trainer.WeightMask(
        model=model,
        gene_sets=gene_sets,
        gene_names=np.array(adata.var_names),
        nonnegative=True,
    )
    n_pos_end, n_neg_end = get_pos_neg(model)
    print(f"final pos/neg counts: {n_pos_end}; {n_neg_end}")
    assert n_pos_init >= n_pos_end
    assert n_neg_init >= n_neg_end    

    ##################################################
    # test forward pass for flipping negative values
    ##################################################
    # set some values to negative first
    with torch.no_grad():
        W = dict(model.named_parameters())["embed.0.weight"]
        idx = np.where(WM.prior_matrix.view(-1).cpu().numpy()>0)[0]
        ridx = np.random.choice(idx, size=30, replace=False)
        W.view(-1)[ridx] = -10
    n_pos_init, n_neg_init = get_pos_neg(model)
    print(f"initial pos/neg counts: {n_pos_init}; {n_neg_init}")    
    

    loss = WM(model=model,)
    print(f"loss: {loss}")
    assert loss == 0.0
    n_pos_end, n_neg_end = get_pos_neg(model)
    print(f"final pos/neg counts: {n_pos_end}; {n_neg_end}")    

    assert n_neg_init >= n_neg_end
    return


def test_weight_mask_train():
    import scnym
    import copy
    from urllib.request import urlretrieve
    import scanpy as sc
    import pandas as pd
    import logging

    # check the number of initial positive, negative values
    def get_pos_neg(model):
        W = dict(model.named_parameters())["embed.0.weight"]
        n_pos = torch.sum(W>0.0)
        n_neg = torch.sum(W<0.0)
        return n_pos, n_neg

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    logging.debug("Test DEBUG statement")
    logging.info("Test INFO statement")

    torch.manual_seed(1)
    np.random.seed(1)

    # load 10x human PBMC data as a sample
    adata = sc.datasets.pbmc3k()
    sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e6)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
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
    dl = torch.utils.data.DataLoader(ds, batch_size=256)

    # load sample gene sets
    HALLMARK_PATH = "https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=MSigDB_Hallmark_2020"
    # ARCHS_TF_PATH = "https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=ARCHS4_TFs_Coexp"
    tmpdir = tempfile.mkdtemp()
    urlretrieve(
        HALLMARK_PATH,
        osp.join(tmpdir, "genes.gmt"),
    )
    gene_sets = {}
    with open(osp.join(tmpdir, "genes.gmt"), "r") as f:
        for line in f:
            values = line.split("\t")
            gene_sets[values[0]] = [x for x in values[1:] if x not in ("", " ", "\n")]
    shutil.rmtree(tmpdir)

    n_hidden_init = len(gene_sets.keys())
    print(f"n_hidden_init: {n_hidden_init}")
    model = scnym.model.CellTypeCLF(
        n_genes=adata.shape[1],
        n_cell_types=len(np.unique(y)),
        n_hidden=256,
        init_dropout=0.0,
        n_hidden_init=n_hidden_init,
    )

    if torch.cuda.is_available():
        model = model.cuda()
    model_device = list(model.parameters())[0].device

    # train for a few epochs and try to reduce losses
    optimizer = torch.optim.Adam(
        model.parameters(),
    )
    CE = scnym.trainer.cross_entropy

    n_pos_init, n_neg_init = get_pos_neg(model)
    print(f"initial pos/neg counts: {n_pos_init}; {n_neg_init}")    

    WM = scnym.trainer.WeightMask(
        model=model,
        gene_sets=gene_sets,
        gene_names=np.array(adata.var_names),
        nonnegative=True,
    )

    n_pos_end, n_neg_end = get_pos_neg(model)
    print(f"post-wrap pos/neg counts: {n_pos_end}; {n_neg_end}")
    assert n_pos_init >= n_pos_end
    assert n_neg_init >= n_neg_end

    weight = 1e-3
    init_loss = None
    for epoch in range(40):

        for data in dl:
            input_ = data["input"].to(device=model_device)
            target = data["output"].to(device=model_device)
            outputs = model(input_)

            l_ce = CE(outputs, target)
            l_wm = WM(model=model,)
            loss = l_ce + l_wm

            if init_loss is None:
                init_loss = loss.detach().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 2 == 0:
            print(f"epoch\t{epoch}")
            print(f"CE\t{l_ce}")
            print(f"WM\t{l_wm}")
            print(f"L\t{loss}")

            print("grad on embed weights:")
            W = dict(model.named_parameters())["embed.0.weight"]
            target_g = W.grad.data[WM.prior_matrix].mean()
            offtarget_g = W.grad.data[torch.logical_not(WM.prior_matrix)].mean()
            print(f"\ttarget/off-target grads: {target_g}; {offtarget_g}")

    final_loss = loss.detach().item()
    print(f"Initial loss: {init_loss}")
    print(f"Final loss  : {final_loss}")
    assert final_loss < init_loss, "final loss was bigger than initial"

    # check that values of specified genes in a gene
    # program are higher than other values
    # set2use = "NFKB1 human tf ARCHS4 coexpression"
    set2use = "Inflammatory Response"
    print(f"Checking weights in program: {set2use}")
    infl_idx = WM.gene_set_names.index(set2use)
    W = dict(model.named_parameters())["embed.0.weight"]

    P_j = WM.prior_matrix[infl_idx, :]
    W_j = W[infl_idx, :]

    W_j_inset = W_j[P_j]
    W_j_outset = W_j[~P_j]

    print("W_j in set:")
    print(W_j_inset)
    print("W_j out of set:")
    print(W_j_outset)

    print("Means:")
    print(f"Inset genes : {W_j_inset.mean()}")
    print(f"Outset genes: {W_j_outset.mean()}")

    print("Absolute value Means:")
    print(f"Inset genes : {W_j_inset.abs().mean()}")
    print(f"Outset genes: {W_j_outset.abs().mean()}")

    msg = "in set mean weight was less than out of set mean weight"
    assert W_j_inset.abs().mean() > W_j_outset.abs().mean(), msg


def test_lv_gene_corr():
    import scnym
    import copy
    from urllib.request import urlretrieve
    import scanpy as sc
    import pandas as pd
    import logging


    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    logging.debug("Test DEBUG statement")
    logging.info("Test INFO statement")

    # load 10x human PBMC data as a sample
    adata = sc.datasets.pbmc3k()
    sc.pp.filter_genes(adata, min_cells=30)
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.log1p(adata)

    X = adata.X
    y = np.random.randint(0, 4, size=adata.shape[0])
    ds = scnym.dataprep.SingleCellDS(
        X=X,
        y=y,
    )
    dl = torch.utils.data.DataLoader(ds, batch_size=256)    

    # load sample TF gene sets
    ARCHS_TF_PATH = "https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=ARCHS4_TFs_Coexp"
    tmpdir = tempfile.mkdtemp()
    urlretrieve(
        ARCHS_TF_PATH,
        osp.join(tmpdir, "genes.gmt"),
    )
    gene_sets = {}
    with open(osp.join(tmpdir, "genes.gmt"), "r") as f:
        for line in f:
            values = line.split("\t")
            gene_sets[values[0]] = [
                x for x in values[1:] if x not in ("", " ", "\n")
            ] + [values[0].split(" ")[0],]
    shutil.rmtree(tmpdir)
    gene_sets = {k.split(" ")[0]:v for k, v in gene_sets.items()}
    print("sample gene set keys:")
    print(list(gene_sets.keys())[:15])
    gene_sets = {k:v for k, v in gene_sets.items() if k in adata.var_names}
    print(f"# gene sets after filtering: {len(gene_sets)}")

    n_hidden_init = len(gene_sets.keys())
    print(f"n_hidden_init: {n_hidden_init}")
    model = scnym.model.CellTypeCLF(
        n_genes=adata.shape[1],
        n_cell_types=len(np.unique(y)),
        n_hidden=256,
        init_dropout=0.0,
        n_hidden_init=n_hidden_init,
    )

    if torch.cuda.is_available():
        model = model.cuda()
    model_device = list(model.parameters())[0].device

    # test a single loss computation
    batch = next(iter(dl))
    
    for k in batch:
        batch[k] = batch[k].to(device=model_device)

    lg_corr = scnym.losses.LatentGeneCorrGuide(
        gene_names=adata.var_names.tolist(),
        latent_var_genes=list(gene_sets.keys()),
        criterion="bce",
    )

    loss = lg_corr(
        model=model,
        labeled_sample=batch,
        unlabeled_sample=None,
    )
    print(f"bce loss: {loss}")

    lg_corr = scnym.losses.LatentGeneCorrGuide(
        gene_names=adata.var_names.tolist(),
        latent_var_genes=list(gene_sets.keys()),
        criterion="pearson",
    )

    loss = lg_corr(
        model=model,
        labeled_sample=batch,
        unlabeled_sample=None,
    )
    print(f"pearson loss: {loss}")

    print("testing the effect of perturbing genes we will later mask")
    batch["input"][:, lg_corr.latent_gene_idx[10]] += 100
    batch["input"][:, lg_corr.latent_gene_idx[20]] += 100

    loss = lg_corr(
        model=model,
        labeled_sample=batch,
        unlabeled_sample=None,
    )
    print(f"post-px pearson loss: {loss}")

    # test running with genes that don't have an mRNA present
    print("masked two genes to test:")
    mask_gene_0 = sorted(list(gene_sets.keys()))[10]
    mask_gene_1 = sorted(list(gene_sets.keys()))[20]
    print(f"\t{mask_gene_0}, {mask_gene_1}")
    masked_gene_names = adata.var_names.tolist()
    mask_idx_0 = masked_gene_names.index(mask_gene_0)
    mask_idx_1 = masked_gene_names.index(mask_gene_1)
    masked_gene_names[mask_idx_0] = "MASK"
    masked_gene_names[mask_idx_1] = "MASK"

    lg_corr = scnym.losses.LatentGeneCorrGuide(
        gene_names=masked_gene_names,
        latent_var_genes=list(gene_sets.keys()),
        criterion="pearson",
    )
    print(len(gene_sets.keys()))
    print("# latents_idx2keep")
    print(len(lg_corr.latents_idx2keep))
    print("# latents_idx2keep")
    print(len(lg_corr.genes_idx2keep))
    assert (len(gene_sets.keys()) - 2) == len(lg_corr.genes_idx2keep)
    assert (10 not in lg_corr.latent_gene_idx) and (20 not in lg_corr.latent_gene_idx)

    loss = lg_corr(
        model=model,
        labeled_sample=batch,
        unlabeled_sample=None,
    ) 
    print(f"masked pearson loss: {loss}")

    # train a few epochs

    lg_corr = scnym.losses.LatentGeneCorrGuide(
        gene_names=adata.var_names.tolist(),
        latent_var_genes=list(gene_sets.keys()),
        criterion="pearson",
        mean_corr_weight=0.1,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3,)

    init_loss = None
    for epoch in range(300):
        print("epoch", epoch)
        epoch_loss = 0.
        for data in dl:
            data["input"] = data["input"].to(device=model_device)
            loss = lg_corr(model=model, labeled_sample=data)

            optimizer.zero_grad()
            loss.backward()

            W = dict(model.named_parameters())["embed.0.weight"]
            print(f"\tmean abs grad: {W.grad.abs().mean()}")

            optimizer.step()
            epoch_loss += loss.detach().item()

            if init_loss is None:
                init_loss = loss.detach().cpu().item()

        print(f"\tloss {epoch_loss/len(dl)}")
    final_loss = epoch_loss/len(dl)
    assert final_loss < init_loss, f"final {final_loss} !<= init {init_loss}"
    return


def test_attrprior():
    """Test training with an attribution prior loss"""
    import scnym
    import copy
    from urllib.request import urlretrieve
    import scanpy as sc
    import pandas as pd
    import logging


    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    logging.debug("Test DEBUG statement")
    logging.info("Test INFO statement")

    # load 10x human PBMC data as a sample
    adata = sc.datasets.pbmc3k()
    sc.pp.filter_genes(adata, min_cells=30)
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.log1p(adata)

    X = adata.X
    y = np.random.randint(0, 4, size=adata.shape[0])
    ds = scnym.dataprep.SingleCellDS(
        X=X,
        y=y,
    )
    dl = torch.utils.data.DataLoader(ds, batch_size=256, drop_last=True)

    ARCHS_TF_PATH = "https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=ARCHS4_TFs_Coexp"
    tmpdir = tempfile.mkdtemp()
    urlretrieve(
        ARCHS_TF_PATH,
        osp.join(tmpdir, "genes.gmt"),
    )
    gene_sets = {}
    with open(osp.join(tmpdir, "genes.gmt"), "r") as f:
        for line in f:
            values = line.split("\t")
            gene_sets[values[0]] = [
                x for x in values[1:] if x not in ("", " ", "\n")
            ] + [values[0].split(" ")[0],]
    shutil.rmtree(tmpdir)
    gene_sets = {k.split(" ")[0]:v for k, v in gene_sets.items()}
    print("sample gene set keys:")
    print(list(gene_sets.keys())[:15])
    gene_sets = {k:v for k, v in gene_sets.items() if k in adata.var_names}
    print(f"# gene sets after filtering: {len(gene_sets)}")

    n_hidden_init = len(gene_sets.keys())
    print(f"n_hidden_init: {n_hidden_init}")
    model = scnym.model.CellTypeCLF(
        n_genes=adata.shape[1],
        n_cell_types=len(np.unique(y)),
        n_hidden=256,
        init_dropout=0.0,
        n_hidden_init=n_hidden_init,
    )

    if torch.cuda.is_available():
        model = model.cuda()
    model_device = list(model.parameters())[0].device

    # test a single loss computation
    batch = next(iter(dl))
    
    for k in batch:
        batch[k] = batch[k].to(device=model_device)

    AP = scnym.losses.AttrPrior(
        reference_dataset=ds,
        batch_size=256,
        attr_prior="gini_classwise",
        grad_activation="first_layer",
    )

    loss = AP(
        model=model,
        labeled_sample=batch,
        unlabeled_sample=None,
        weight=1.,
    )
    print(f"AP Gini loss: {loss}")

    print("Training with Gini loss")
    optimizer = torch.optim.Adam(model.parameters(),)

    for epoch in range(10):
        print("epoch", epoch)
        for batch in dl:
            loss = AP(model, labeled_sample=batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"\t{loss}")

    return


def main():
    logging.basicConfig(level=logging.INFO)
    # print("Test prior setting")
    # test_gene_set_to_priors()
    print("Test structured sparsity in and out of set perturbations")
    test_ss_in_and_out_of_set()
    # print("Test forward/backward")
    # test_sparsity_loss()
    # print("Test non-negativity prior")
    # test_nonneg_guide()
    # print("Test simple weight masking")
    # test_weight_mask()
    # print("Test training with weight mask")
    # test_weight_mask_train()
    # print("Test LatentGeneCorrGuide")
    # test_lv_gene_corr()
    # print("Test AttrPrior")
    # test_attrprior()
    # print("Test WithinGeneSetNorm")
    # test_withingenesetnorm()


if __name__ == "__main__":
    main()
