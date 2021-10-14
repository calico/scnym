"""Test guided latent variables using structured sparsity penalties"""

import pytest
import os
import os.path as osp
import shutil
import tempfile
import sys
import numpy as np
import torch

sys.path.append("../")


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
    import copy
    from urllib.request import urlretrieve
    import scanpy as sc

    torch.manual_seed(1)
    np.random.seed(1)

    # load 10x human PBMC data as a sample
    adata = sc.datasets.pbmc3k()

    # load sample gene sets
    HALLMARK_PATH = "https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=MSigDB_Hallmark_2020"
    urlretrieve(HALLMARK_PATH, "./hallmark.gmt")
    gene_sets = {}
    with open("./hallmark.gmt", "r") as f:
        for line in f:
            values = line.split("\t")
            gene_sets[values[0]] = [x for x in values[1:] if x not in ("", " ", "\n")]

    for k in gene_sets.keys():
        gene_sets[k] = [x for x in gene_sets[k] if x in adata.var_names]

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
    print(SS.prior_matrix)
    print("Program names")
    print(SS.gene_set_names)
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


def main():
    print("Test prior setting")
    # test_gene_set_to_priors()
    print("Test forward/backward")
    test_sparsity_loss()
    print("Test non-negativity prior")
    test_nonneg_guide()


if __name__ == "__main__":
    main()
