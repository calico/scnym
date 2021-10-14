import pytest
import os
import os.path as osp
import sys
import numpy as np
import pandas as pd
import torch
import scanpy as sc

sys.path.append("../")


def _load_10x_pbmc():
    adata = sc.datasets.pbmc3k()
    sc.pp.filter_cells(adata, min_counts=100)
    sc.pp.filter_genes(adata, min_cells=100)
    sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e6)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=3000)
    sc.pp.pca(adata)
    sc.pp.neighbors(adata, n_neighbors=15)
    sc.tl.leiden(adata, resolution=0.3)
    # name one class T cell and one B cell
    cd4 = adata.obs_vector("CD4")
    cd22 = adata.obs_vector("CD22")
    leiden = adata.obs_vector("leiden")
    tmp = pd.DataFrame({"CD4": cd4, "CD22": cd22, "leiden": leiden})
    grp = tmp.groupby("leiden").mean().reset_index()
    print(grp)
    t_cell_cl = grp.sort_values("CD4", ascending=False)["leiden"].tolist()[0]
    b_cell_cl = grp.sort_values("CD22", ascending=False)["leiden"].tolist()[0]
    print(f"T {t_cell_cl}, B {b_cell_cl}")
    leiden = np.array(leiden)
    leiden[adata.obs["leiden"] == t_cell_cl] = "T"
    leiden[adata.obs["leiden"] == b_cell_cl] = "B"
    adata.obs["annot"] = leiden
    return adata


def test_expgrad():
    import scnym
    from scnym.dataprep import SingleCellDS

    torch.manual_seed(1)
    np.random.seed(1)

    # load 10x human PBMC data as a sample
    print("Loading toy dataset...")
    adata = _load_10x_pbmc()
    print("Data ready.")
    # train a model for a few epochs
    model = scnym.model.CellTypeCLF(
        n_layers=1,
        n_hidden=128,
        residual=False,
        init_dropout=0.0,
        n_genes=adata.shape[1],
        n_cell_types=len(np.unique(adata.obs["leiden"])),
    )
    if torch.cuda.is_available():
        model = model.cuda()
    model_device = list(model.parameters())[0].device

    y = np.array(pd.Categorical(adata.obs["leiden"]).codes)
    ds = SingleCellDS(
        X=adata.X,
        y=y,
    )
    dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)

    print("Training toy model...")
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(10):
        for data in dl:
            input_ = data["input"].to(device=model_device)
            target = data["output"].to(device=model_device)

            outputs = model(input_)
            outputs = torch.nn.functional.softmax(outputs, dim=-1)
            loss = scnym.trainer.cross_entropy(outputs, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(loss.item())
    print("Training complete.")

    _, y_hat = torch.max(outputs, dim=1)
    _, y = torch.max(target, dim=1)

    print("pred\ttruth")
    for i in range(10):
        print(f"{y_hat[i]}\t{y[i]}")

    print("Testing expected gradients...")
    EG = scnym.interpret.ExpectedGradient(
        model=model,
        class_names=np.unique(adata.obs["annot"]),
        gene_names=np.array(adata.var_names),
    )

    print("\tTesting with single source :: target pair")
    saliency = EG.query(
        adata=adata,
        source="T",
        target="B",
        cell_type_col="annot",
        n_cells=50,
        n_batches=3,
    )
    print(saliency)
    print("E[Grad] computed.")

    print("\tTesting with all data as reference")
    saliency = EG.query(
        adata=adata,
        target="B",
        source="all",
        cell_type_col="annot",
        n_cells=50,
        n_batches=3,
    )
    print(saliency)
    print("E[Grad] computed.")

    return


def main():
    import logging

    logging.basicConfig(level=logging.DEBUG)
    test_expgrad()
    return


if __name__ == "__main__":
    main()
