import pytest
import os
import os.path as osp
import sys
import numpy as np
import torch

sys.path.append("../")


def test_reconstruction_loss():
    """Test the reconstruction loss and associated
    encoder/decoder model.
    """
    import scnym
    import scanpy as sc
    import pandas as pd

    torch.manual_seed(1)
    np.random.seed(1)

    #####################################
    # Generate sample data loaders
    #####################################

    # load 10x human PBMC data as a sample
    adata = sc.datasets.pbmc3k()
    # seurat flavor operates on counts
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat_v3")
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
        domain=np.zeros(X.shape[0]),
        num_domains=1,
    )

    l_dl = torch.utils.data.DataLoader(ds, batch_size=256)
    t_dl = torch.utils.data.DataLoader(ds, batch_size=256)
    u_dl = torch.utils.data.DataLoader(ds, batch_size=256)

    dataloaders = {
        "train": l_dl,
        "val": t_dl,
    }

    #####################################
    # Setup the model
    #####################################

    model = scnym.model.CellTypeCLF(
        n_genes=X.shape[1],
        n_cell_types=len(np.unique(y)),
        n_hidden=128,
        n_layers=1,
        init_dropout=0.1,
        use_raw_counts=True,
    )
    if torch.cuda.is_available():
        model = model.cuda()

    model_device = list(model.parameters())[1].device

    #####################################
    # Setup reconstruction loss
    #####################################

    # autoencoder is initialized internally
    rec_crit = scnym.losses.ReconstructionLoss(
        rec_criterion=scnym.trainer.negative_binomial_loss,
        model=model,
        n_domains=1,
    )
    print("Models setup.")

    #####################################
    # Setup optimizer
    #####################################

    optimizer = torch.optim.Adadelta(
        [
            {"params": model.parameters(), "name": "clf"},
            {
                "params": rec_crit.rec_model.decoder.parameters(),
                "name": "ae_dec",
            },
            {"params": rec_crit.rec_model.dispersion, "name": "dispersion"},
            {
                "params": rec_crit.rec_model.libenc.parameters(),
                "name": "ae_libenc",
            },
        ],
    )

    # run a single forward pass through the reconstruction loss
    batch = next(iter(t_dl))

    for k in batch.keys():
        batch[k] = batch[k].to(device=model_device)

    rec_loss = rec_crit(
        labeled_sample=batch,
    )
    print(f"initial rec loss: {rec_loss}")
    assert rec_loss > 0.0

    # train a couple epochs and make sure that the loss decreases
    init_rec_loss = float(rec_loss.detach().cpu().item())

    for _iter in range(100):
        rec_loss = rec_crit(labeled_sample=batch)
        optimizer.zero_grad()
        rec_loss.backward()
        optimizer.step()
    final_rec_loss = float(rec_loss.detach().cpu().item())

    if init_rec_loss < final_rec_loss:
        msg = "reconstruction loss didn't decrease.\n"
        msg += f"\tinit  {init_rec_loss:.6f}\n"
        msg += f"\tfinal {final_rec_loss:.6f}\n"
        raise ValueError(msg)

    print(f"\tinit  {init_rec_loss:.6f}")
    print(f"\tfinal {final_rec_loss:.6f}")

    return


def main():
    test_reconstruction_loss()
    return


if __name__ == "__main__":
    main()
