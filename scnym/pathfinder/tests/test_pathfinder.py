import pytest
import os
import os.path as osp
import sys
import numpy as np
import pandas as pd
import torch
import anndata
from typing import Tuple

sys.path.append(osp.abspath('./'))
sys.path.append(osp.abspath('../'))

def _load_data_and_gene_sets() -> Tuple[anndata.AnnData, dict]:
    from pathfinder.bench.identity_bench import load_gene_sets, IDENT_DATASETS
    from pathfinder.bench.reprog_bench import load_dataset, REPROG_DATASETS
    
    adata = load_dataset(
        dataset="tabula_muris_senis_liver",
        dataset_map=REPROG_DATASETS,
    )
    
    gene_sets = load_gene_sets(
        gene_sets_path=IDENT_DATASETS["gene_sets"]["mouse"]["dorothea_mod_tfs"],
    )
    
    return adata, gene_sets


def test_attrpriors():
    from pathfinder.models import attributionpriors as attrprior
    import scnym
    
    adata, gene_sets = _load_data_and_gene_sets()
    
    ds = scnym.dataprep.SingleCellDS(
        X=adata.X,
        y=np.array(
            pd.Categorical(adata.obs["cell_type"]).codes,
            dtype=np.int32,
        ),
    )
    dl = torch.utils.data.DataLoader(ds, batch_size=128, drop_last=True)
    
    print("Built dataloaders.")
    # test forward call without batch transformation
    APExp = attrprior.AttributionPriorExplainer(
        background_dataset=ds,
        batch_size=128,
        input_batch_index="input",
    )
    
    model = scnym.model.CellTypeCLF(
        n_genes=adata.shape[1],
        n_cell_types=len(np.unique(adata.obs["cell_type"])),
        n_hidden=64,
        n_layers=2,
    )
    
    batch = next(iter(dl))
    
    input_ = batch["input"].float()
    _, target = torch.max(batch["output"].long(), dim=-1)
    target = target.long()
    
    print(f"Batch drawn, size: {input_.size()}")
    exp_grad = APExp.shap_values(
        model=model,
        input_tensor=input_,
        sparse_labels=target,
    )
    print("Gradients computed without embedder.")
    print(f"{exp_grad.size()}")
    print(f"{exp_grad}")
    print()
    
    # test graph prior
    print("Computing graph prior...")
    GP = attrprior.GeneSetGraphPrior(
        gene_sets=gene_sets,
        gene_names=adata.var_names.tolist(),
        weighting="count",
    )
    graph_prior = GP(exp_grad)
    max_w = torch.max(GP.gene_gene_matrix)
    sum_w = torch.sum(GP.gene_gene_matrix)
    sps_w = torch.sum(GP.gene_gene_matrix>0)
    print(f"graph_prior: {graph_prior}")
    print(f"graph size: {GP.gene_gene_matrix.size()}")
    print(f"max graph weight: {max_w}")
    print(f"sum graph weight: {sum_w}")
    print(f"num non-zero graph: {sps_w}")
    print()
    
    # test forward call *with* batch transformation
    embedder_model = torch.nn.Sequential(
        *list(model.modules())[3:7]
    ).train(False)
    score_model = torch.nn.Sequential(
        *list(model.modules())[7:-1]
    )
    def batch_transformation(x):
        y = embedder_model(x)
        return y.detach()
    # embed the input
    embedded_input_ = batch_transformation(
        input_,
    )
    print(f"Input embedded, size: {embedded_input_.size()}")

    exp_grad = APExp.shap_values(
        score_model, 
        embedded_input_,
        sparse_labels=target,
        batch_transformation=batch_transformation,
    )
    print("Gradients computed *with* embedder.")
    print(f"{exp_grad.size()}")    
    print(f"{exp_grad}")
    return


def test_pathfinder():
    import pathfinder
    print(f"pathfinder: {pathfinder.__file__}")
    from pathfinder.models import Pathfinder

    adata, gene_sets = _load_data_and_gene_sets()
    gene_sets = {
        k:v for k,v in gene_sets.items() if k in ["Ascl1", "Myod1", "Hnf4a", "Gata4"]
    }
    print("Loaded data.")
    print("%d cells, %d genes." % adata.shape)
    print(f"{len(gene_sets.keys())} gene sets.")
    print()
    
    pfnd = Pathfinder(
        gene_sets=gene_sets,
        reg_weight=1e-4,
        grad_activation="input",
        log_dir="./tmp_pathfinder_test",
        attr_weight=1e-2,
    )
    
    pfnd.fit(
        X=adata.X,
        y=adata.obs["cell_type"],
        adata=adata,
        n_epochs=5,
    )
    
    grns = pfnd.query(
        adata=adata,
        source="hepatic stellate cell",
        target="hepatocyte",
    )
    
    idx = grns.index("Hnf1a")
    print(f"stellate -> hepatocyte Hnf1a rank: {idx+1}")
    print("Done.")
    pfnd.saliency.to_csv("tmp.csv")
    return


def test_pathfinder_giniprior():
    from pathfinder.models import Pathfinder

    adata, gene_sets = _load_data_and_gene_sets()
    print("Loaded data.")
    print("%d cells, %d genes." % adata.shape)
    print(f"{len(gene_sets.keys())} gene sets.")
    print()
    
    print("Testing Pathfinder with input level attribution priors.")
    pfnd = Pathfinder(
        log_dir="./tmp_pathfinder_test",
        gene_sets=gene_sets,
        reg_weight=1e-4,
        attr_weight=0.0,
        grad_activation="input",
        interp_method="exp_grad",
    )
    
    pfnd.fit(
        X=adata.X,
        y=adata.obs["cell_type"],
        adata=adata,
        n_epochs=1,
    )
    
    print("Testing expected gradients interpretation")
    grns = pfnd.query(
        adata=adata,
        source="hepatic stellate cell",
        target="hepatocyte",
        n_cells=300,
    )
    idx = grns.index("Hnf1a")
    print(f"stellate -> hepatocyte Hnf1a rank: {idx+1}")
    
    print("Testing expected gradients interpretation using only source references.")
    grns = pfnd.query(
        adata=adata,
        source="hepatic stellate cell",
        target="hepatocyte",
        n_cells=300,        
        only_source_reference=True,
    )
    idx = grns.index("Hnf1a")
    print(f"stellate -> hepatocyte Hnf1a rank: {idx+1}")        
    
    print("Testing integrated gradients interpretation")
    pfnd.interp_method = "int_grad"
    grns = pfnd.query(
        adata=adata,
        source="hepatic stellate cell",
        target="hepatocyte",
    )    
    
    idx = grns.index("Hnf1a")
    print(f"stellate -> hepatocyte Hnf1a rank: {idx+1}")
    print("Done.")
    
    print("Testing Pathfinder with first_layer attribution priors.")
    pfnd = Pathfinder(
        gene_sets=gene_sets,
        log_dir="./tmp_pathfinder_test",
        reg_weight=1e-3,
        attr_weight=1e-2,
        grad_activation="first_layer",
        interp_method="exp_grad",
        attr_prior="gini",
        scoring_metric="intgrad_lv",
    )
    
    pfnd.fit(
        X=adata.X,
        y=adata.obs["cell_type"],
        adata=adata,
        n_epochs=5,
    )
    
    grns = pfnd.query(
        adata=adata,
        source="hepatic stellate cell",
        target="hepatocyte",
        n_cells=300,
        n_batches=100,        
    )
    
    idx = grns.index("Hnf1a")
    print(f"stellate -> hepatocyte Hnf1a rank: {idx+1}")
    print("Done.")
    return


def test_pathfinder_graphprior(n_epochs: int=5):
    from pathfinder.models import Pathfinder

    adata, gene_sets = _load_data_and_gene_sets()
    print("Loaded data.")
    print("%d cells, %d genes." % adata.shape)
    
    gene_sets = {
        k: gene_sets[k] for k in ["Hnf1a", "Neurod1", "Foxa2", "Pax7", "Sox2", "Hoxc10"]
    }    
    print(f"{len(gene_sets.keys())} gene sets.")
    print()
    
    print("Testing Pathfinder with input level attribution priors.")
    
    
    def attr_weight(x):
        w = 0. if x < 2 else ((x-2)/3)*1e-2
        return w
    
    pfnd = Pathfinder(
        gene_sets=gene_sets,
        reg_weight=0.0,
        attr_weight=attr_weight,
        grad_activation="input",
        interp_method="exp_grad",
        attr_prior="graph_log",
        log_dir="./tmp_pathfinder_test"
    )
    
    pfnd.fit(
        X=adata.X,
        y=adata.obs["cell_type"],
        adata=adata,
        n_epochs=n_epochs,
    )
    
    print("Testing expected gradients interpretation")
    grns = pfnd.query(
        adata=adata,
        source="hepatic stellate cell",
        target="hepatocyte",
        n_cells=100,
    )
    idx = grns.index("Hnf1a")
    print(f"stellate -> hepatocyte Hnf1a rank: {idx+1}")
    
    print("Testing expected gradients interpretation using only source references.")
    grns = pfnd.query(
        adata=adata,
        source="hepatic stellate cell",
        target="hepatocyte",
        n_cells=100,        
        only_source_reference=True,
    )
    idx = grns.index("Hnf1a")
    print(f"stellate -> hepatocyte Hnf1a rank: {idx+1}")
    return


def test_graphprior_optimization():
    """Test that the graph prior can be optimized"""
    from pathfinder.models import Pathfinder
    import scnym

    adata, gene_sets = _load_data_and_gene_sets()
    print("Loaded data.")
    print("%d cells, %d genes." % adata.shape)
    
    gene_sets = {
        k: gene_sets[k] for k in ["Hnf1a", "Neurod1", "Foxa2", "Pax7", "Sox2", "Hoxc10"]
    }    
    print(f"{len(gene_sets.keys())} gene sets.")
    print()
    
    print("Testing Pathfinder with input level attribution priors.")
    
    
    def attr_weight(x):
        w = 0. if x < 1 else 100.
        return w
    
    pfnd = Pathfinder(
        gene_sets=gene_sets,
        log_dir="./tmp_pathfinder_test/",
        reg_weight=0.0,
        attr_weight=100.0,
        grad_activation="input",
        interp_method="exp_grad",
        attr_prior="graph_log",
    )
    
    pfnd._setup_fit(
        batch_size=512,
        n_epochs=1,
        X=adata.X,
        y=adata.obs["cell_type"],
        adata=adata,
    )
    
    
    pfnd.model.train(True)
    
    def _print_grad(model):
        g = 0.0
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                g += torch.norm(m.weight.grad, p=2)
                g += torch.norm(m.bias.grad, p=2)
        print(f"\t{g}")
        return

    epoch_losses = []
    print("Training, attr losses:")
    epoch = 0
    for data in pfnd.train_dl:

        data = pfnd.trainer.batch_transformers["train"](data)

        input_ = data["input"].to(device=pfnd.model_device)
        target = data["output"].to(device=pfnd.model_device)

        outputs = pfnd.model(input_)
        # _, preds = torch.max(outputs, dim=-1)
        _, target_int = torch.max(target, dim=-1)

        l_ce = scnym.losses.cross_entropy(outputs, target)
        l_ss = pfnd.reg_criterion(
            pfnd.model,
        )
        l_ap = pfnd.attr_criterion(
            model=pfnd.model,
            labeled_sample={
                "input": input_,
                "output_int": target_int,
            },
            unlabeled_sample=None,
            weight=1.,
        )

        loss = (
            0.0 * l_ce 
            + pfnd.reg_weight(epoch)  * l_ss
            + pfnd.attr_weight(epoch) * l_ap
        )
        print(f"\t{pfnd.attr_weight(epoch) * l_ap}")

        pfnd.trainer.optimizer.zero_grad()
        loss.backward()
        
        print("\tGradient on parameters")
        _print_grad(pfnd.model)
        pfnd.trainer.optimizer.step()
        print()

        epoch_losses.append(
            l_ap.detach().item(),
        )
    
    epoch_losses = np.array(epoch_losses)
    print(f"Initial loss: {epoch_losses[0]}")
    print(f"Final loss: {epoch_losses[-1]}")
    
    delta = np.mean(epoch_losses[-3:]) - np.mean(epoch_losses[:3])
    if not (delta < 0):
        msg = f"diff in loss: {delta} was >0."
        raise RuntimeError(msg)
    return
    


def test_pathfauxgrify():
    from pathfinder.models.pathfinder import PathFauxgrify
    
    adata, gene_sets = _load_data_and_gene_sets()
    print("Loaded data.")
    print("%d cells, %d genes." % adata.shape)
    print(f"{len(gene_sets.keys())} gene sets.")
    print()    

    gene_sets = {
        k: gene_sets[k] for k in ["Hnf1a", "Neurod1", "Foxa2", "Pax7", "Sox2", "Hoxc10"]
    }
    
    model = PathFauxgrify(
        gene_sets=gene_sets,
        log_dir="./tmp_pathfinder_test",
        reg_weight=0.0,
        attr_weight=1.0,
        grad_activation="input",
        interp_method="exp_grad",
        attr_prior="graph_binary",
    )
    
    model.fit(
        X=adata.X,
        y=adata.obs["cell_type"],
        adata=adata,
        n_epochs=5,
    )
    
    source = "Kupffer cell"
    target = "hepatocyte"
    
    grns = model.query(
        adata=adata,
        source=source,
        target=target,
        n_cells=512,
        n_batches=100,
    )
    network_scores = model.network_scores
    
    print("network_scores")
    print(network_scores.head(15))
    
    idx = grns.index("Hnf1a")
    print(f"endothelial -> hepatocyte Hnf1a rank: {idx+1}")
    print("Done.")
    assert idx < 3
    
    print("\ngene_scores")
    print(model.gene_scores.head(15))
    print()
    return



def main():
    import logging
    logging.basicConfig(level=logging.DEBUG)
    # test_attrpriors()
    # test_pathfinder()
    test_pathfinder_giniprior()
    # test_pathfinder_graphprior(n_epochs=5)
    # test_graphprior_optimization()
    # test_pathfauxgrify()
    return


if __name__ == "__main__":
    main()
