"""Implement a Pathfinder API wrapper around scBasset TF activity inference"""
import numpy as np
import pandas as pd
import anndata
import os.path as osp
import glob
import logging
logger = logging.getLogger(__name__)

try:
    import scbasset
except:
    msg = "Could not import scbasset. Install scBasset before benchmarking these models."
    logging.warn(msg)
from . import ModelAPI


MOTIFS = {
    "mouse": "/home/jacob/data/references/mouse/cisbp_1/scbasset_fasta",
    "human": "/home/jacob/data/references/human/cisbp_1",
}

GENOMES = {
    "mm_brain": "/home/jacob/data/references/mouse/mm10_Rn45s_mask/fasta/genome.fa",
    "mm_calico": "/home/jacob/data/references/mouse/mm10_Rn45s_mask/fasta/genome.fa",
    "hs_pbmc": "/home/jacob/data/references/human/refdata-cellranger-GRCh38-3.0.0/fasta/genome.fa",
    "hs_lymph": "/home/jacob/data/references/human/refdata-cellranger-GRCh38-3.0.0/fasta/genome.fa",
}

DATASETS = {
    "mm_brain": {
        "path" : "/group/singlecell/mouse/10x_multiome_e18_brain/",
        "species": "mouse",
    },
    "mm_calico": {
        "path": "/group/singlecell/mouse/2021_calico_multiomics_brain/exp1013_multiome_1000um_flownozzle_cellranger/outs/",
        "species": "mouse",
    },
    "hs_lymph": {
        "path": "/group/singlecell/human/10x_multiome_fresh_frozen_b_lymphoma/",
        "species": "human",
    },
    "hs_pbmc": {
        "path": "/group/singlecell/human/10x_multiome_pbmc/",
        "species": "human",
    },    
}


def make_datasets():
    for ds in DATASETS.keys():
        path = DATASETS[ds]["path"]
        DATASETS[ds]["atac"] = osp.join(path, "h5ad", "atac_raw.h5ad")
        DATASETS[ds]["rna"] = osp.join(path, "h5ad", "rna_raw.h5ad")
        DATASETS[ds]["peaks_bed"] = osp.join(path, "h5ad", "basset_peaks.bed")
        DATASETS[ds]["genome"] = GENOMES[ds]
        DATASETS[ds]["motifs"] = MOTIFS[DATASETS[ds]["species"]]
        DATASETS[ds]["seq_data"] = osp.join(path, "scbasset.h5")
        DATASETS[ds]["model"] = osp.join(path, "trained_scbasset", "model_best.h5")
    return

make_datasets()


class pscBasset(ModelAPI):

    def __init__(self, dataset: dict, n_latent: int=32) -> None:
        super(pscBasset, ModelAPI).__init__()
        self.dataset_dict = dataset

        # get list of available motifs
        self.motif_fastas = sorted(
            glob.glob(osp.join(dataset["motifs"], "shuffled_peaks_motifs", "*.fa*"))
        )
        logger.info(f"Found {len(self.motif_fastas)} sequence motifs for pscBasset.")
        self.motif_names = [osp.basename(x).split(".fa")[0] for x in self.motif_fastas]
        self.n_latent = n_latent
        return

    
    def fit(self, X, y, adata) -> None:
        """Check to ensure that a fit model exists.
        
        NOTE: We're pre-fitting models -- they might not match the dataset here.
        """
        if not osp.exists(self.dataset_dict["model"]):
            msg = f"{self.dataset_dict['model']} model path does not exist."
            return FileNotFoundError(msg)

        self.model = scbasset.utils.make_model(
            self.n_latent, adata.shape[0], show_summary=False
        )
        self.model.load_weights(self.dataset_dict["model"])
        self.trained = True
        return

    
    def transform(self, X, y, adata) -> pd.DataFrame:
        """Score TF activities with the trained model"""
        if not self.trained:
            msg = "Must train the model first, running .fit()"
            logger.warn(msg)
            self.fit(X=X, y=y, adata=adata)
        
        # score motifs
        motif_scores = pd.DataFrame(
            columns=self.motif_names,
            index=adata.obs_names,
        )
        for motif in self.motif_names:
            scores = scbasset.utils.motif_score(
                motif,
                self.model,
                motif_fasta_folder=self.dataset_dict["motifs"],
            )
            motif_scores.loc[:, motif] = scores
        self.motif_scores = motif_scores
        return motif_scores

    
    def query(
        self,
        adata: anndata.AnnData,
        source: str,
        target: str,
    ) -> list:
        """Query differential motif activity across cell types."""
        # get motif scores if needed
        if not hasattr(self.motif_scores):
            self.transform(X=adata.X, y=adata.obs[self.cell_type_col], adata=adata)
        # extract cell type specific scores
        source_bidx = adata.obs[self.cell_type_col] == source
        target_bidx = adata.obs[self.cell_type_col] == target
        source_scores = self.motif_scores.loc[source_bidx]
        target_scores = self.motif_scores.loc[target_bidx]

        # compute differential motif activity (DMA)
        # use a Mogrify-like heuristic -- lfc * -log10(t-test p-value)
        dma_scores = pd.DataFrame(
            index=self.motif_names, 
            columns=["lfc", "p_val", "source_mean", "target_mean", "score"],
        )
        clip_min = 1e-12
        for motif in self.motif_names:
            sx = np.array(source_scores[motif])
            tx = np.array(target_scores[motif])
            lfc = np.log2(tx) - np.log2(sx)

            t, p = stats.ttest_ind(
                sx,
                tx,
            )
            p = max(p, clip_min)
            dma_scores.loc[motif, "p_val"] = p
            dma_scores.loc[motif, "source_mean"] = sx.mean()
            dma_scores.loc[motif, "target_mean"] = tx.mean()
            dma_scores.loc[motif, "lfc"] = lfc
            dma_scores.loc[motif, "score"] = np.abs(lfc) * (-1 * np.log10(p))
        self.dma_scores = dma_scores

        grns = dma_scores.sort_values("score", ascending=False).index.tolist()
        return grns