"""Implement Pathfinder reprogramming approach by wrapping
scNym models with Pathfinder training and interpretation
schemes"""
import numpy as np
import pandas as pd
import anndata
import scanpy as sc
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
import time
import copy
import logging
from pathlib import Path
from typing import Union, Callable, Tuple

# self-dependencies
import scnym
from . import ModelAPI
from .baseline import Fauxgrify
from . import attributionpriors as attrprior
from ..utils import ad_groupby


logger = logging.getLogger(__name__)


def _get_data_split(X, y) -> Tuple[np.ndarray, np.ndarray]:
    """Get a data split from `X, y`

    Parameters
    ----------
    X : np.ndarray, csr_matrix
        [Observations, Features] matrix.
    y : np.ndarray
        [Observations,] class labels.

    Returns
    -------
    traintest_idx : np.ndarray
        [int,] training and early stopping samples.
    val_idx : np.ndarray
        [int,] final evaluation samples.
    """
    val_idx = np.random.choice(
        np.arange(X.shape[0]),
        size=int(0.9 * X.shape[0]),
        replace=False,
    )
    traintest_idx = np.setdiff1d(np.arange(X.shape[0]), val_idx)
    return traintest_idx, val_idx


def _get_data_split_from_anndata(
    adata: anndata.AnnData,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract train and test sets from an AnnData if they're specified.

    Parameters
    ----------
    adata : anndata.AnnDat
        [Observations, Features] experiment possibly containing pre-specified
        data splits in `.obs["data_split"]`.

    Returns
    -------
    traintest_idx : np.ndarray
        [int,] training and early stopping samples.
    val_idx : np.ndarray
        [int,] final evaluation samples.

    Notes
    -----
    Generates a random data split if no datasplit is found.
    """
    if "data_split" in adata.obs.columns:
        traintest_idx = np.where(adata.obs["data_split"] == "train")[0]
        val_idx = np.where(adata.obs["data_split"] == "test")[0]
    else:
        traintest_idx, val_idx = _get_data_split(
            X=adata.X,
            y=np.zeros(adata.shape[0]),
        )
    return traintest_idx, val_idx


class APWrapper(nn.Module):
    def __init__(self, attr_criterion: Callable) -> None:
        """Wrapper around `attr_criterion` function for compatability with the
        `MultiTaskTrainer` API."""
        super(APWrapper, self).__init__()
        self.attr_criterion = attr_criterion
        return

    def __call__(
        self, model, labeled_sample, unlabeled_sample, weight
    ) -> torch.FloatTensor:
        # cast targets to integers for attribution prior calculation
        if labeled_sample["output"].dim() > 1:
            _, target_int = torch.max(labeled_sample["output"], dim=-1)
            labeled_sample["output_int"] = target_int
        c = self.attr_criterion(
            model=model,
            labeled_sample=labeled_sample,
            unlabeled_sample=unlabeled_sample,
            weight=weight,
        )
        logger.debug(f"APWrapper criterion: {c}")
        return c


class RegCritWrapper(nn.Module):
    def __init__(self, criterion: Callable) -> None:
        """Wrapper around `reg_criterion` function for compatability with the
        `MultiTaskTrainer` API."""
        super(RegCritWrapper, self).__init__()
        self.criterion = criterion
        return

    def __call__(
        self, model, labeled_sample, unlabeled_sample, weight
    ) -> torch.FloatTensor:
        c = self.criterion(
            model=model,
            labeled_sample=labeled_sample,
            unlabeled_sample=unlabeled_sample,
            weight=weight,
        )
        logger.debug(f"RegCritWrapper criterion: {c}")
        return c


class Pathfinder(ModelAPI):
    """
    Attributes
    ----------
    gene_sets : dict
        keys are GRN names, values are lists of gene names.
    cell_type_col : str
        column in `adata.obs` that will hold cell type classes.
    n_hidden : int
        number of hidden units in unregularized hidden classifier layers.
    n_sparse_latent : int
        number of sparse latent units without structured regularization.
    n_dense_latent : int
        number of unregularized latent units.
    n_layers : int
        number of classifier network hidden layers.
    reg_weight : float, Callable
        l1 regularization weight.
        if `Callable`, takes a single `int` epoch as argument and returns
        weight for the loss.
    attr_strength: float, Callable
        attribution prior loss weight.
        if `Callable`, takes a single `int` epoch as argument and returns
        weight for the loss.
    scoring_metric : str
        ranking metric for GRNs.
        "intgrad_gene" uses integrated gradients on individual GRN genes.
        "intgrad_lv" uses integrated gradients on the GRN LVs.
        "saliency" uses a saliency based dot-prod of intgrad LV and gene expr.
    grad_activation : str
        activations to use for gradient computation {"first_layer", "input"}.
    interp_method : str
        interpretation method to use in {"int_grad", "exp_grad"}.
    attr_prior : str
        attribution prior in {"gini", "classwise_gini", "graph_{weighting}"},
        where {weighting} is a valid weight style for `GeneSetsGraphPrior`.
    log_dir : str
        directory for tensorboard logging. if `None`, logging is not performed.
    model_gene_names : List[str]
        gene names for each of the features used during model training.
    trained : bool
        `True` if the model has been trained.
    restart_save_epochs : int
        Number of epochs at which to restart the weight saving process.
        Useful for saving weights that may not have the lowest possible val loss,
        but happen to be generated after some regularization process has warmed up.
    """

    trained = False
    restart_save_epochs = -1

    def __init__(
        self,
        gene_sets: dict,
        log_dir: str,
        cell_type_col: str = "cell_type",
        domain_groupby: str = None,
        n_hidden: int = 256,
        n_sparse_latent: int = None,
        n_dense_latent: int = None,
        n_layers: int = 2,
        reg_weight: Union[float, Callable] = 0.0,
        attr_weight: Union[float, Callable] = 0.0,
        scoring_metric: str = "intgrad_gene",
        grad_activation: str = "input",
        interp_method: str = "exp_grad",
        attr_prior: str = "graph_log",
        patience: int = 20,
        domain_adv_kwargs: dict={},
        trainer_kwargs: dict={},
        reg_criterion_kwargs: dict={},
        model_kwargs: dict={"hidden_init_dropout": True},
        **kwargs,
    ) -> None:
        """Pathfinder models find reprogramming strategies between cell types
        using a learned single cell classifier, structured latent variable
        learning, and integrated gradients for attribution scoring.

        Parameters
        ----------
        gene_sets : dict
            keys are GRN names, values are lists of gene names.
        log_dir : str
            directory for tensorboard logging. if `None`, logging is not performed.
        cell_type_col : str
            column in `adata.obs` that will hold cell type classes.
        domain_groupby : str
            column in `adata.obs` that specifies domains of origin.
        n_hidden : int
            number of hidden units in unregularized hidden classifier layers.
        n_sparse_latent : int
            number of sparse latent units without structured regularization.
            if not None and `n_dense_latent is not None`, uses a latent variable layer
            with a size `len(gene_sets) + n_sparse_latent + n_dense latent`.
        n_dense_latent : int
            number of unregularized latent units.
            if not None and `n_sparse_latent is not None`, uses a latent variable layer
            with a size `len(gene_sets) + n_sparse_latent + n_dense latent`.            
        n_layers : int
            number of classifier network hidden layers.
        reg_weight : float, Callable
            l1 regularization weight.
            if `Callable`, takes a single `int` epoch as argument and returns
            weight for the loss.
        attr_strength: float, Callable
            attribution prior loss weight.
            if `Callable`, takes a single `int` epoch as argument and returns
            weight for the loss.
        scoring_metric : str
            ranking metric for GRNs.
            "intgrad_gene" uses integrated gradients on individual GRN genes.
            "intgrad_lv" uses integrated gradients on the GRN LVs.
            "saliency" uses a saliency based dot-prod of intgrad LV and gene expr.
        grad_activation : str
            activations to use for gradient computation {"first_layer", "input"}.
        interp_method : str
            interpretation method to use in {"int_grad", "exp_grad"}.
        attr_prior : str
            attribution prior in {"gini", "classwise_gini", "graph_{weighting}"},
            where {weighting} is a valid weight style for `GeneSetsGraphPrior`.
        patience : int
            epochs to wait before early stopping model training.
        domain_adv_kwargs : dict
            keyword arguments to parameterize a domain adversary.
        trainer_kwargs : dict
            keyword arguments passed to `scnym.trainer.MultiTaskTrainer`.
        reg_criterion_kwargs : dict
            keyword arguments passed to `scnym.losses.StructuredSparsity`.

        Returns
        -------
        None.

        See Also
        --------
        scnym.model.CellTypeCLF
        scnym.trainer.MultiTaskTrainer
        """
        # pass kwargs to allow for multiple inheritance
        super(Pathfinder, self).__init__(gene_sets=gene_sets, **kwargs)

        self.gene_sets = gene_sets
        self.gene_set_names = sorted(list(gene_sets.keys()))
        self.genes_in_gene_sets = sorted(list(set(sum(list(gene_sets.values()), []))))
        self.cell_type_col = cell_type_col
        self.domain_groupby = domain_groupby
        # we have to initialize a `model` in the `.fit` method
        # because we don't yet know how many cell types there
        # are.
        self.n_hidden = n_hidden
        if n_sparse_latent is not None and n_dense_latent is not None:
            # train an LV model
            logger.info("Building a latent variable Pathfinder model.")
            self.n_hidden_init = (
                len(self.gene_set_names) + n_sparse_latent + n_dense_latent
            )
            logger.info(f"Using {self.n_hidden_init} latent variables.")
        else:
            self.n_hidden_init = n_hidden
        self.n_sparse_latent = n_sparse_latent
        self.n_dense_latent = n_dense_latent
        self.n_layers = n_layers
        self.weight_decay = 1e-5
        self.patience = patience
        self.domain_adv_kwargs = domain_adv_kwargs
        self.trainer_kwargs = trainer_kwargs
        self.reg_criterion_kwargs = reg_criterion_kwargs
        self.model_kwargs = model_kwargs

        if type(reg_weight) == float:
            logger.info(f"Set reg_weight to lambda constant {reg_weight}")
            # avoid using a simple `lambda` so class is pickleable
            self._reg_weight_constant = reg_weight
            self.reg_weight = self._get_reg_weight_constant
        elif callable(reg_weight):
            self.reg_weight = reg_weight
        else:
            msg = f"reg_weight must by `float,Callable`, not {type(reg_weight)}."
            raise TypeError(msg)

        if type(attr_weight) == float:
            logger.info(f"Set attr_weight to lambda constant {attr_weight}")
            self._attr_weight_constant = attr_weight
            self.attr_weight = self._get_attr_weight_constant
        elif callable(attr_weight):
            self.attr_weight = attr_weight
        else:
            msg = f"attr_weight must by `float,Callable`, not {type(attr_weight)}."
            raise TypeError(msg)

        self.scoring_metric = scoring_metric
        self.grad_activation = grad_activation
        self.interp_method = interp_method
        self.attr_prior = attr_prior
        self.log_dir = log_dir
        self.tbw = None

        # set the default number of epochs. can be updated by the user in `.fit`
        self.n_epochs = 400
        # pass after `super` to override `ModelAPI` defaults
        self._fit_before_query = True
        self._needs_unique_log_dir = True
        return

    def _get_reg_weight_constant(self, x):
        return self._reg_weight_constant

    def _get_attr_weight_constant(self, x):
        return self._attr_weight_constant

    def _split_data(self, X, y, adata=None) -> None:
        """Split the dataset into training and testing sets, extracting
        pre-specified data splits if available

        Returns
        -------
        None. Sets `.traintest_idx` and `.val_idx` with `np.ndarray[int]`
        indices for each data split.
        """
        # setup dataset -- note we use the original scNym dataset
        # nomenclature which is actually wrong
        # train/test -> training and early stopping set
        # val -> final test set
        if adata is not None:
            traintest_idx, val_idx = _get_data_split_from_anndata(adata)
        else:
            traintest_idx, val_idx = _get_data_split(X, y)
        self.traintest_idx = traintest_idx
        self.val_idx = val_idx
        return

    def _setup_dataset(self, X, y, adata=None) -> None:
        """Setup `Dataset` and `DataLoader`classes for train
        and validation data.

        Returns
        -------
        None.
        Sets `.train_ds`, `.val_ds` and `.train_dl`, `.val_dl`.
        """
        self.n_cell_types = len(np.unique(y))
        self.n_genes = X.shape[1]

        self.y_orig = y
        y = np.array(pd.Categorical(y, categories=np.unique(y)).codes)
        self.y = y
        self.y_categories = np.unique(self.y_orig)

        # parse domain information if provided
        if self.domain_groupby is not None:
            self.domain_orig = np.array(adata.obs[self.domain_groupby])
            self.domain = np.array(
                pd.Categorical(
                    self.domain_orig,
                    categories=np.unique(self.domain_orig),
                ).codes,
                dtype=np.int32,
            )
            self.n_domains = len(np.unique(self.domain))
        else:
            self.domain = None
            self.n_domains = 1

        # setup dataset, model, and training components
        self.model_gene_names = [
            x for x in adata.var_names if x in self.genes_in_gene_sets
        ]

        self._split_data(X, y, adata)

        # setup datasets
        self.train_ds = scnym.dataprep.SingleCellDS(
            X=X[self.traintest_idx],
            y=np.array(y[self.traintest_idx]),
            domain=self.domain[self.traintest_idx] if self.domain is not None else None,
        )
        self.val_ds = scnym.dataprep.SingleCellDS(
            X=X[self.val_idx],
            y=np.array(y[self.val_idx]),
            domain=self.domain[self.val_idx] if self.domain is not None else None,
        )

        # setup dataloaders
        self.train_dl = torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )
        self.val_dl = torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )

        # for querying, we also set a dataset with all of the data
        self.all_ds = scnym.dataprep.SingleCellDS(
            X=X, y=np.array(y), domain=self.domain if self.domain is not None else None
        )
        self.all_dl = torch.utils.data.DataLoader(
            self.all_ds,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )

        return

    def _setup_model(
        self,
    ) -> None:
        """Initialize an underlying scNym classifier to train

        Returns
        -------
        None.
        Sets `.model`.
        """
        self.model = scnym.model.CellTypeCLF(
            n_genes=self.n_genes,
            n_cell_types=self.n_cell_types,
            n_hidden=self.n_hidden,
            n_hidden_init=self.n_hidden_init,
            n_layers=self.n_layers,
            **self.model_kwargs,
        )

        # move objects to CUDA
        if torch.cuda.is_available():
            logger.info("CUDA device found for Pathfinder.")
            self.model = self.model.cuda()

        self.model_device = list(self.model.parameters())[0].device
        return

    def _setup_structured_sparsity(self, adata: anndata.AnnData) -> None:
        """Setup the structured sparsity l1 regularization function."""
        self.reg_criterion = scnym.trainer.StructuredSparsity(
            n_genes=self.model.n_genes,
            n_hidden=self.model.n_hidden_init,
            gene_sets=self.gene_sets,
            gene_names=np.array(adata.var_names),
            n_dense_latent=self.n_dense_latent,
            **self.reg_criterion_kwargs,
        )
        if not np.all(
            np.array(self.reg_criterion.gene_set_names) == np.array(self.gene_set_names)
        ):
            msg = "gene set name orders do not match"
            raise ValueError(msg)
        self.reg_criterion.prior_matrix = self.reg_criterion.prior_matrix.to(
            device=self.model_device
        )
        return

    def _input_exp_grad(
        self,
        model,
        input_: torch.FloatTensor,
        target: torch.LongTensor,
    ) -> torch.FloatTensor:
        """Get expected gradients from the input layer"""
        exp_grad = self.APExp.shap_values(
            model,
            input_,
            sparse_labels=target,
        )
        return exp_grad

    def _first_layer_exp_grad(
        self,
        model,
        input_: torch.FloatTensor,
        target: torch.LongTensor,
    ) -> torch.FloatTensor:
        """Get expected gradients for the first layer activations"""
        embedder_model = copy.deepcopy(model.get_initial_embedder())
        embedder_model.train(False)

        score_model = nn.Sequential(
            *model.mid_stack,
            *model.hidden_layers,
            model.classif,
        )
        # display the score and embedder models for debugging
        m_desc = list(embedder_model.modules())[0]
        logger.debug(f"embedder_model extracted\n{m_desc}")
        m_desc = list(score_model.modules())[0]
        logger.debug(f"score_model extracted\n{m_desc}")

        # debugger model visualization
        emdl = list(embedder_model.modules())[0]
        smdl = list(score_model.modules())[0]
        logger.debug(f"embedder_model: {emdl}")
        logger.debug(f"score_model: {smdl}")

        def batch_transformation(x):
            y = embedder_model(x)
            return y.detach()

        # embed the input
        embedded_input_ = batch_transformation(
            input_,
        )
        logger.debug(f"input_ size: {input_.size()}")
        logger.debug(f"embedded_input_ size: {embedded_input_.size()}")

        exp_grad = self.APExp.shap_values(
            model=score_model,
            input_tensor=embedded_input_,
            sparse_labels=target,
            batch_transformation=batch_transformation,
        )
        return exp_grad

    def _setup_attribution_prior(self, adata: anndata.AnnData) -> None:
        """Setup an expected gradient sparsity penalty

        Notes
        -----
        If `grad_activation != "input"`, sets up attributions to be
        computed at a different embedding layer within the model.
        """
        self.APExp = attrprior.AttributionPriorExplainer(
            self.train_ds,
            batch_size=self.batch_size,
            k=1,
            input_batch_index="input",
        )

        if self.grad_activation == "input":
            self._get_exp_grad = self._input_exp_grad
        elif self.grad_activation == "first_layer":
            self._get_exp_grad = self._first_layer_exp_grad
        else:
            msg = f"{self.grad_activation} is not a valid attribution layer."
            raise ValueError(msg)

        if self.attr_prior == "classwise_gini":

            def attr_criterion(model, labeled_sample, unlabeled_sample, weight):
                input_ = labeled_sample["input"]
                target = labeled_sample["output_int"]
                exp_grad = self._get_exp_grad(model, input_, target)
                # l_ap = attrprior.gini_eg(exp_grad)
                l_ap = attrprior.gini_classwise_eg(exp_grad, target)
                return l_ap

        elif self.attr_prior == "gini":

            def attr_criterion(model, labeled_sample, unlabeled_sample, weight):
                input_ = labeled_sample["input"]
                target = labeled_sample["output_int"]
                exp_grad = self._get_exp_grad(model, input_, target)
                # l_ap = attrprior.gini_eg(exp_grad)
                l_ap = attrprior.gini_eg(
                    exp_grad,
                )
                return l_ap

        elif "graph" in self.attr_prior:

            self.GP = attrprior.GeneSetGraphPrior(
                gene_sets=self.gene_sets,
                gene_names=adata.var_names.tolist(),
                weighting=self.attr_prior.split("_")[1],
            )

            def attr_criterion(model, labeled_sample, unlabeled_sample, weight):
                input_ = labeled_sample["input"]
                target = labeled_sample["output_int"]
                exp_grad = self._get_exp_grad(model, input_, target)
                return self.GP(exp_grad)

        elif (self.attr_prior is None) or (self.attr_prior.lower() == "none"):
            
            logger.info("Deactivating the attribution prior.")
            def attr_criterion(*args, **kwargs):
                return 0.0

        else:
            msg = f"{self.attr_prior} is not a valid attribution prior"
            raise ValueError(msg)

        self.attr_criterion = attr_criterion
        return

    def _zero_op(self, *args, **kwargs) -> float:
        """Returns zero"""
        return torch.zeros((1,)).to(device=self.model_device)

    def _setup_trainer(self, adata: anndata.AnnData) -> None:
        """Setup a trainer object with regularized latent variables

        Returns
        -------
        None.
        Sets `.trainer`.

        Notes
        -----
        `adata` argument is only needed to extract gene names.
        We could replace this with a `.gene_names` attribute
        in the future.
        """
        # setup the sparsity regularizer
        if self.reg_weight(self.n_epochs) > 0.0:
            self._setup_structured_sparsity(adata=adata)
        else:
            self.reg_criterion = self._zero_op
        rc = RegCritWrapper(criterion=self.reg_criterion)
        # setup the attribution prior
        self._setup_attribution_prior(adata=adata)
        if self.attr_weight(self.n_epochs) == 0.0:
            # still setup APExp, but then override the criterion
            self.attr_criterion = self._zero_op

        # parameters for the optimizer
        # we add additional parameters from the criteria (e.g. DAN parameters) as
        # needed below
        opt_params = [
            {
                "params": self.model.parameters(),
            },
        ]

        # setup the relevant loss criteria
        ce = scnym.losses.scNymCrossEntropy()
        ap = APWrapper(attr_criterion=self.attr_criterion)

        # critera: List[dict]
        # {"function": Callable, "weight": Callable, float, "validation": bool}
        criteria = [
            {
                "function": ce,
                "weight": 1.0,
                "validation": True,
                "name": "cross_entropy",
            },
            {
                "function": ap,
                "weight": self.attr_weight,
                "validation": False,
                "name": "attr_prior",
            },
            {
                "function": rc,
                "weight": self.reg_weight,
                "validation": False,
                "name": "struc_sparse",
            },
        ]

        if self.n_domains > 1:
            # setup a domain adversary loss
            da = scnym.losses.DANLoss(
                dan_criterion=scnym.losses.cross_entropy,
                model=self.model,
                n_domains=self.n_domains,
            )
            da_weight = scnym.losses.ICLWeight(
                ramp_epochs=self.domain_adv_kwargs.get("ramp_epochs", 20),
                burn_in_epochs=self.domain_adv_kwargs.get("burn_in_epochs", 0),
                max_unsup_weight=self.domain_adv_kwargs.get("max_unsup_weight", 0.1),
                sigmoid=self.domain_adv_kwargs.get("sigmoid", True),
            )
            re_zero = (self.domain_adv_kwargs.get("ramp_epochs", 20)==0) 
            bi_zero = (self.domain_adv_kwargs.get("burn_in_epochs", 0)==0)
            if re_zero and bi_zero:
                # start the DA at full weight
                da_weight = self.domain_adv_kwargs.get("max_unsup_weight", 0.1)
                logger.info(f"Starting DA at full weight: {da_weight}")

            da_dict = {
                "function": da,
                "weight": da_weight,
                "validation": False,
                "name": "domain_adv",
            }
            da_opt = {
                "params": da.dann.domain_clf.parameters(),
            }
            opt_params.append(da_opt)
            criteria.append(da_dict)

        # setup MixUp
        batch_transformers = {}
        batch_transformers["train"] = scnym.dataprep.SampleMixUp(alpha=0.3)

        optimizer = torch.optim.AdamW(
            opt_params,
            weight_decay=self.weight_decay,
            lr=self.lr,
        )

        self.trainer = scnym.trainer.MultiTaskTrainer(
            criteria=criteria,
            unsup_dataloader=None,
            model=self.model,
            batch_transformers=batch_transformers,
            optimizer=optimizer,
            dataloaders={"train": self.train_dl, "val": self.val_dl},
            n_epochs=self.n_epochs,
            out_path=self.log_dir,
            tb_writer=self.log_dir,
            patience=self.patience,
            **self.trainer_kwargs,
        )

        return

    def _setup_fit(
        self, 
        X, 
        y, 
        adata, 
        batch_size: int=512, 
        n_epochs: int=None, 
        lr: float=1e-3,
    ) -> None:
        """Setup the model for fitting or weight loading."""
        # update `n_epochs` from the default if a user passed it in `.fit()`
        self.n_epochs = n_epochs if n_epochs is not None else self.n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.model_gene_names = adata.var_names.tolist()
        self._setup_dataset(X, y, adata=adata)
        if not self.trained:
            self._setup_model()
        else:
            logger.warn("Using existing model weights as initialization.")
        # if self.trained, assume model has been preloaded
        self._setup_trainer(adata=adata)
        return

    def train_epoch(self, epoch: int) -> list:
        """Run a training epoch

        Parameters
        ----------
        epoch : int
            current epoch.

        Returns
        -------
        l_ce : float
            classification loss from `.criterion`.
        l_ss : float
            regularization loss from structured sparsity.
        loss : float
            weighted sum of the `l_ce` and `l_ss` terms.
        """
        self.model.train(True)

        epoch_losses = []
        for data in self.train_dl:

            data = self.batch_transformers["train"](data)

            input_ = data["input"].to(device=self.model_device)
            target = data["output"].to(device=self.model_device)

            outputs = self.model(input_)
            # _, preds = torch.max(outputs, dim=-1)
            _, target_int = torch.max(target, dim=-1)

            l_ce = self.criterion(outputs, target)
            l_ss = self.reg_criterion(
                self.model,
            )
            l_ap = self.attr_criterion(
                model=self.model,
                input_=input_,
                target=target_int.long(),
            )

            loss = l_ce + self.reg_weight(epoch) * l_ss + self.attr_weight(epoch) * l_ap

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_losses.append(
                [
                    l_ce.detach().item(),
                    l_ss.detach().item(),
                    l_ap.detach().item(),
                    loss.detach().item(),
                ]
            )

        d = {
            "l_ce": np.mean([x[0] for x in epoch_losses]),
            "l_ss": np.mean([x[1] for x in epoch_losses]),
            "l_ap": np.mean([x[2] for x in epoch_losses]),
            "loss": np.mean([x[3] for x in epoch_losses]),
            "reg_weight": self.reg_weight(epoch),
            "attr_weight": self.attr_weight(epoch),
        }

        return d

    @torch.no_grad()
    def val_epoch(self, epoch: int) -> list:
        """Run a validation epoch

        Parameters
        ----------
        epoch : int
            current epoch.

        Returns
        -------
        l_ce : float
            classification loss from `.criterion`.
        l_ss : float
            regularization loss from structured sparsity.
        loss : float
            weighted sum of the `l_ce` and `l_ss` terms.
        acc : float
            fraction of correct classifications.
        """
        self.model.train(False)

        epoch_losses = []
        y_hat = []
        y_true = []
        for data in self.val_dl:

            input_ = data["input"].to(device=self.model_device)
            target = data["output"].to(device=self.model_device)

            outputs = self.model(input_)

            _, preds = torch.max(outputs, dim=1)

            l_ce = self.criterion(outputs, target)
            l_ss = self.reg_criterion(self.model)

            loss = l_ce

            epoch_losses.append(
                [
                    l_ce.detach().item(),
                    l_ss.detach().item(),
                    loss.detach().item(),
                ]
            )

            _, target_labs = torch.max(target, dim=1)
            y_true.append(target_labs.detach().cpu())
            y_hat.append(preds.detach().cpu())

        y_hat = torch.cat(y_hat, dim=0).cpu().numpy()
        y_true = torch.cat(y_true, dim=0).cpu().numpy()
        acc = np.sum(y_hat == y_true) / len(y_hat)

        d = {
            "l_ce": np.mean([x[0] for x in epoch_losses]),
            "l_ss": np.mean([x[1] for x in epoch_losses]),
            "loss": np.mean([x[2] for x in epoch_losses]),
            "acc": acc,
        }

        return d

    def _write_stats(
        self,
        out_dict: dict,
        epoch: int,
        dataset: str = "train",
    ) -> None:
        """Write stats to tensorboard if desired"""
        if self.tbw is None:
            return

        for k in out_dict.keys():
            self.tbw.add_scalar(f"{k}/{dataset}", out_dict[k], epoch)
        return

    def fit(
        self,
        X,
        y,
        adata: anndata.AnnData,
        n_epochs: int = None,
        batch_size: int = 256,
        lr: float=1e-3,
    ) -> None:
        """Fit the underlying cell type classifier"""
        # convert y to int indices
        logger.info("Performing setup for fit")
        self._setup_fit(
            X=X,
            y=y,
            adata=adata,
            batch_size=batch_size,
            n_epochs=n_epochs,
            lr=lr,
        )

        # train the model
        logger.info("Entering training loop")
        self.trainer.train()
        logger.info("Training complete.")
        # save final weights
        if self.log_dir is not None:
            logger.info("\tSaving final model weights.")
            self.save(path=str(Path(self.log_dir) / Path("final_model_weights.pt")))
        self.best_weights = self.trainer.best_weights
        self.trained = True
        # set the model with the best weights
        self.model.load_state_dict(self.best_weights)
        if self.log_dir is not None:
            logger.info("\tSaving best model weights.")
            self.save(path=str(Path(self.log_dir) / Path("best_model_weights.pt")))
        return

    def save(self, path: str) -> None:
        """Save the model and relevant information"""
        torch.save(
            self.model.state_dict(),
            path,
        )
        return

    def load(self, path: str, X=None, y=None, adata=None) -> None:
        """Load model weights"""
        if not hasattr(self, "model"):
            if X is None or y is None or adata is None:
                raise ValueError("No model exists and no data provided.")
            self.batch_size = 256 # placeholder, overwritten by `fit`
            self._setup_dataset(X=X, y=y, adata=adata)
            logger.warn("Creating new model before loading model weights.")
            self._setup_model()
        device = list(self.model.parameters())[0].device
        self.model.load_state_dict(
            torch.load(
                path,
                map_location=device,
            ),
        )
        self.trained = True
        return

    @torch.no_grad()
    def transform(
        self,
        X,
        adata: anndata.AnnData,
    ) -> np.ndarray:
        """Transform data into the GRN activity and embedding space"""
        if not self.trained:
            msg = "must train the classifier with `.fit` before transforming."
            raise ValueError(msg)
        self.model.train(False)

        embeddings = []
        probabilities = []
        predictions = []
        tf_activ = []

        # embed to scNym penultimate embedding layer
        lz_02 = torch.nn.Sequential(
            *list(list(self.model.modules())[0].children())[1][:-1]
        )
        # embed to latent GRNs
        tfs = self.model.get_initial_embedder()
        # ensure models are in eval mode
        lz_02.eval()
        tfs.eval()

        # create dataloader for minibatch transformation
        # class labels are just placeholders, not used for transformation
        ds = scnym.dataprep.SingleCellDS(
            X=adata.X,
            y=np.zeros(adata.shape[0]),
        )
        dl = torch.utils.data.DataLoader(
            dataset=ds,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )

        for data in dl:

            input_ = data["input"].to(device=self.model_device)

            outputs = self.model(input_)
            _, preds = torch.max(outputs, dim=1)

            x_embed = lz_02(input_)
            x_tfs = tfs(input_)

            # probabilities.append(
            #     torch.nn.functional.softmax(outputs, dim=1).detach().cpu()
            # )
            predictions.append(preds.detach().cpu())
            embeddings.append(x_embed.detach().cpu())
            tf_activ.append(x_tfs.detach().cpu())

        # probabilities = torch.cat(probabilities, dim=0).cpu().numpy()
        embeddings = torch.cat(embeddings, dim=0).cpu().numpy()
        # predictions = torch.cat(predictions, dim=0).cpu().numpy()
        tf_activities = torch.cat(tf_activ, dim=0).cpu().numpy()

        adata.obsm["X_scnym"] = embeddings
        adata.obsm["X_grn"] = tf_activities
        adata.uns["grn_names"] = self.gene_set_names
        return tf_activities

    def _name_gradients_from_gene_sets(
        self,
        gradients: pd.DataFrame,
    ) -> pd.DataFrame:
        n_cols = gradients.shape[1]
        columns = self.gene_set_names + (n_cols - len(self.gene_set_names)) * ["other"]
        gradients.columns = columns
        return gradients

    def _get_lv_saliency(
        self,
        adata: anndata.AnnData,
        gradients: pd.DataFrame,
        target: str,
        source: str,
        scale_means: str = "cells",
    ) -> pd.DataFrame:
        """Get saliency score equivalents by taking the dot product
        of regulator mRNA expression with int. gradient for the regulatory program.

        e.g. saliency = tf_mrna_expression * tf_grn_intgrad

        Parameters
        ----------
        adata : anndata.AnnData
            [Cells, Features]
        gradients : pd.DataFrame
            [target_class_cells, LatentVariables]
        target : str
            target class in `groupby` for saliency analysis.
        scale_means : str
            method for scaling gene expression mean values before saliency scoring.
            `"group"` scales [0, 1] across means of each class.
            `"cells"` scales [0, 1] across cells then computes means for each class.
            `None` does not scale and uses raw mean expression values in `adata.X`.

        Returns
        -------
        set_saliency : pd.DataFrame
            [programs, (int_grad, mean_expr, saliency)]
        """
        detected_regulators = np.intersect1d(
            gradients.columns,
            adata.var_names,
        )
        if scale_means == "cells":
            adata = sc.pp.scale(
                adata,
                copy=True,
                zero_center=False,
                max_value=1.0,
            )

        if source == "rest":
            adata.obs["dummy"] = "rest"
            adata.obs.loc[adata.obs[self.cell_type_col]==target, "dummy"] = target

        groupby = self.cell_type_col if source != "rest" else "dummy"
        target_ad_means = ad_groupby(adata, groupby=groupby, npop=np.mean)

        if scale_means == "group":
            X = np.array(target_ad_means)
            X -= X.min(0)
            X /= X.max(0)
            target_ad_means = pd.DataFrame(
                X, columns=target_ad_means.columns, index=target_ad_means.index
            )

        index = gradients.columns.tolist()
        index = [x for x in index if x != "other"] # remove LVs without a GRN mapping
        set_saliency = pd.DataFrame(
            index=index,
            columns=["int_grad", "int_grad_std", "target_mean_expr", "source_mean_expr", "saliency"],
        )
        set_saliency["target_mean_expr"] = 0.0
        set_saliency["source_mean_expr"] = 0.0

        int_grad = gradients.loc[:, index].mean(0)
        mean_expr = np.array(
            target_ad_means.loc[target, detected_regulators],
        )
        mean_expr = mean_expr.flatten()

        set_saliency["int_grad"] = int_grad
        set_saliency["int_grad_std"] = gradients.loc[:, index].std(0)
        z_out = np.zeros(set_saliency.shape[0])
        set_saliency["int_grad_z"] = np.divide(
            np.array(set_saliency["int_grad"]),
            np.array(set_saliency["int_grad_std"]),
            where=np.array(set_saliency["int_grad_std"])>0,
        )
        set_saliency.loc[detected_regulators, "target_mean_expr"] = mean_expr

        mean_expr = np.array(
            target_ad_means.loc[source, detected_regulators],
        )
        mean_expr = mean_expr.flatten()
        set_saliency.loc[detected_regulators, "source_mean_expr"] = mean_expr

        set_saliency["saliency"] = (
            set_saliency["int_grad"] * set_saliency["target_mean_expr"]
        )
        set_saliency = set_saliency.sort_values("saliency", ascending=False)
        set_saliency["rank"] = np.arange(set_saliency.shape[0]) + 1
        return set_saliency

    def _query_ig(
        self,
        adata: anndata.AnnData,
        source: str,
        target: str,
    ) -> list:
        """Find the most likely reprogramming strategy for a given
        source and target pair.

        Parameters
        ----------
        adata : anndata.AnnData
            [Cells, Genes]
        source : str
            class name for source class.
        target : str
            class name for target class.

        Returns
        -------
        grns : list
            list of GRNs ranked by importance [high -> low].
        """
        if not self.trained:
            msg = "must train the classifier with `.fit` before querying."
            raise ValueError(msg)
        self.model.train(False)

        self.IG = scnym.interpret.IntegratedGradient(
            model=self.model.eval(),
            class_names=np.unique(adata.obs[self.cell_type_col]),
            gene_names=np.array(adata.var_names),
            grad_activation=self.grad_activation,
        )

        gradients = self.IG.get_gradients_for_class(
            adata=adata,
            groupby=self.cell_type_col,
            target_class=target,
            n_cells=100,
            M=300,
        )

        if self.grad_activation == "first_layer":
            gradients = self._name_gradients_from_gene_sets(gradients)

            saliency = self._get_lv_saliency(
                adata=adata,
                gradients=gradients,
                target=target,
                source=source,
                scale_means="cells",
            )
            sortby = {
                "saliency": "saliency",
                "intgrad_lv": "int_grad",
            }.get(self.scoring_metric, None)
            saliency = saliency.sort_values(sortby, ascending=False)
        else:
            # gradients are on individual regulators
            saliency = gradients.mean(0).sort_values(ascending=False)

        self.saliency = saliency
        grns = saliency.index.tolist()
        return grns

    def _query_eg(
        self,
        adata: anndata.AnnData,
        source: str,
        target: str,
        n_batches: int = 100,
        n_cells: int = 200,
        only_source_reference: bool = False,
    ) -> list:
        """Find the most likely reprogramming strategy for a given
        source and target pair using expected gradients.

        Parameters
        ----------
        adata : anndata.AnnData
            [Cells, Genes]
        source : str
            class name for source class.
        target : str
            class name for target class.
        n_batches : int
            number of reference batches to draw for each target sample.
        n_cells : int
            number of target samples to use for E[G] estimation.
            if `None`, uses all available samples.
        only_source_reference : bool
            use only source cells as reference examples.

        Returns
        -------
        grns : list
            list of GRNs ranked by importance [high -> low].
        """
        if not self.trained:
            msg = "must train the classifier with `.fit` before querying."
            raise ValueError(msg)
        self.model.train(False)

        # setup exp grad if needed
        if not hasattr(self, "_get_exp_grad"):
            if self.grad_activation == "input":
                self._get_exp_grad = self._input_exp_grad
            elif self.grad_activation == "first_layer":
                self._get_exp_grad = self._first_layer_exp_grad
            else:
                msg = f"{self.grad_activation} is not a valid attribution layer."
                raise ValueError(msg)

        source_bidx = adata.obs[self.cell_type_col] == source
        target_bidx = adata.obs[self.cell_type_col] == target

        if source == "rest":
            # use everything other than the target
            source_bidx = ~target_bidx

        # regenerate labels in case the query dataset is different from the
        # training dataset
        target_y = np.array(
            [self.y_categories.tolist().index(target)] * sum(target_bidx),
            dtype=np.int32,
        )
        source_y = (
            np.array(
                [self.y_categories.tolist().index(source)] * sum(source_bidx),
                dtype=np.int32,
            )
            if source != "rest"
            else
            np.zeros(sum(source_bidx)).astype(np.int32)
        )

        source_adata = adata[source_bidx, :].copy()
        target_adata = adata[target_bidx, :].copy()
        logging.info(f"Subset adata to {target_adata.shape[0]} target cells.")

        if n_cells is not None:
            # resample if `n_cells > n_target_cells`
            target_idx = np.random.choice(
                np.arange(target_adata.shape[0]),
                size=n_cells,
                replace=n_cells > target_adata.shape[0],
            ).astype(np.int32)
        else:
            target_idx = np.arange(target_adata.shape[0])

        target_ds = scnym.dataprep.SingleCellDS(
            X=target_adata.X[target_idx],
            y=target_y[target_idx],
        )
        logging.info(
            f"Using {target_ds.X.shape[0]} target cells for expgrad estimation."
        )
        # save the cell indices in attributes
        self._query_cell_obs_names = pd.DataFrame(
            {
                "names": source_adata.obs_names.tolist()
                + target_adata.obs_names[target_idx].tolist(),
                "dataset": ["source"] * source_adata.shape[0]
                + ["target"] * len(target_idx),
            },
        )

        # make sure the source dataset has at least as many examples as
        # the target by replicating at random
        n_reps = int(np.ceil(sum(target_bidx) / sum(source_bidx)))
        source_indices = np.arange(source_adata.X.shape[0])
        source_indices = np.tile(source_indices, (n_reps,))
        source_ds = scnym.dataprep.SingleCellDS(
            X=source_adata.X[source_indices],
            y=source_y[source_indices],
        )

        batch_size = min(self.batch_size, len(target_idx))
        target_dl = torch.utils.data.DataLoader(
            target_ds,
            batch_size=batch_size,
            shuffle=False,
            drop_last=self.batch_size == batch_size,
        )

        # use only the source samples as references if specified
        # otherwise, use the whole training set
        ref_ds = source_ds if only_source_reference else self.train_ds
        self.APExp = attrprior.AttributionPriorExplainer(
            ref_ds,
            batch_size=batch_size,
            k=1,
            input_batch_index="input",
        )
        logger.debug("Set up Attribution Prior Explainer")

        logger.info("Estimating Expected Gradients")
        gradients_by_batch = []
        for input_batch in target_dl:
            batch_grads = []
            input_ = input_batch["input"].to(device=self.model_device)
            _, labels = torch.max(input_batch["output"], dim=1)
            labels = labels.to(device=self.model_device).long()
            # for each input, use `n_batches` different random references
            for i in range(n_batches):
                s = time.time()
                logger.debug(f"gradient batch {i}")
                g = self._get_exp_grad(
                    self.model,
                    input_,
                    target=labels,
                )
                g = g.detach()
                batch_grads.append(g.detach().cpu())
                e = time.time()
                logger.debug(f"time: {e-s} secs")
            # [Obs, Features, estimation_batch]
            batch_grads = torch.stack(batch_grads, dim=-1)
            batch_grads = torch.mean(batch_grads, dim=-1)

            gradients_by_batch.append(batch_grads)
        gradients = torch.cat(gradients_by_batch, dim=0)
        gradients = gradients.detach().cpu().numpy()

        gradients = pd.DataFrame(
            gradients,
            index=target_adata.obs_names[target_idx][: gradients.shape[0]],
        )
        if gradients.shape[1] == len(adata.var_names):
            gradients.columns = adata.var_names.tolist()

        if self.grad_activation == "first_layer":
            gradients = self._name_gradients_from_gene_sets(gradients)

            saliency = self._get_lv_saliency(
                adata=adata,
                gradients=gradients,
                target=target,
                source=source,
                scale_means="cells",
            )
            sortby = {
                "saliency": "saliency",
                "intgrad_lv": "int_grad",
                "intgrad_lv_z": "int_grad_z",
            }.get(self.scoring_metric, None)
            saliency = saliency.sort_values(sortby, ascending=False)
        else:
            # gradients are on individual regulators
            saliency = gradients.mean(0).sort_values(ascending=False)
            saliency.columns = ["int_grad"]

        self.saliency = saliency
        self.gradients = gradients
        grns = saliency.index.tolist()
        return grns

    def query(
        self,
        adata: anndata.AnnData,
        source: str,
        target: str,
        **kwargs,
    ) -> list:
        """Find the most likely reprogramming strategy for a given
        source and target pair.

        Parameters
        ----------
        adata : anndata.AnnData
            [Cells, Genes]
        source : str
            class name for source class.
        target : str
            class name for target class.

        Returns
        -------
        grns : list
            list of GRNs ranked by importance [high -> low].

        See Also
        --------
        _query_ig
        _query_eg
        """
        if torch.cuda.is_available():
            logger.warning("Moved model to CUDA compute device.")
            self.model = self.model.cuda()

        if getattr(self, "query_kwargs", None) is not None:
            kwargs.update(getattr(self, "query_kwargs"))
            logger.info("\tLoaded `query_kwargs`")

        if self.interp_method == "int_grad":
            return self._query_ig(
                adata=adata,
                source=source,
                target=target,
                **kwargs,
            )
        elif self.interp_method == "exp_grad":
            return self._query_eg(
                adata=adata,
                source=source,
                target=target,
                **kwargs,
            )
        else:
            msg = f"{self.interp_method} is not a valid interp_method."
            raise ValueError(msg)

    def save_query(
        self,
        path: str,
    ) -> None:
        """Save intermediary representations generated during a
        `query` call"""
        if path is None:
            return
        # save query outputs
        saliency_path = str(Path(path) / Path("saliency.csv"))
        self.saliency.to_csv(saliency_path)
        gradients_path = str(Path(path) / Path("gradients.csv"))
        self.gradients.to_csv(gradients_path)
        obs_names_path = str(Path(path) / Path("obs_names.csv"))
        self._query_cell_obs_names.to_csv(obs_names_path)
        return


class PathFauxgrify(Pathfinder, Fauxgrify):

    only_source_reference = True

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """Initialize the PathFauxgrify model.

        Notes
        -----
        Uses multiple inheritance to setup a Pathfinder model
        alongside Fauxgrify heuristics.
        Overrides `_get_deg_score` to use expected gradients on the
        input
        Overrides `.query` manually to allow for additional
        documentation.

        See Also
        --------
        Pathfinder.__init__
        .baseline.Fauxgrify.__init__
        """
        super(PathFauxgrify, self).__init__(*args, **kwargs)
        self._fit_before_query = True
        self.net_score_preprocess = self._relu
        return

    def _relu(self, x: np.ndarray) -> np.ndarray:
        """Rectified linear unit activation on a copy of the input array"""
        y = np.array(x).copy()
        y[y < 0] = 0
        return y

    def __getstate__(self):
        """Override the default get state to avoid pickling unnecessary objects while
        multiprocessing network scoring."""
        d = dict(self.__dict__)
        # only keep keys needed by `Fauxgrify._get_target_genes`
        # and `._calc_single_net_score`
        keys2keep = (
            "network_scores",
            "precomp_target_genes",
            "gene_sets",
            "gene_set_names",
            "reg_graph",
            "network_degree",
            "_get_target_genes",
        )
        d = {k: v for k, v in d.items() if k in keys2keep}
        return d

    def _get_gene_score(self, adata, source, target, **kwargs) -> pd.DataFrame:
        """Wrap gradient based gene scores in a DataFrame for
        Fauxgrify processing"""
        # get mean expression levels
        logger.info("Getting gene scores in PathFauxgrify model.")
        source_mean = pd.DataFrame(
            {
                "source_mean": np.array(
                    adata[adata.obs[self.cell_type_col] == source].X.mean(0)
                ).flatten()
            },
            index=adata.var_names,
        )
        target_mean = pd.DataFrame(
            {
                "target_mean": np.array(
                    adata[adata.obs[self.cell_type_col] == target].X.mean(0)
                ).flatten()
            },
            index=adata.var_names,
        )
        # setup the model to use input expected gradients,
        # regardless of training
        self._get_exp_grad = self._input_exp_grad
        # sets `self.saliency` with a pd.DataFrame [Genes, (int_grad,)]
        grns = self._query_eg(
            adata=adata,
            source=source,
            target=target,
            only_source_reference=self.only_source_reference,
            **kwargs,
        )
        gene_scores = pd.DataFrame(
            {
                "gene_score": np.array(self.saliency),
            },
            index=self.saliency.index.tolist(),
        )
        gene_scores.columns = ["gene_score"]
        self.gene_scores = gene_scores
        # add mean expr level in source and target
        gene_scores["source_mean"] = np.array(
            source_mean.loc[gene_scores.index, "source_mean"],
        )
        gene_scores["target_mean"] = np.array(
            target_mean.loc[gene_scores.index, "target_mean"],
        )
        return gene_scores

    def query(
        self,
        adata: anndata.AnnData,
        source: str,
        target: str,
        **kwargs,
    ) -> list:
        """Find reprogramming GRNs using Mogrify-inspired heuristics.

        Parameters
        ----------
        adata : anndata.AnnData
            [Cells, Genes]
        source : str
            class name for source class.
        target : str
            class name for target class.
        kwargs : dict
            passed to `_get_gene_score()`.

        Returns
        -------
        grns : list
            ranked GRNs, high to low.
        """
        if torch.cuda.is_available():
            logger.info("Moved model to CUDA compute device.")
            self.model = self.model.cuda()
        # validate inputs
        self._check_source_target(adata, source, target)
        # get gene scores (wraps expected gradient estimator)
        gene_scores = self._get_gene_score(
            adata,
            source,
            target,
            **kwargs,
        )
        # also get differential expression scores for gene level ranking
        # of regulatory molecules
        deg_scores = self._get_deg_score(
            adata,
            source,
            target,
        )
        self.gene_scores = gene_scores
        # get network scores
        network_scores = self._get_network_score(gene_scores=gene_scores)
        # set gene scores for individual regulators to their differential
        # expression, rather than expected gradient score
        regs_in_dex = np.intersect1d(
            self.gene_set_names,
            deg_scores.index,
        )
        network_scores["eg_gene_score"] = np.array(network_scores["gene_score"])
        for k in ("gene_score",):
            network_scores.loc[regs_in_dex, k] = np.array(
                deg_scores.loc[regs_in_dex, k],
                dtype=np.float64,
            )
        # rank network scores
        network_scores = self._rank_from_scores(network_scores=network_scores)
        self.network_scores = network_scores.copy()
        # set up a matrix quantifying overlap of regulons
        self._construct_redundancy_matrix()
        # prune redundant and low scoring GRNs
        network_scores = self._prune_network_results(network_scores=network_scores)
        self.network_scores_pruned = network_scores.copy()
        logger.info(f"network_scores\n{network_scores.head(15)}")
        grns = network_scores.index.tolist()
        return grns
