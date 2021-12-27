"""Tools for interpreting trained scNym models"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from scipy import sparse
import anndata

# self
from .utils import build_classification_matrix, get_adata_asarray
from . import dataprep
from . import attributionpriors as attrprior

# stdlib
import typing
import copy
import warnings
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class Salience(object):
    """
    Performs backpropogation to compute gradients on a target
    class with regards to an input.

    Notes
    -----
    Saliency analysis computes a gradient on a target class
    score :math:`f_i(x)` with regards to some input :math:`x`.


    .. math::

        S_i = \frac{\partial f_i(x)}{\partial x}
    """

    def __init__(
        self,
        model: nn.Module,
        class_names: np.ndarray,
        gene_names: np.ndarray = None,
        layer_to_hook: int = None,
        verbose: bool = False,
    ) -> None:
        """
        Performs backpropogation to compute gradients on a target
        class with regards to an input.

        Parameters
        ----------
        model : torch.nn.Module
            trained scNym model.
        class_names : np.ndarray
            list of str names matching output nodes in `model`.
        gene_names : np.ndarray, optional
            gene names for the model.
        layer_to_hook : int
            index of the layer from which to record gradients.
            defaults to the gene level input features.

        Returns
        -------
        None.
        """
        # ensure class names are unique for each output node
        if len(np.unique(class_names)) != len(class_names):
            msg = "`class_names` must all be unique."
            raise ValueError(msg)

        self.class_names = np.array(class_names)
        self.n_classes = len(class_names)
        self.verbose = verbose

        # load model into CUDA compute if available
        if torch.cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model
        # ensure we're not in training mode
        self.model = self.model.eval()

        self.gene_names = gene_names

        if layer_to_hook is None:
            self._hook_first_layer_gradients()
        else:
            self._hook_nth_layer_gradients(n=layer_to_hook)
        return

    def _hook_first_layer_gradients(self):
        """Set up hooks to record gradients from the first linear
        layer into a target tensor.

        References
        ----------
        https://pytorch.org/docs/stable/nn.html#torch.nn.Module.register_backward_hook
        """

        def _record_gradients(module, grad_in, grad_out):
            """Record gradients of a layer with the correct input
            shape"""
            self.gradients = grad_in[1]
            if self.verbose:
                print([x.size() if x is not None else "None" for x in grad_in])
                print("Hooked gradients to: ", module)

        for module in self.model.modules():
            if isinstance(module, nn.Linear) and module.in_features == len(
                self.gene_names
            ):
                module.register_backward_hook(_record_gradients)
        return

    def _hook_nth_layer_gradients(self, n: int):
        """Set up hooks to record gradients from an arbitrary layer.

        References
        ----------
        https://pytorch.org/docs/stable/nn.html#torch.nn.Module.register_backward_hook
        """

        def _record_gradients(module, grad_in, grad_out):
            """Record gradients of a layer with the correct input
            shape"""
            self.gradients = grad_in[1]
            if self.verbose:
                print([x.size() if x is not None else "None" for x in grad_in])
                print("Hooked gradients to: ", module)

        module = list(self.model.modules())[n]
        module.register_backward_hook(_record_gradients)
        return

    def _guided_backprop_hooks(self):
        """Set up forward and backward hook functions to perform
        "Guided backpropogation"

        Notes
        -----
        Guided backpropogation only passes positive gradients upward through the network.

        Normal backprop:

        .. math::

            f_i^{(l + 1)} = ReLU(f_i^{(l)})

            R_i^{(l)} = (f_i^{(l)} > 0) \cdot R_i^{(l+1)}

        where

        .. math::

            R_i^{(l + 1)} = \frac{\partial f_{out}}{\partial f_i^{l + 1}}


        By contrast, guided backpropogation only passes gradient values where both
        the activates :math:`f_i^{(l)}` and the gradients :math:`R_i^{(l + 1)}` are
        greater than :math:`0`.


        References
        ----------
        https://arxiv.org/pdf/1412.6806.pdf

        https://pytorch.org/docs/stable/nn.html#torch.nn.Module.register_forward_hook
        https://pytorch.org/docs/stable/nn.html#torch.nn.Module.register_backward_hook
        """

        def _record_relu_outputs(module, in_, out_):
            """Store the outputs to each ReLU layer"""
            self.rectified_outputs.append(
                out_,
            )
            self.store_rectified_outputs.append(
                out_,
            )

        def _clamp_grad(module, grad_in, grad_out):
            """Clamp ReLU gradients to [0, inf] and return a
            new gradient to be used in subsequent outputs.
            """
            self.store_grad.append(grad_in[0])

            grad = grad_in[0].clamp(min=0.0)
            self.store_clamped_grad.append(grad)

            # here we pop the outputs off to ensure that the
            # final output is always the current ReLU layer
            # we're investigating
            last_relu_output = self.rectified_outputs.pop()
            last_relu_output = copy.copy(last_relu_output)
            last_relu_output[last_relu_output > 0] = 1
            rectified_grad = last_relu_output * grad

            self.store_rectified_grad.append(rectified_grad)
            return (rectified_grad,)

        self.store_rectified_outputs = []
        self.store_grad = []
        self.store_clamped_grad = []

        for _, module in self.model.named_modules():
            if isinstance(module, nn.ReLU):
                module.register_forward_hook(_record_relu_outputs)
                module.register_backward_hook(_clamp_grad)

        return

    def get_saliency(
        self,
        x: torch.FloatTensor,
        target_class: str,
        guide_backprop: bool = False,
    ) -> torch.FloatTensor:
        """Compute the saliency of a target class on an input
        vector `x`.

        Parameters
        ----------
        x : torch.FloatTensor
            [1, Genes] vector of gene expression.
        target_class : str
            class in `.class_names` for which to compute gradients.
        guide_backprop : bool
            perform "guided backpropogation" by clamping gradients
            to only positive values at each ReLU.
            see: https://arxiv.org/pdf/1412.6806.pdf

        Returns
        -------
        salience : torch.FloatTensor
            gradients on `target_class` with respect to `x`.
        """
        if target_class not in self.class_names:
            msg = f"{target_class} is not in `.class_names`"
            raise ValueError(msg)

        target_idx = np.where(target_class == self.class_names)[0].astype(np.int)
        target_idx = int(target_idx)

        self.model.zero_grad()

        if guide_backprop:
            self.rectified_outputs = []
            self.store_rectified_grad = []
            self._guided_backprop_hooks()

        # store gradients on the input
        if torch.cuda.is_available():
            x = x.cuda()
        x.requires_grad = True

        # module hook will record gradients here
        self.gradients = torch.zeros_like(x)

        # forward pass
        output = self.model(x)

        # create a [N, C] tensor to store gradients
        target = torch.zeros_like(output)
        # set the target class to `1`, creating a one-hot
        # of the target class
        target[:, target_idx] = 1

        # compute gradients with backprop
        output.backward(
            gradient=target,
        )

        # detach from the graph and move to main memory
        target = target.detach().cpu()

        return self.gradients

    def rank_genes_by_saliency(
        self,
        **kwargs,
    ) -> np.ndarray:
        """
        Rank genes by saliency for a target class and input.

        Passes **kwargs to `.get_saliency` and uses the output
        to rank genes.

        Returns
        -------
        ranked_genes : np.ndarray
            gene names with high saliency, ranked highest to
            lowest.
        """
        s = self.get_saliency(**kwargs)
        sort_idx = torch.argsort(s)
        idx = sort_idx[0].numpy()[::-1]
        return self.gene_names[idx.astype(np.int)]


class IntegratedGradient(object):
    def __init__(
        self,
        model: nn.Module,
        class_names: typing.Union[list, np.ndarray],
        gene_names: typing.Union[list, np.ndarray] = None,
        grad_activation: str = "input",
        verbose: bool = False,
    ) -> None:
        """Performs integrated gradient computations for feature attribution
        in scNym models.
        
        Parameters
        ----------
        model : torch.nn.Module
            trained scNym model.
        class_names : list or np.ndarray
            list of str names matching output nodes in `model`.
        gene_names : list or np.ndarray, optional
            gene names for the model.
        grad_activation : str
            activations where gradients should be collected.
            default "input" collects gradients at the level of input features.
        verbose : bool
            verbose outputs for stdout.
        
        Returns
        -------
        None.
        
        Notes
        -----
        Integrated gradients are computed as the path integral between a "baseline"
        gene expression vector (all 0 counts) and an observed gene expression vector.
        The path integral is computed along a straight line in the feature space.
        
        Stated formally, we define a our baseline gene expression vector as :math:`x`,
        our observed vector as :math:`x'`, an scnym model :math:`f(\cdot)`, and a 
        number of steps :math:`M` for approximating the integral by Reimann sums.
        
        The integrated gradient :math:`\int \nabla` for a feature :math:`x_i` is then
        
        .. math::
        
            r = \sum_{m=1}^M \partial f(x' + \frac{m}{M}(x - x')) / \partial x_i \\
            \int \nabla_i = (x_i' - x_i) \frac{1}{M} r
        """
        self.model = copy.deepcopy(model)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            print("Model loaded on CUDA compute device.")
        self.model.zero_grad()
        for param in self.model.parameters():
            param.requires_grad = False

        # get gradients on the specified layer activation if
        # the specified layer is not "input"
        self.grad_activation = grad_activation

        if grad_activation == "input":
            self.get_grad = self._get_grad_input
        elif grad_activation == "first_layer":
            self.get_grad = self._get_grad_first_layer
            self.input2first = nn.Sequential(*list(model.modules())[3:7])
            self.first2output = nn.Sequential(*list(model.modules())[7:])
        else:
            msg = f"`grad_activation={grad_activation}` is not implemented."
            raise NotImplementedError(msg)

        self.class_names = class_names
        self.gene_names = gene_names
        self.verbose = verbose
        self.grads_for_class = {}

        if type(self.class_names) == np.ndarray:
            self.class_names = self.class_names.tolist()

        return

    def _get_grad_input(
        self,
        x: torch.Tensor,
        target_class: str,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """Get the gradient on the observed features with respect
        to a target class.

        Parameters
        ----------
        x : torch.Tensor
            [Batch, Features] input tensor.
        target_class : str
            target class for gradient computation.

        Returns
        -------
        grad : torch.Tensor
            [Batch, Features] feature gradients with respect to the
            target class.
        target : torch.Tensor
            [Batch,] value of the target class score.
        """
        target_idx = self.class_names.index(target_class)

        # store gradients on the input
        if torch.cuda.is_available():
            x = x.cuda()
        x.requires_grad = True

        # forward pass through the model
        output = self.model(x)
        sm_output = F.softmax(output, dim=-1)

        # get the softmax output on the target class for each
        # observation as a loss
        index = torch.ones(output.size(0)).view(-1, 1) * target_idx
        index = index.long()
        index = index.to(device=sm_output.device)
        # `.gather(dim, index)` takes a dimension number and a tensor
        # of indices size [Batch,] where each val is an integer index
        # grabs the specific element for each observation along the given dim.
        target = sm_output.gather(1, index)

        # zero any existing gradients
        self.model.zero_grad()
        if x.grad is not None:
            x.grad.zero_()
        target.backward()

        grad = x.grad.detach().cpu()

        return grad, target

    def _catch_grad(self, grad) -> None:
        """Hook to catch gradients from an activation
        of interest."""
        self.caught_grad = grad.detach()
        return

    def _get_grad_first_layer(
        self,
        x: torch.Tensor,
        target_class: str,
    ):
        """Get the gradient on the first layer activations.

        Parameters
        ----------
        x : torch.Tensor
            [Batch, Features] input tensor. e.g. first layer
            embedding coordinates to pass to the rest of the model.
        target_class : str
            target class for gradient computation.

        Returns
        -------
        grad : torch.Tensor
            [Batch, Features] feature gradients with respect to the
            target class.
        target : torch.Tensor
            [Batch,] value of the target class score.
        """
        target_idx = self.class_names.index(target_class)
        # store gradients on the input
        if torch.cuda.is_available():
            x = x.cuda()
        x.requires_grad = True

        # forward through the activation embedder
        x.register_hook(self._catch_grad)
        # forward through to outputs
        output = self.first2output(x)
        sm_output = F.softmax(output, dim=-1)

        # get the softmax output on the target class for each
        # observation as a loss
        index = torch.ones(output.size(0)).view(-1, 1) * target_idx
        index = index.long()
        index = index.to(device=sm_output.device)
        # `.gather(dim, index)` takes a dimension number and a tensor
        # of indices size [Batch,] where each val is an integer index
        # grabs the specific element for each observation along the given dim.
        target = sm_output.gather(1, index)

        # zero any existing gradients
        self.model.zero_grad()
        if x.grad is not None:
            x.grad.zero_()

        target.backward()
        grad = self.caught_grad

        return grad, target

    def _check_integration(
        self,
        integrated_grad: torch.Tensor,
    ) -> bool:
        """Check that the approximation of the path integral is appropriate.
        If we used a sufficient number of steps in the Reimann sum, we should
        find that the gradient sum is roughly equivalent to the difference in
        class scores for the baseline vector and target vector.
        """
        score_difference = self.raw_scores[-1] - self.raw_scores[0]
        check = torch.isclose(
            integrated_grad.sum(),
            score_difference,
            rtol=0.1,
        )
        if not check:
            msg = "integrated gradient magnitude does not match the difference in scores.\n"
            msg += f"magnitude {integrated_grad.sum().item()} vs. {score_difference.item()}.\n"
            msg += "consider using more steps to estimate the path integral."
            warnings.warn(msg)
        return check

    def get_integrated_gradient(
        self,
        x: torch.Tensor,
        target_class: str,
        M: int = 300,
        baseline: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute the integrated gradient for a single observation.

        Parameters
        ----------
        x : torch.Tensor
            [Features,] input tensor.
        target_class : str
            class in `self.class_names` for optimization.
        M : int
            number of gradient steps to use when approximating
            the path integral.
        baseline : torch.Tensor
            [Features,] baseline gene expression vector to use.
            if `None`, uses the `0` vector.

        Returns
        -------
        integrated_grad : torch.Tensor
            [Features,] integrated gradient tensor.

        Notes
        -----
        1. Define a difference between the baseline input and observation.
        2. Approximate a linear path between the baseline and observation
        with `M` steps.
        3. Compute the gradient at each step in the path.
        4. Sum gradients across steps and divide by number of steps.
        5. Elementwise multiply with input features as in saliency.
        """
        if baseline is None:
            n_dims = (
                len(self.gene_names)
                if self.grad_activation == "input"
                else self.model.n_hidden_init
            )

            if self.verbose:
                print("Using the 0-vector as a baseline.")
            base = self.baseline_input = torch.zeros((1, n_dims)).float()
        else:
            base = self.baseline_input = baseline
            if base.dim() > 1 and base.size(0) != 1:
                msg = "baseline must be a single gene expression vector"
                raise ValueError(msg)
            base = base.view(1, -1)

        self.target_class = target_class

        if x.dim() > 1 and x.size(0) == 1:
            # tensor has an empty batch dimension, flatten it
            x = x.view(-1)

        # create a batch of observations where each observation is
        # a single step along the path integral
        path = base.repeat((M, 1))

        # if `first_layer` activations are used, x_activ is the relevant
        # activation setting for saliency
        if self.grad_activation == "first_layer":
            x = x.to(device=list(self.input2first.parameters())[0].device)
            x_rel = self.input2first(x.view(1, -1)).detach().cpu()
        else:
            x_rel = x
        self.x_rel = x_rel

        # create a tensor marking the "step number" for each observation
        step = ((x_rel - base) / M).view(1, -1)
        step_coord = torch.arange(1, M + 1).view(-1, 1).repeat((1, path.size(1)))

        # add the correct number of steps to fill the path tensor
        path += step * step_coord

        if self.verbose:
            print("baseline", base.size())
            print(base.sort())
            print("observation", x.size())
            print(x.sort())
            print()
            print("step : ", step.size())
            print(step)
            print("step_coord : ", step_coord.size())
            print(step_coord)
            print("path : ", path.size())
            print(path[0].sort())
            print("-" * 3)
            print(path[-1].sort())

        # compute the gradient on the input at each step
        # along the path
        grad_dim = (
            path.size(1)
            if self.grad_activation == "input"
            else self.model.n_hidden_init
        )
        gradients = torch.zeros((path.size(0), grad_dim))
        scores = torch.zeros(path.size(0))

        for m in range(M):
            gradients[m, :], target_scores = self.get_grad(
                path[m, :].view(1, -1),
                self.target_class,
            )
            scores[m] = target_scores

        self.raw_gradients = gradients
        self.raw_scores = scores
        self.path = path

        # sum gradients and normalize by step number
        integrated_grad = x_rel * (gradients.sum(0) / M)

        self._check_integration(integrated_grad)

        return integrated_grad

    def get_gradients_for_class(
        self,
        adata: anndata.AnnData,
        groupby: str,
        target_class: str,
        reference_class: str = None,
        n_cells: int = None,
        *args,
        **kwargs,
    ) -> pd.DataFrame:
        """Get integrated gradients for a target class given
        an AnnData experiment.

        Parameters
        ----------
        adata : anndata.AnnData
            [Cells, Features] experiment.
        groupby : str
            column in `adata.obs` containing class names.
        target_class : str
            class in `self.class_names` and `adata.obs[groupby]`
            for optimization.
        reference_class : str
            reference class in `self.class_names`. "all" uses all
            non-target classes as a reference.
        n_cells : int
            number of cells to use to compute a characteristic
            integrated gradient.
            if `None`, uses all cells.
        *args, **kwargs : dict
            passed to `self.get_integrated_gradient`.

        Returns
        -------
        gradients : pd.DataFrame
            [Cells, Features] integrated gradients.
        Sets `self.grads_for_class[target_class]` with the value
        of `gradients`.

        See Also
        --------
        get_integrated_gradient
        """
        if not np.all(adata.var_names == self.gene_names):
            # gene names don't match, check if IG names are a subset
            shared_genes = np.intersect1d(
                adata.var_names,
                self.gene_names,
            )
            if len(shared_genes) < len(self.gene_names):
                # some genes are missing
                msg = "Not all genes in `gene_names` were found in `adata`."
                raise ValueError(msg)
            else:
                # subset adata to the gene set used
                # this will also handle gene name permutations
                adata = adata[:, self.gene_names]

        if groupby not in adata.obs_keys():
            msg = f"{groupby} not in `adata.obs` columns."
            raise ValueError(msg)

        groups = np.unique(adata.obs[groupby])
        if target_class not in groups:
            msg = f"`{target_class}` is not a class in `{groupby}`"
            raise ValueError(msg)
        if target_class not in self.class_names:
            msg = f"`{target_class}` is not a class in `self.class_names`"
            raise ValueError(msg)

        # get the indices for cells of the target class
        cell_idx = np.where(adata.obs[groupby] == target_class)[0].astype(np.int)
        if n_cells is not None:
            if n_cells < len(cell_idx):
                # subset if a specific number of cells was specified
                cell_idx = np.random.choice(
                    cell_idx,
                    size=n_cells,
                    replace=False,
                )
                msg = f"Using {n_cells} cells for integrated gradient analysis."
                logger.debug(msg)
            else:
                msg = f"n_cells {n_cells} > n_cells_in_class {len(cell_idx)}.\n"
                msg += "Using all available cells."
                logger.warning(msg)

        # compute integrated gradients
        grads = []
        for i, idx in enumerate(cell_idx):
            x = adata.X[idx, :]
            if type(x) == np.matrix:
                x = np.array(x)
            if type(x) == sparse.csr_matrix:
                x = x.toarray()
            if type(x) != np.ndarray:
                msg = "gene vector was not coerced to np.ndarray"
                raise TypeError(msg)
            x = x.flatten()
            x = torch.from_numpy(x).float()

            g = self.get_integrated_gradient(
                x=x,
                target_class=target_class,
                *args,
                **kwargs,
            )
            grads.append(g.view(-1))

            logger.debug(f"x size: {x.size()}")
            logger.debug(f"g size: {g.size()}")

        G = torch.stack(grads, dim=0).cpu().numpy()

        if self.grad_activation == "input":
            col_names = self.gene_names
        else:
            col_names = [f"z_{i}" for i in range(G.shape[1])]

        gradients = pd.DataFrame(
            G,
            columns=col_names,
            index=adata.obs_names[cell_idx],
        )

        self.grads_for_class[target_class] = gradients

        return gradients

    def get_top_features_from_gradients(
        self,
        target_class: str = None,
        gradients: pd.DataFrame = None,
    ) -> np.ndarray:
        """Get the top features from a set of pre-computed integrated
        gradients.

        Parameters
        ----------
        target_class : str
            target class with gradients stored in `self.grads_for_class[target_class]`.
        gradients : pd.DataFrame
            [Cells, Features] integrated gradients to use. If provided, supercedes
            `target_class`.

        Returns
        -------
        top_features : np.ndarray
            [Features,] sorted [High, Low] values.
            i.e. `top_features[0]` is the top feature.
        """
        if target_class is None and gradients is None:
            raise ValueError("must provide `gradients` or `target_class`")

        # `if gradients is not None`, use gradients instead of
        # the stored gradients regardless of whether or not
        # target_class as provided
        if gradients is None:
            gradients = self.grads_for_class[target_class]
            logger.debug(f"Using stored gradients for {target_class}")

        grad_means = gradients.mean(0)
        sort_idx = np.argsort(grad_means)[::-1]  # high to low

        top_features = self.gene_names[sort_idx]
        return top_features


class ExpectedGradient(object):
    def __init__(
        self,
        model: nn.Module,
        class_names: typing.Union[list, np.ndarray],
        gene_names: typing.Union[list, np.ndarray] = None,
        verbose: bool = False,
    ) -> None:
        """Performs expected gradient computations for feature attribution
        in scNym models.
        
        Parameters
        ----------
        model : torch.nn.Module
            trained scNym model.
        class_names : list or np.ndarray
            list of str names matching output nodes in `model`.
        gene_names : list or np.ndarray, optional
            gene names for the model.
        verbose : bool
            verbose outputs for stdout.
        
        Returns
        -------
        None.
        
        Notes
        -----
        Integrated gradients are computed as the path integral between a "baseline"
        gene expression vector (all 0 counts) and an observed gene expression vector.
        The path integral is computed along a straight line in the feature space.
        
        Stated formally, we define a our baseline gene expression vector as :math:`x`,
        our observed vector as :math:`x'`, an scnym model :math:`f(\cdot)`, and a 
        number of steps :math:`M` for approximating the integral by Reimann sums.
        
        The integrated gradient :math:`\int \nabla` for a feature :math:`x_i` is then
        
        .. math::
        
            r = \sum_{m=1}^M \partial f(x' + \frac{m}{M}(x - x')) / \partial x_i \\
            \int \nabla_i = (x_i' - x_i) \frac{1}{M} r
        """
        self.model = model
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            logger.info("Model loaded on CUDA device for E[Grad] estimation.")
        self.model.zero_grad()
        for param in self.model.parameters():
            param.requires_grad = False

        self.model_device = list(self.model.parameters())[0].device

        self.class_names = np.array(class_names)
        self.gene_names = np.array(gene_names)
        self.verbose = verbose
        self.grads_for_class = {}
        # define the values for `source` that will trigger using all data as the
        # reference dataset
        self.background_vals = (
            "all",
            None,
        )

        return

    def _check_inputs(
        self,
        adata: anndata.AnnData,
        source: str,
        target: str,
        cell_type_col: str,
    ) -> anndata.AnnData:
        """Check that inputs match model expectations.

        Parameters
        ----------
        adata : anndata.AnnData
            [Cells, Genes]
        source : str
            class name for source class.
        target : str
            class name for target class.
        cell_type_col : str
            column in `adata.obs` containing cell type labels.

        Returns
        -------
        adata : anndata.AnnData
            [Cells, len(self.gene_names)] experiment.
            modifies anndata to match model training gene names
            if necessary.
        """
        # check cell type arguments
        if cell_type_col not in adata.obs.columns:
            msg = f"{cell_type_col} is not a column in `adata.obs`"
            raise ValueError(msg)
        self.cell_type_col = cell_type_col

        cell_types = np.unique(adata.obs[self.cell_type_col])
        if source not in cell_types and source not in self.background_vals:
            msg = f"{source} not in the detected cell types or background values."
            raise ValueError(msg)
        if target not in cell_types:
            msg = f"{target} not in the detected cell types."
            raise ValueError(msg)

        # check that genes match the training gene names
        match = np.all(np.array(adata.var_names) == np.array(self.gene_names))
        if not match:
            msg = "Gene names for model and `adata` query do not match.\n"
            msg += "\t Coercing..."
            logger.warn(msg)
            X = build_classification_matrix(
                X=get_adata_asarray(
                    adata,
                ),
                model_genes=np.array(self.gene_names),
                sample_genes=np.array(adata.var_names),
                gene_batch_size=1024,
            )
            adata2 = anndata.AnnData(
                X=X,
                obs=adata.obs.copy(),
            )
            adata2.var_names = self.gene_names
        else:
            logger.debug("Model and query gene names match.")
            adata2 = adata

        return adata2

    def _get_exp_grad(
        self,
        model: torch.nn.Module,
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
        # setup dataset, model, and training components
        # for querying, we also set a dataset with all of the data
        self.all_ds = dataprep.SingleCellDS(
            X=X,
            y=np.array(y),
        )
        self.all_dl = torch.utils.data.DataLoader(
            self.all_ds,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )

        return

    def query(
        self,
        adata: anndata.AnnData,
        target: str,
        source: str = "all",
        cell_type_col: str = "cell_ontology_class",
        batch_size: int = 512,
        n_batches: int = 100,
        n_cells: int = None,
    ) -> pd.DataFrame:
        """Find the features that distinguish `target` cells from `source` cells
        using expected gradient estimation.

        Parameters
        ----------
        adata : anndata.AnnData
            [Cells, Genes]
        target : str
            class name for target class.
            expected gradients highlight important features that define this cell type.
        source : str
            class name for source class to use as reference cells for expected
            gradient estimation.
            if `"all"` or `None`, uses all cells in `adata` as possible references.
        cell_type_col : str
            column in `adata.obs` containing cell type labels.
        n_batches : int
            number of reference batches to draw for each target sample.
        n_cells : int
            number of target samples to use for E[G] estimation.
            if `None`, uses all available samples.

        Returns
        -------
        saliency : pd.DataFrame
            [Genes, 1] mean expected gradient across cells used for
            estimation for each gene.
        """
        self.batch_size = batch_size
        adata = self._check_inputs(
            adata=adata,
            source=source,
            target=target,
            cell_type_col=cell_type_col,
        )
        self._setup_dataset(adata.X, adata.obs[cell_type_col], adata=adata)
        self.model.train(False)

        target_bidx = adata.obs[self.cell_type_col] == target
        if source in self.background_vals:
            source_bidx = np.ones(adata.shape[0], dtype=np.bool)
            # ensure target cells aren't in the source data
            source_bidx[target_bidx] = False
        else:
            source_bidx = adata.obs[self.cell_type_col] == source
        # regenerate labels in case the query dataset is different from the
        # training dataset
        class_names = self.class_names.tolist()
        target_y = np.array(
            [class_names.index(target)] * sum(target_bidx),
            dtype=np.int32,
        )
        source_y = adata.obs.loc[source_bidx, self.cell_type_col].tolist()
        source_y = np.array(
            [class_names.index(x) for x in source_y],
            dtype=np.int32,
        )

        source_adata = adata[source_bidx, :].copy()
        target_adata = adata[target_bidx, :].copy()
        logging.info(f"Subset adata to {target_adata.shape[0]} target cells.")

        if n_cells is not None:
            target_idx = np.random.choice(
                np.arange(target_adata.shape[0]),
                size=n_cells,
                replace=target_adata.shape[0] < n_cells,
            ).astype(np.int32)
        else:
            target_idx = np.arange(target_adata.shape[0])

        target_ds = dataprep.SingleCellDS(
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
        source_ds = dataprep.SingleCellDS(
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
        self.APExp = attrprior.AttributionPriorExplainer(
            source_ds,
            batch_size=batch_size,
            k=1,
            input_batch_index="input",
        )
        logger.debug("Set up Attribution Prior Explainer")
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

        # compute mean gradients across cells
        saliency = gradients.mean(0).sort_values(ascending=False)
        saliency.columns = ["exp_grad"]

        self.saliency = saliency
        self.gradients = gradients
        return saliency

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


class ClassificationEntropy(object):
    def __init__(self, reduce: str = "mean") -> None:
        """Compute the entropy of a classification probability vector"""
        self.reduce = reduce
        return

    def __call__(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Compute entropy for a probability tensor `x`

        Parameters
        ----------
        x : torch.FloatTensor
            [Cells, Classes] probability tensor

        Returns
        -------
        H : torch.FloatTensor
            either [Cells,] or [1,] if `reduce is not None`.
        """
        H = -1 * torch.sum(x * torch.log(x), dim=1)
        if self.reduce == "mean":
            H = torch.mean(H)
        return H


class Tesseract(IntegratedGradient):
    """Tessaract finds a path from a source vector in feature
    space to a destination vector.

    Attributes
    ----------
    model : torch.nn.Module
        trained scNym model.
    class_names : list or np.ndarray
        list of str names matching output nodes in `model`.
    gene_names : list or np.ndarray, optional
        gene names for the model.
    grad_activation : str
        activations where gradients should be collected.
        default "input" collects gradients at the level of input features.
    verbose : bool
        verbose outputs for stdout.
    energy_criterion : Callable
        criterion to evaluate the potential energy of a gene
        expression state given args `model` and `x` where
        `x` is a gene expression vector.
    optimizer : torch.optim.Optimizer
        optimizer for finding paths through gene expression
        space using a parametric gene expression vector.
    """

    def __init__(
        self,
        *,
        energy_criterion: typing.Callable,
        optimizer_class: typing.Callable,
        **kwargs,
    ) -> None:
        """Tessaract finds a path from a source vector in feature
        space to a destination vector that maximizes the likelihood
        of observing each intermediate position using a trained
        classification model.

        Parameters
        ----------
        energy_criterion : Callable
            criterion to evaluate the potential energy of a gene
            expression state given args `model` and `x` where
            `x` is a gene expression vector.
        optimizer_class : Callable
            function to initialize a `torch.optim.Optimizer`.

        Returns
        -------
        None
        """
        super(Tesseract, self).__init__(**kwargs)
        self.energy_criterion = energy_criterion
        self.optimizer_class = optimizer_class
        return

    def find_path(
        self,
        adata: anndata.AnnData,
        groupby: str,
        source_class: str,
        target_class: str,
        energy_weight: float = 1.0,
        n_epochs: int = 500,
        tol: float = 1.0,
        patience: int = 10,
    ) -> torch.FloatTensor:
        """Find a path between a source and target cell class
        given an AnnData experiment containing both.

        Parameters
        ----------
        adata : anndata.AnnData
            [Cells, Features] experiment.
        groupby : str
            column in `adata.obs` containing class names.
        source_class : str
            class in `self.class_names` and `adata.obs[groupby]`
            for initialization.
        target_class : str
            class in `self.class_names` and `adata.obs[groupby]`
            for optimization.
        energy_weight : float, optional
            weight for the energy criterion relative to the class
            scores.
        n_epochs : int, optional
            number of epochs for optimization.
        tol : float, optional
            minimum L2 difference across epochs to consider
            the optimization to be progressing.
        patience : int, optional
            number of epochs to wait before early stopping.

        Returns
        -------
        path : torch.FloatTensor
            [epochs, Features] path through gene expression space.
        Sets `self.last_path` with the value of path.
        """

        source_cell_idx = adata.obs[groupby] == source_class
        source_mean = torch.from_numpy(
            np.array(adata[source_cell_idx, :].X.mean(0))
        ).float()
        model_device = list(self.model.parameters())[0].device
        source_mean = source_mean.to(device=model_device)

        if self.grad_activation == "first_layer":
            # we're using first layer embeddings as the relevant
            # space for integrated gradient computation and
            # optimization
            source_mean2use = self.input2first(source_mean)
            self.scoring_model = self.first2output
        else:
            source_mean2use = source_mean
            self.scoring_model = self.model

        # initialize the gene expression vector at the source
        x = copy.deepcopy(source_mean2use)
        self.optimizer = self.optimizer_class({"params": x, "name": "expression_path"})

        # perform optimization to the target class while
        # minimizing an energy criterion
        def loss(
            x,
        ):
            target_idx = self.class_names.index(target_class)
            source_idx = self.class_names.index(source_class)
            outputs = self.scoring_model(
                x,
            )
            probs = torch.nn.functional.softmax(outputs, dim=1)
            energy = (
                self.energy_criterion(
                    x,
                )
                * energy_weight
            )
            l = (probs[source_idx] - probs[target_idx]) + energy
            return l

        # intialize path collector and set the `waiting_epochs`
        # for early stopping to an initial zero value
        path_points = []
        waiting_epochs = 0
        logger.info("Beginning pathfinding optimization")
        for epoch in range(n_epochs):
            # save path locations
            path_points.append(copy.deepcopy(x.detach().cpu()))

            # compute loss and perform an update step
            l = loss(
                x,
            )
            self.optimizer.zero_grad()
            l.backward()
            self.optimizer.step()

            # check if x is changing substantially
            delta = x.data - path_points[-1]
            l2 = torch.norm(delta, p=2)
            if l2 < tol and waiting_epochs > patience:
                msg = f"\tchange in l2 < {tol} for {patience} epochs\n"
                msg += "\tending optimizing."
                logger.warning(msg)
            elif l2 < tol:
                waiting_epochs += 1
            else:
                waiting_epochs = 0

        path = torch.cat(path_points, dim=0)
        self.last_path = path
        return path
