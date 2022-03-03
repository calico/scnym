import torch
import torch.nn as nn
from typing import Callable, Iterable, Union, Tuple
import logging

logger = logging.getLogger(__name__)


class ResBlock(nn.Module):
    """Residual block.

    References
    ----------
    Deep Residual Learning for Image Recognition
    Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    arXiv:1512.03385
    """

    def __init__(
        self,
        n_inputs: int,
        n_hidden: int,
    ) -> None:
        """Residual block with fully-connected neural network
        layers.

        Parameters
        ----------
        n_inputs : int
            number of input dimensions.
        n_hidden : int
            number of hidden dimensions in the Residual Block.

        Returns
        -------
        None.
        """
        super(ResBlock, self).__init__()

        self.n_inputs = n_inputs
        self.n_hidden = n_hidden

        # Build the initial projection layer
        self.linear00 = nn.Linear(self.n_inputs, self.n_hidden)
        self.norm00 = nn.BatchNorm1d(num_features=self.n_hidden)
        self.relu00 = nn.ReLU(inplace=True)

        # Map from the latent space to output space
        self.linear01 = nn.Linear(self.n_hidden, self.n_hidden)
        self.norm01 = nn.BatchNorm1d(num_features=self.n_hidden)
        self.relu01 = nn.ReLU(inplace=True)
        return

    def forward(
        self,
        x: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Residual block forward pass.

        Parameters
        ----------
        x : torch.FloatTensor
            [Batch, self.n_inputs]

        Returns
        -------
        o : torch.FloatTensor
            [Batch, self.n_hidden]
        """
        identity = x

        # Project input to the latent space
        o = self.norm00(self.linear00(x))
        o = self.relu00(o)

        # Project from the latent space to output space
        o = self.norm01(self.linear01(o))

        # Make this a residual connection
        # by additive identity operation
        o += identity
        return self.relu01(o)


class CellTypeCLF(nn.Module):
    """Cell type classifier from expression data.

    Attributes
    ----------
    n_genes : int
        number of input genes in the model.
    n_cell_types : int
        number of output classes in the model.
    n_hidden : int
        number of hidden units in the model.
    n_layers : int
        number of hidden layers in the model.
    init_dropout : float
        dropout proportion prior to the first layer.
    residual : bool
        use residual connections.
    """

    def __init__(
        self,
        n_genes: int,
        n_cell_types: int,
        n_hidden: int = 256,
        n_hidden_init: int = 256,
        hidden_init_dropout: bool=True,
        hidden_init_activ: str="relu",
        n_layers: int = 2,
        init_dropout: float = 0.0,
        residual: bool = False,
        batch_norm: bool = True,
        track_running_stats: bool = True,
        n_decoder_layers: int = 0,
        use_raw_counts: bool = False,
    ) -> None:
        """
        Cell type classifier from expression data.
        Linear layers with batch norm and dropout.

        Parameters
        ----------
        n_genes : int
            number of genes in the input
        n_cell_types : int
            number of cell types for the output
        n_hidden : int
            number of hidden unit
        n_hidden_init :
            number of hidden units for the initial encoding layer.
        hidden_init_dropout : bool
            perform dropout on the first set of activations.
        hidden_init_activ : str
            activation for the initial embedding. one of {relu, softmax, sigmoid}.            
        n_layers : int
            number of hidden layers.
        init_dropout : float
            dropout proportion prior to the first layer.
        residual : bool
            use residual connections.
        batch_norm : bool
            use batch normalization in hidden layers.
        track_running_stats : bool
            track running statistics in batch norm layers.
        n_decoder_layers : int
            number of layers in the decoder.
        use_raw_counts : bool
            provide raw counts as input.

        Returns
        -------
        None.
        """
        super(CellTypeCLF, self).__init__()

        self.n_genes = n_genes
        self.n_cell_types = n_cell_types
        self.n_hidden = n_hidden
        self.n_hidden_init = n_hidden_init
        self.hidden_init_dropout = hidden_init_dropout
        self.n_decoder_layers = n_decoder_layers
        self.n_layers = n_layers
        self.init_dropout = init_dropout
        self.residual = residual
        self.batch_norm = batch_norm
        self.track_running_stats = track_running_stats
        self.use_raw_counts = use_raw_counts

        # simulate technical dropout of scRNAseq
        self.init_dropout = nn.Dropout(p=self.init_dropout)

        # Define a vanilla NN layer with batch norm, dropout, ReLU
        vanilla_layer = [
            nn.Linear(self.n_hidden, self.n_hidden),
        ]
        if self.batch_norm:
            vanilla_layer += [
                nn.BatchNorm1d(
                    num_features=self.n_hidden,
                    track_running_stats=self.track_running_stats,
                ),
            ]
        vanilla_layer += [
            nn.Dropout(),
            nn.ReLU(inplace=True),
        ]

        # Define a residual NN layer with batch norm, dropout, ReLU
        residual_layer = [
            ResBlock(self.n_hidden, self.n_hidden),
        ]
        if self.batch_norm:
            residual_layer += [
                nn.BatchNorm1d(
                    num_features=self.n_hidden,
                    track_running_stats=self.track_running_stats,
                ),
            ]

        residual_layer += [
            nn.Dropout(),
            nn.ReLU(inplace=True),
        ]

        # Build the intermediary layers of the model
        if self.residual:
            hidden_layer = residual_layer
        else:
            hidden_layer = vanilla_layer

        self.hidden_layers = hidden_layer * (self.n_layers - 1)

        # build a stack of the very first embedding layer, to be called separately
        # later if needed
        # because this is just a list, the parameters won't be registed by this setup
        # alone
        self.input_stack = [
            nn.Linear(self.n_genes, self.n_hidden_init),
            nn.BatchNorm1d(
                num_features=self.n_hidden_init,
                track_running_stats=self.track_running_stats,
            ),
        ]
        if self.hidden_init_dropout:
            self.input_stack.append(nn.Dropout())

        if hidden_init_activ.lower() == "relu":
            init_activ_mod = nn.ReLU(inplace=True)
        elif hidden_init_activ.lower() == "softmax":
            init_activ_mod = nn.Softmax(dim=1)
        elif hidden_init_activ.lower() == "sigmoid":
            init_activ_mod = nn.Sigmoid()
        else:
            msg = f"{hidden_init_activ} must fit one of the specified initial activations."
            raise ValueError(msg)
        self.input_stack.append(init_activ_mod,)

        self.mid_stack = [
            nn.Linear(self.n_hidden_init, self.n_hidden),
            nn.BatchNorm1d(
                num_features=self.n_hidden,
                track_running_stats=self.track_running_stats,
            ),
            nn.Dropout(),
            nn.ReLU(inplace=True),
        ]

        # Build the embed and classifier `nn.Module`.
        self.embed = nn.Sequential(
            *self.input_stack,
            *self.mid_stack,
            *self.hidden_layers,
        )

        dec_hidden = hidden_layer * (self.n_decoder_layers - 1)
        final_clf = nn.Linear(self.n_hidden, self.n_cell_types)
        self.classif = nn.Sequential(
            *dec_hidden,
            final_clf,
        )
        return

    def get_initial_embedder(self,):
        """Convenience function to return the initial embedding layers as `nn.Module`"""
        return nn.Sequential(*self.input_stack)

    def forward(
        self,
        x: torch.FloatTensor,
        return_embed: bool = False,
    ) -> torch.FloatTensor:
        """Perform a forward pass through the model

        Parameters
        ----------
        x : torch.FloatTensor
            [Batch, self.n_genes]
        return_embed : bool
            return the embedding and the class predictions.

        Returns
        -------
        pred : torch.FloatTensor
            [Batch, self.n_cell_types]
        embed : torch.FloatTensor, optional
            [Batch, n_hidden], only returned if `return_embed`.
        """
        # add initial dropout noise
        if self.init_dropout.p > 0 and not self.use_raw_counts:
            # counts are log1p(CPM)
            # expm1 to normed counts
            x = torch.expm1(x)
            x = self.init_dropout(x)
            # renorm to log1p CPM
            size = torch.sum(x, dim=1).reshape(-1, 1)
            prop_input_ = x / size
            norm_input_ = prop_input_ * 1e6
            x = torch.log1p(norm_input_)
        elif self.init_dropout.p > 0 and self.use_raw_counts:
            x = self.init_dropout(x)
        else:
            # we don't need to do initial dropout
            pass
        x_embed = self.embed(x)
        pred = self.classif(x_embed)

        if return_embed:
            r = (
                pred,
                x_embed,
            )
        else:
            r = pred
        return r


class GradReverse(torch.autograd.Function):
    """Layer that reverses and scales gradients before
    passing them up to earlier ops in the computation graph
    during backpropogation.
    """

    @staticmethod
    def forward(ctx, x, weight):
        """
        Perform a no-op forward pass that stores a weight for later
        gradient scaling during backprop.

        Parameters
        ----------
        x : torch.FloatTensor
            [Batch, Features]
        weight : float
            weight for scaling gradients during backpropogation.
            stored in the "context" ctx variable.

        Notes
        -----
        We subclass `Function` and use only @staticmethod as specified
        in the newstyle pytorch autograd functions.
        https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function

        We define a "context" ctx of the class that will hold any values
        passed during forward for use in the backward pass.

        `x.view_as(x)` and `*1` are necessary so that `GradReverse`
        is actually called
        `torch.autograd` tries to optimize backprop and
        excludes no-ops, so we have to trick it :)
        """
        # store the weight we'll use in backward in the context
        ctx.weight = weight
        return x.view_as(x) * 1.0

    @staticmethod
    def backward(ctx, grad_output):
        """Return gradients

        Returns
        -------
        rev_grad : torch.FloatTensor
            reversed gradients scaled by `weight` passed in `.forward()`
        None : None
            a dummy "gradient" required since we passed a weight float
            in `.forward()`.
        """
        # here scale the gradient and multiply by -1
        # to reverse the gradients
        return (grad_output * -1 * ctx.weight), None


class DANN(nn.Module):
    """Build a domain adaptation neural network"""

    def __init__(
        self,
        model: CellTypeCLF,
        n_domains: int = 2,
        weight: float = 1.0,
        n_layers: int = 1,
    ) -> None:
        """Build a domain adaptation neural network using
        the embedding of a provided model.

        Parameters
        ----------
        model : CellTypeCLF
            cell type classification model.
        n_domains : int
            number of domains to adapt.
        weight : float
            weight for reversed gradients.
        n_layers : int
            number of hidden layers in the network.

        Returns
        -------
        None.
        """
        super(DANN, self).__init__()

        self.model = model
        self.n_domains = n_domains

        self.embed = model.embed

        hidden_layers = [
            nn.Linear(self.model.n_hidden, self.model.n_hidden),
            nn.ReLU(),
        ] * n_layers

        self.domain_clf = nn.Sequential(
            *hidden_layers,
            nn.Linear(self.model.n_hidden, self.n_domains),
        )
        return

    def set_rev_grad_weight(
        self,
        weight: float,
    ) -> None:
        """Set the weight term used after reversing gradients"""
        self.weight = weight
        return

    def forward(
        self,
        x: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Perform a forward pass.

        Parameters
        ----------
        x : torch.FloatTensor
            [Batch, Features] input.

        Returns
        -------
        domain_pred : torch.FloatTensor
            [Batch, n_domains] logits.
        x_embed : torch.FloatTensor
            [Batch, n_hidden]
        """
        # get the model embedding
        x_embed = self.embed(x)
        # reverse gradients and scale by a weight
        # domain_pred -> x_rev -> GradReverse -> x_embed
        #      d+     ->  d+   ->     d-      ->   d-
        x_rev = GradReverse.apply(
            x_embed,
            self.weight,
        )
        # classify the domains
        domain_pred = self.domain_clf(x_rev)
        return domain_pred, x_embed


class AE(nn.Module):
    """Build an autoencoder that shares the classifier embedding.

    Attributes
    ----------
    model : CellTypeCLF
        cell type classification model.
    n_layers : int
        number of hidden layers in the network.
    n_hidden : int
        number of hidden units in each hidden layer.
        defaults to the hidden layer size of the model.
    dispersion : torch.nn.Parameter
        [model.n_genes,] dispersion parameters for each gene.
        `None` unless `model.use_raw_counts`.
    latent_libsize : bool
        use a latent variable to store library size. if `False`,
        uses the observed library size to scale abundance profiles.
    """

    noise_scale = 1.0

    def __init__(
        self,
        model: CellTypeCLF,
        n_layers: int = 2,
        n_hidden: int = None,
        n_domains: int = None,
        latent_libsize: bool = False,
    ) -> None:
        """Build an autoencoder using the embedding of a provided model.

        Parameters
        ----------
        model : CellTypeCLF
            cell type classification model.
        n_layers : int
            number of hidden layers in the network.
        n_hidden : int
            number of hidden units in each hidden layer.
            defaults to the hidden layer size of the model.
        n_domains : int
            number of domain covariates to include.
        latent_libsize : bool
            use a latent variable to store library size. if `False`,
            uses the observed library size to scale abundance profiles.

        Returns
        -------
        None.

        Notes
        -----
        Maps gene expression vectors to an embedding using the same
        trunk as the classification model. If `model.use_raw_counts`,
        reconstructs library depth using the latent library size and
        also learns a set of dispersion parameters for each gene.
        Reconstructs profiles using a decoder model that mirrors the
        classification embedding trunk.
        """
        super(AE, self).__init__()

        self.model = model
        self.n_hidden = self.model.n_hidden if n_hidden is None else n_hidden
        self.latent_libsize = latent_libsize
        self.n_domains = n_domains if n_domains is not None else 0

        # extract the embedder from the classification model
        self.embed = self.model.embed

        # append decoder layers
        dec_input = [
            nn.Linear(self.model.n_hidden + self.n_domains, self.n_hidden),
            nn.ReLU(),
        ]

        hidden_layers = [
            nn.Linear(self.model.n_hidden, self.n_hidden),
            nn.ReLU(),
        ] * (n_layers - 1)

        self.decoder = nn.Sequential(
            *dec_input,
            *hidden_layers,
            nn.Linear(self.n_hidden, self.model.n_genes),
        )

        if self.model.use_raw_counts:
            # initialize dispersion parameters from a unit Gaussian
            self.dispersion = nn.Parameter(torch.randn(self.model.n_genes))
        else:
            self.dispersion = torch.ones((1,))

        # encode log(library_size) as a latent variable
        self.libenc = nn.Sequential(
            nn.Linear(self.model.n_genes + self.n_domains, 1),
            nn.ReLU(),
        )

        return

    def noise(
        self,
        x_embed: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Add white noise to the latent embedding"""
        eps = torch.randn_like(x_embed) * self.noise_scale
        return torch.nn.functional.relu(x_embed + eps)

    def forward(
        self,
        x: torch.FloatTensor,
        x_embed: torch.FloatTensor = None,
        x_domain: torch.FloatTensor = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Perform a forward pass.

        Parameters
        ----------
        x : torch.FloatTensor
            [Batch, Features] input.
        x_embed : torch.FloatTensor, optional.
            [Batch, n_hidden] embedding.
        x_domain : torch.FloatTensor, optional.
            [Batch, Domains] one-hot labels.
            used for conditional decoding.

        Returns
        -------
        reconstructed_profiles : torch.FloatTensor
            [Batch, Features] abundance profiles [0, 1].
        scaled_profiles : torch.FloatTensor
            [Batch, Features] profiles scaled by latent depths.
        dispersion : torch.FloatTensor
            [Features,] dispersion parameters for each gene.
        x_embed : torch.FloatTensor
            [Batch, n_hidden]
        """
        # get the model embedding, avoid recomputing if a precomputed
        # embedding is passed in
        x_embed = self.embed(x) if x_embed is None else x_embed

        if self.training:
            x_embed = self.noise(x_embed)

        # check the dimensions are sane
        if x_embed.size(-1) > 2048:
            logger.warn(
                f"AE `x_embed` dimension is larger than expected: {x_embed.size(1)}"
            )

        # add domain covariates if provided and initialized to use covars
        if x_domain is None and self.n_domains > 0:
            msg = "Must provide domain covariates for a conditional model. Received `None`."
            raise TypeError(msg)
        if x_domain is not None and self.n_domains > 0:
            logger.debug(f"Domain covariates added. Size {x_domain.size()}.")
            x_embed = torch.cat([x_embed, x_domain], dim=-1)
            x2libsz = torch.cat([x, x_domain], dim=-1)
        else:
            x2libsz = x

        # reconstruct gene expression abundance profiles, first with raw
        # activations
        x_rec = self.decoder(x_embed)
        # use softmax to go from logits to relative abundance profiles
        x_rec = nn.functional.softmax(x_rec, dim=1)

        if self.latent_libsize:
            # `libenc` returns the log of the library size
            lib_size = self.libenc(x2libsz)
            lib_size = torch.clamp(lib_size, max=12)  # numerical stability
        else:
            lib_size = torch.log(x.sum(1)).view(-1, 1)  # [Cells, 1]
        x_scaled = x_rec * torch.exp(lib_size)

        return x_rec, x_scaled, torch.exp(self.dispersion), x_embed
