import torch
import torch.nn as nn


class ResBlock(nn.Module):
    '''Residual block.

    References
    ----------
    Deep Residual Learning for Image Recognition
    Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    arXiv:1512.03385
    '''

    def __init__(
        self,
        n_inputs: int,
        n_hidden: int,
    ) -> None:
        '''Residual block with fully-connected neural network
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
        '''
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

    def forward(self, x: torch.FloatTensor,
                ) -> torch.FloatTensor:
        '''Residual block forward pass.

        Parameters
        ----------
        x : torch.FloatTensor
            [Batch, self.n_inputs]

        Returns
        -------
        o : torch.FloatTensor
            [Batch, self.n_hidden]
        '''
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
    '''Cell type classifier from expression data.

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
    '''

    def __init__(
        self,
        n_genes: int,
        n_cell_types: int,
        n_hidden: int = 256,
        n_layers: int = 2,
        init_dropout: float = 0.0,
        residual: bool = False,
        batch_norm: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        '''
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

        Returns
        -------
        None.
        '''
        super(CellTypeCLF, self).__init__()

        self.n_genes = n_genes
        self.n_cell_types = n_cell_types
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.init_dropout = init_dropout
        self.residual = residual
        self.batch_norm = batch_norm
        self.track_running_stats = track_running_stats
            
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

        hidden_layers = hidden_layer*self.n_layers

        # Build the classifier `nn.Module`.
        self.classif = nn.Sequential(
            nn.Linear(self.n_genes, self.n_hidden),
            nn.BatchNorm1d(
                num_features=self.n_hidden,
                track_running_stats=self.track_running_stats,                
            ),
            nn.Dropout(),
            nn.ReLU(inplace=True),
            *hidden_layers,
            nn.Linear(self.n_hidden, self.n_cell_types)
        )
        return

    def forward(
        self,
        x: torch.FloatTensor,
    ) -> torch.FloatTensor:
        '''Perform a forward pass through the model

        Parameters
        ----------
        x : torch.FloatTensor
            [Batch, self.n_genes]

        Returns
        -------
        pred : torch.FloatTensor
            [Batch, self.n_cell_types]
        '''
        # add initial dropout noise
        if self.init_dropout.p > 0:
            # expm1 to normed counts
            x = torch.expm1(x)
            x = self.init_dropout(x)
            # renorm to log1p CPM
            size = torch.sum(x, dim=1).reshape(-1, 1)
            prop_input_ = x / size
            norm_input_ = prop_input_ * 1e6
            x = torch.log1p(norm_input_)
        pred = self.classif(x)
        return pred
    
    
class GradReverse(torch.autograd.Function):
    '''Layer that reverses and scales gradients before
    passing them up to earlier ops in the computation graph
    during backpropogation.
    '''
    @staticmethod
    def forward(ctx, x, weight):
        '''
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
        '''
        # store the weight we'll use in backward in the context
        ctx.weight = weight
        return x.view_as(x) * 1
    
    @staticmethod
    def backward(ctx, grad_output):
        '''Return gradients
        
        Returns
        -------
        rev_grad : torch.FloatTensor
            reversed gradients scaled by `weight` passed in `.forward()`
        None : None
            a dummy "gradient" required since we passed a weight float
            in `.forward()`.
        '''
        # here scale the gradient and multiply by -1
        # to reverse the gradients
        return (grad_output * -1 * ctx.weight), None

    
class DANN(nn.Module):
    '''Build a domain adaptation neural network'''
    
    def __init__(
        self,
        model: CellTypeCLF,
        n_domains: int=2,
        weight: float=1.,
        n_layers: int=1,
    ) -> None:
        '''Build a domain adaptation neural network using
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
        '''
        super(DANN, self).__init__()
        
        self.model = model
        self.n_domains = n_domains
        
        self.embed = nn.Sequential(
            *list(list(model.modules())[0].children())[1][:-1]
        )
        
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
        '''Set the weight term used after reversing gradients'''
        self.weight = weight
        return
    
    def forward(
        self,
        x: torch.FloatTensor,
    ) -> (torch.FloatTensor, torch.FloatTensor):
        '''Perform a forward pass.
        
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
        '''
        # get the model embedding
        x_embed = self.embed(x)
        # reverse gradients and scale by a weight
        x_rev   = GradReverse.apply(
            x_embed,
            self.weight,
        )
        # classify the domains
        domain_pred = self.domain_clf(x_rev)
        return domain_pred, x_embed
        


class CellTypeCLFConditional(CellTypeCLF):
    '''Conditional vartiaton of the `CellTypeCLF`

    Attributes
    ----------
    n_genes : int
        number of the input features corresponding to genes.
    n_tissues : int
        length of the one-hot `upper_group` vector appended
        to inputs.
    '''

    def __init__(
        self,
        n_genes: int,
        n_tissues: int,
        **kwargs,
    ) -> None:
        '''Conditional vartiaton of the `CellTypeCLF`.

        Parameters
        ----------
        n_genes : int
            number of genes in the input
        n_tissues : int
            number of tissues encoded in the conditional vector.

        Returns
        -------
        None.

        Notes
        -----
        Assumes that inputs are `n_genes + n_tissues`, concatenated
        [Genes :: Tissue-One-Hot].

        Passes `**kwargs` to `CellTypeCLF`.
        '''
        # Build a CellTypeCLF with `n_genes` + `n_tissues` input nodes to
        # take both the gene vector and one-hot upper_group label as input
        super(CellTypeCLFConditional, self).__init__(
            n_genes=(n_genes + n_tissues),
            **kwargs)
        self.n_tissues = n_tissues
        return

    def forward(
        self, 
        x: torch.FloatTensor,
    ) -> torch.FloatTensor:
        '''Perform a forward pass through the model

        Parameters
        ----------
        x : torch.FloatTensor
            [Batch, self.n_genes + self.n_tissues]

        Returns
        -------
        pred : torch.FloatTensor
            [Batch, self.n_cell_types]
        '''
        # don't pass one_hot labels through the initial dropout
        one_hot = x[:, -self.n_tissues:]
        genes = x[:, :-self.n_tissues]

        genes = self.init_dropout(genes)
        x_drop = torch.cat([genes, one_hot], dim=1)

        # classify on the full genes + one-hot input
        pred = self.classif(x_drop)
        return pred
