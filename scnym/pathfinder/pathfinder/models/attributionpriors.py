#!/usr/bin/env python
# adopted from https://github.com/suinleelab/attributionpriors
import functools
import operator
from typing import Callable, Union
import numpy as np
import torch
from torch.autograd import grad
from torch.utils.data import DataLoader
import logging
from multiprocessing import Pool

logger = logging.getLogger(__name__)

DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def gather_nd(params, indices):
    """
    Args:
        params: Tensor to index
        indices: k-dimension tensor of integers. 
    Returns:
        output: 1-dimensional tensor of elements of ``params``, where
            output[i] = params[i][indices[i]]

            params   indices   output

            1 2       1 1       4
            3 4       2 0 ----> 5
            5 6       0 0       1
    """
    max_value = functools.reduce(operator.mul, list(params.size())) - 1
    indices = indices.t().long()
    ndim = indices.size(0)
    idx = torch.zeros_like(indices[0]).long()
    m = 1

    for i in range(ndim)[::-1]:
        idx += indices[i]*m
        m *= params.size(i)

    idx[idx < 0] = 0
    idx[idx > max_value] = 0
    return torch.take(params, idx)


def adj2lap(adj: torch.FloatTensor,) -> torch.FloatTensor:
    """Convert an adjacency matrix to a graph Laplacian

    Notes
    -----
    Graph Laplacian is 

    .. math::

        L = D - A

    where :math:`D` is a diagonal matrix with the degree of
    each node and :math:`A` is the graph adjacency matrix.
    """
    adj = (adj > 0).float()
    row_sum = torch.sum(adj, dim=1)
    # constructs [n_vertices, n_vertices] with row_sum on diagonal
    D = torch.diag(row_sum)
    return D - adj

    
def tgini(x):
    mad = torch.mean(torch.abs(x.reshape(-1,1)-x.reshape(1,-1)))
    rmad = mad/torch.mean(x)
    g = 0.5*rmad
    return g


def gini_eg(
    shaps: torch.FloatTensor
) -> torch.FloatTensor:
    """Gini coefficient sparsity prior
    
    Parameters
    ----------
    shaps : torch.FloatTensor
        [Observations, Features] estimated Shapley values.
        
    Returns
    -------
    gini_prior : torch.FloatTensor
        inverse Gini coefficient prior penalty.
    """
    abs_attrib = shaps.abs()
    return -tgini(abs_attrib.mean(0))


def gini_classwise_eg(
    shaps: torch.FloatTensor,
    target: torch.LongTensor,
) -> torch.FloatTensor:
    """Compute Gini coefficient sparsity prior within individual
    classes. This allows each class to have a unique set of sparsely
    activated features, rather than globally requiring all classes
    to use the same small feature set.
    
    Parameters
    ----------
    shaps : torch.FloatTensor
        [Observations, Features] estimated Shapley values.
    target : torch.LongTenspr
        [Observations,] int class labels.
        
    Returns
    -------
    gini_prior : torch.FloatTensor
        inverse Gini coefficient prior penalty.    
    """
    classes = torch.unique(target)
    ginis = torch.zeros((len(classes),)).to(device=shaps.device)
    n_obs = torch.zeros((len(classes),)).to(device=shaps.device)
    for i, c in enumerate(classes):
        c_shaps = shaps[target==c]
        c_gini = gini_eg(c_shaps)
        ginis[i] = c_gini
        n_obs[i] = c_shaps.size(0)
    # compute weighted gini coefficient
    p_obs = n_obs/torch.sum(n_obs)
    weighted_gini = torch.sum(p_obs * ginis)
    return weighted_gini

def graph_eg(
    shaps: torch.FloatTensor, 
    graph: torch.FloatTensor,
) -> torch.FloatTensor:
    """Graph attribution prior
    
    Parameters
    ----------
    shaps : torch.FloatTensor
        [Observations, Features] estimated Shapley values.
    graph : torch.FloatTensor
        [Features, Features] adjacency matrix (weighted or binary).
        
    Returns
    -------
    graph_prior : torch.FloatTensor
        graph prior penalty.
    """
    # get mean gradient for each feature
    feature_grad = torch.mean(shaps, dim=0)
    # get a matrix of differences between feature grads
    cols = feature_grad.view(1, -1).repeat(feature_grad.size(0), 1)
    rows = feature_grad.view(-1, 1).repeat(1, feature_grad.size(0))
    # delta[i, j] is grad_i - grad_j
    delta = rows - cols
    # "Gaussian" penalty is just square of delta
    penalty = torch.pow(delta, 2)
    weighted_penalty = penalty * graph
    return weighted_penalty


def check(key, sets: dict, reference: set) -> list:
    return [x in reference for x in sets[key]]


class AttributionPriorExplainer(object):
    def __init__(
        self, 
        background_dataset: torch.utils.data.Dataset, 
        batch_size: int, 
        random_alpha: bool=True,
        k: int=1,
        scale_by_inputs: bool=True,
        abs_scale: bool=True,
        input_batch_index: Union[str,int,tuple]=None,
    ) -> None:
        """Estimates feature gradients using expected gradients.
        
        Parameters
        ----------
        background_dataset : torch.utils.data.Dataset
            dataset of samples to use as background references.
            most commonly, this is the whole training set.
        batch_size : int
            batch size used for training. must be the same as the 
            batch size for the training dataloader.
        random_alpha : bool
            use randomized `alpha ~ Unif(0, 1)` values for computing
            an intermediary sample between the reference and target
            sample at each minibatch.
        k : int
            number of references to use per training example per minibatch.
            `k=1` works well as a default with minimal computational 
            overhead.
        scale_by_inputs : bool
            scale expected gradient values using a dot-product with the
            difference `(input-reference)` feature values.
        abs_scale : bool
            only considered if `scale_by_inputs=True`. Rather than scaling
            by the raw difference, scale by the absolute value of the 
            difference.
        input_batch_index : Union[str,int,tuple], optional
            key for extracting the input values from a batch drawn from
            `background_dataset`. e.g. if batches are stored in `dict`,
            this is the key for the input tensor. if batches are `tuple`,
            this is the index of the input tensor.
        
        Returns
        -------
        None.
        
        References
        ----------
        https://github.com/suinleelab/attributionpriors
        """
        self.random_alpha = random_alpha
        self.k = k
        self.scale_by_inputs = scale_by_inputs
        self.abs_scale = abs_scale
        self.batch_size = batch_size
        self.ref_set = background_dataset
        self.ref_sampler = DataLoader(
            dataset=background_dataset, 
            batch_size=batch_size*k, 
            shuffle=True, 
            drop_last=True,
        )
        self.input_batch_index = input_batch_index
        return

    def _get_ref_batch(self, k=None,):
        """Get a batch from the reference dataset"""
        b = next(iter(self.ref_sampler))
        if self.input_batch_index is not None:
            # extract the input tensor using a provided index
            b = b[self.input_batch_index].float()
        b = b.to(device=self.DEFAULT_DEVICE)
        if self.batch_transformation is not None:
            # transform the reference batch with a specified transformation
            b = self.batch_transformation(b)
        return b
    
    def _get_samples_input(
        self, 
        input_tensor: torch.FloatTensor, 
        reference_tensor: torch.FloatTensor,
    ) -> torch.FloatTensor:
        '''
        Calculate interpolation points between input samples and reference
        samples.
        
        Parameters
        ----------
        input_tensor : torch.FloatTensor 
            shape (batch, ...), where ... indicates the input dimensions. 
        reference_tensor : torch.FloatTensor
            shape (batch, k, ...) where k represents the number of 
            background reference samples to draw per input in the batch.
            
        Returns
        -------
        samples_input : torch.FloatTensor
            shape (batch, k, ...) with the interpolated points between 
            input and ref.
            
        Notes
        -----
        For integrated gradients, we compute some `M=100+` samples interpolating
        between each input and a relevant reference sample. For expected
        gradients, we rather compute interpolation points that lie randomly
        along the linear path between the sample and reference in each minibatch.
        '''
        input_dims = list(input_tensor.size())[1:]
        num_input_dims = len(input_dims)
            
        batch_size = reference_tensor.size()[0]
        k_ = reference_tensor.size()[1]

        # Grab a [batch_size, k]-sized interpolation sample
        if self.random_alpha:
            t_tensor = torch.FloatTensor(batch_size, k_).uniform_(0,1).to(self.DEFAULT_DEVICE)
        else:
            if k_==1:
                t_tensor = torch.cat([torch.Tensor([1.0]) for i in range(batch_size)]).to(self.DEFAULT_DEVICE)
            else:
                t_tensor = torch.cat([torch.linspace(0,1,k_) for i in range(batch_size)]).to(self.DEFAULT_DEVICE)

        shape = [batch_size, k_] + [1] * num_input_dims
        interp_coef = t_tensor.view(*shape)

        # Evaluate the end points
        end_point_ref = (1.0 - interp_coef) * reference_tensor

        input_expand_mult = input_tensor.unsqueeze(1)
        end_point_input = interp_coef * input_expand_mult
        
        # A fine Affine Combine
        samples_input = end_point_input + end_point_ref
        return samples_input
    
    def _get_samples_delta(
        self, 
        input_tensor: torch.FloatTensor, 
        reference_tensor: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Compute the distance in feature space between input samples
        and reference samples.
        
        Parameters
        ----------
        input_tensor : torch.FloatTensor 
            shape (batch, ...), where ... indicates the input dimensions. 
        reference_tensor : torch.FloatTensor
            shape (batch, k, ...) where k represents the number of 
            background reference samples to draw per input in the batch.        
        
        Returns
        -------
        sd : torch.FloatTensor
            (batch, k, ...) differences in each feature between input
            samples and the assigned reference.
        """
        input_expand_mult = input_tensor.unsqueeze(1)
        sd = input_expand_mult - reference_tensor
        if self.abs_scale:
            sd = torch.abs(sd)
        return sd
    
    def _get_grads(
        self, 
        samples_input: torch.FloatTensor, 
        model: torch.nn.Module,
        sparse_labels: torch.LongTensor=None,
    ) -> torch.FloatTensor:
        """Compute gradients for a given model and input tensor, 
        taking into account sparse labels if provided.
        
        Parameters
        ----------
        samples_input : torch.FloatTensor
            (batch, k, ...) input features.
            during training, these are interpolated samples between input
            and reference.
            during evaluation, these are raw input samples.
        model : torch.nn.Module
            model for evaluation.
        sparse_labels : torch.LongTensor, optional
            (batch, classes) one-hot labels for class assignments.
            must be provided if `classes > 1`.
            
        Returns
        -------
        grad_tensor : torch.FloatTensor
            (batch, ...) gradient values
        """
        samples_input.requires_grad = True

        grad_tensor = torch.zeros(samples_input.shape).float().to(self.DEFAULT_DEVICE)
        logger.debug(f"grad_tensor size: {grad_tensor.size()}")
        
        for i in range(self.k):
            particular_slice = samples_input[:,i]
            logger.debug(f"particular_slice size: {particular_slice.size()}")
            batch_output = model(particular_slice)
            logger.debug(f"batch_output size: {batch_output.size()}")

            # should check that users pass in sparse labels
            # Only look at the user-specified label
            # if there is only one class, `batch_output` is already `(batch, 1)`
            if batch_output.size(1) > 1:
                if sparse_labels is None:
                    msg = "`sparse_labels` must be provided if more than one\n"
                    msg += "output class is present."
                    raise TypeError(msg)

                sample_indices = torch.arange(0, batch_output.size(0)).to(self.DEFAULT_DEVICE)
                indices_tensor = torch.cat(
                    [
                        sample_indices.unsqueeze(1), 
                        sparse_labels.unsqueeze(1),
                    ], 
                    dim=1,
                )
                # gathers the relevant class output for each sample to create
                # batch_output shape : (batch, 1).
                batch_output = gather_nd(batch_output, indices_tensor)

            model_grads = grad(
                    outputs=batch_output,
                    inputs=particular_slice,
                    grad_outputs=torch.ones_like(batch_output).to(self.DEFAULT_DEVICE),
                    create_graph=True,
            )
            grad_tensor[:,i,:] = model_grads[0]
        return grad_tensor
           
    def shap_values(
        self, 
        model: torch.nn.Module, 
        input_tensor: torch.FloatTensor, 
        sparse_labels: torch.LongTensor=None,
        batch_transformation: Callable=None,
    ) -> torch.FloatTensor:
        """
        Calculate expected gradients approximation of Shapley values for the 
        sample ``input_tensor``.

        Parameters
        ----------
        model : torch.nn.Module 
            Pytorch model for which the output should be explained.
        input_tensor : torch.Tensor 
            (batch, ...) tensor representing the input to be explained,
            where `...` are feature dimensions.
        sparse_labels : torch.LongTensor, optional
            (batch, classes) one-hot class labels.
            not required if only one output class is present.
        batch_transformation : Callable, optional.
            transformation to apply to reference batches after drawing.            
            
        Returns
        -------
        expected_grads : torch.FloatTensor
            (batch, ...) expected gradients for each sample in the input.
        """
        # set device to use
        self.DEFAULT_DEVICE = list(model.parameters())[0].device
        # set a batch transformation if applicable
        self.batch_transformation = batch_transformation
        if batch_transformation is not None and not callable(batch_transformation):
            msg = "`batch_transformation` arguments must be callable."
            raise TypeError(msg)
        # sample a batch from the reference dataset and reshape
        # to match the inputs
        reference_tensor = self._get_ref_batch()
        shape = reference_tensor.shape
        reference_tensor = reference_tensor.view(
                self.batch_size, 
                self.k, 
                *(shape[1:]),
        ).to(self.DEFAULT_DEVICE)
        # get interpolation points between provided inputs and the 
        # assigned reference sample for each sample in the batch
        samples_input = self._get_samples_input(input_tensor, reference_tensor)
        # compute the difference across each feature between
        # input and reference samples
        samples_delta = self._get_samples_delta(input_tensor, reference_tensor)
        # debugging
        logger.debug(f"samples_input size: {samples_input.size()}")
        logger.debug(f"samples_delta size: {samples_delta.size()}")
        logger.debug(f"reference_tensor size: {reference_tensor.size()}")
        logger.debug(f"sparse_labels size: {sparse_labels.size()}")

        # compute gradients on label scores w.r.t. the interpolation inputs
        grad_tensor = self._get_grads(samples_input, model, sparse_labels)
        # scale the gradient tensor by the difference 
        mult_grads = samples_delta * grad_tensor if self.scale_by_inputs else grad_tensor
        expected_grads = mult_grads.mean(1)
        return expected_grads

class VariableBatchExplainer(AttributionPriorExplainer):
    """
    Subclasses AttributionPriorExplainer to avoid pre-specified batch size. Will adapt batch
    size based on shape of input tensor.
    """
    def __init__(self, background_dataset, random_alpha=True,scale_by_inputs=True):
        """
        Arguments:
        background_dataset: PyTorch dataset - may not work with iterable-type (vs map-type) datasets
        random_alpha: boolean - Whether references should be interpolated randomly (True, corresponds
            to Expected Gradients) or on a uniform grid (False - corresponds to Integrated Gradients)
        """
        self.random_alpha = random_alpha
        self.k = None
        self.scale_by_inputs=scale_by_inputs
        self.ref_set = background_dataset
        self.ref_sampler = DataLoader(
                dataset=background_dataset, 
                batch_size=1, 
                shuffle=True, 
                drop_last=True)
        self.refs_needed = -1
        return

    def _get_ref_batch(self,refs_needed=None):
        """
        Arguments:
        refs_needed: int - number of references to provide
        """
        if refs_needed!=self.refs_needed:
            self.ref_sampler = DataLoader(
                dataset=self.ref_set, 
                batch_size=refs_needed, 
                shuffle=True, 
                drop_last=True)
            self.refs_needed = refs_needed
        return next(iter(self.ref_sampler))[0].float()
           
    def shap_values(self, model, input_tensor, sparse_labels=None,k=1):
        """
        Arguments:
        base_model: PyTorch network
        input_tensor: PyTorch tensor to get attributions for, as in normal torch.nn.Module API
        sparse_labels:  np.array of sparse integer labels, i.e. 0-9 for MNIST. Used if you only
            want to explain the prediction for the true class per sample.
        k: int - Number of references to use default for explanations. As low as 1 for training.
            100-200 for reliable explanations. 
        """
        self.k = k
        n_input = input_tensor.shape[0]
        refs_needed = n_input*self.k
        # This is a reasonable check but prevents compatibility with non-Map datasets
        assert refs_needed<=len(self.ref_set), "Can't have more samples*references than there are reference points!"
        reference_tensor = self._get_ref_batch(refs_needed)
        shape = reference_tensor.shape
        reference_tensor = reference_tensor.view(
                n_input, 
                self.k,
                *(shape[1:])).to(DEFAULT_DEVICE)
        samples_input = self._get_samples_input(input_tensor, reference_tensor)
        samples_delta = self._get_samples_delta(input_tensor, reference_tensor)
        grad_tensor = self._get_grads(samples_input, model, sparse_labels)
        mult_grads = samples_delta * grad_tensor if self.scale_by_inputs else grad_tensor
        expected_grads = mult_grads.mean(1)

        return expected_grads

class ExpectedGradientsModel(torch.nn.Module):
    """
    Wraps a PyTorch model (one that implements torch.nn.Module) so that model(x)
    produces SHAP values as well as predictions (controllable by 'shap_values'
    flag.
    """
    def __init__(self,base_model,refset,k=1,random_alpha=True,scale_by_inputs=True):
        """
        Arguments:
        base_model: PyTorch network that subclasses torch.nn.Module
        refset: PyTorch dataset - may not work with iterable-type (vs map-type) datasets
        k: int - Number of references to use by default for explanations. As low as 1 for training.
            100-200 for reliable explanations. 
        """
        super(ExpectedGradientsModel,self).__init__()
        self.k = k
        self.base = base_model
        self.refset = refset
        self.random_alpha = random_alpha
        self.exp = VariableBatchExplainer(
            self.refset,
            random_alpha=random_alpha,
            scale_by_inputs=scale_by_inputs,
        )
        
    def forward(self,x,shap_values=False,sparse_labels=None,k=1):
        """
        Arguments:
        x: PyTorch tensor to predict with, as in normal torch.nn.Module API
        shap_values:     Binary flag -- whether to produce SHAP values
        sparse_labels:  np.array of sparse integer labels, i.e. 0-9 for MNIST. Used if you only
            want to explain the prediction for the true class per sample.
        k: int - Number of references to use default for explanations. As low as 1 for training.
            100-200 for reliable explanations. 
        """
        output = self.base(x)
        if not shap_values: return output
        else: shaps = self.exp.shap_values(self.base,x,sparse_labels=sparse_labels,k=k)
        return output, shaps
    
    
    
def tmp():
    """
    def convert_csr_to_sparse_tensor_inputs(X):
        coo = sp.coo_matrix(X)
        indices = np.mat([coo.row, coo.col]).transpose()
        return indices, coo.data, coo.shape
    
    def graph_mult(values, indices, shape, y):
        # sparse tensor multiplication function
        x_tensor = tf.SparseTensor(indices, values, shape)    
        out_layer = tf.sparse_tensor_dense_matmul(x_tensor, y)
        return out_layer
        
    def adj_to_lap(x):
        # calculate graph laplacian from adjacency matrix
        rowsum = np.array(x.sum(1))
        D = sp.diags(rowsum)
        return D - x
        
    adj = adj_to_lap(adj)
    adj_indices, adj_values, adj_shape = convert_csr_to_sparse_tensor_inputs(adj)
    
    # ... during training ...
    ma_eg = tf.reduce_mean(tf.abs(expected_gradients_op),axis=0)
    graph_reg = tf.matmul(tf.transpose(graph_mult(adj_values, adj_indices, adj_shape, ma_eg[145:,:])),ma_eg[145:,:])
    """
    # pass
    return
    
    
class GeneSetGraphPrior(object):
    
    def __init__(
        self,
        gene_sets: dict,
        gene_names: list,
        weighting: str="count",
    ) -> None:
        """Apply a graph prior on gradients within gene sets and
        an l1 penalty on genes outside of gene sets.
        
        Parameters
        ----------
        weighting : str
            weighting for the gene-gene co-occurence matrix.
            "count" - uses the integer count of co-occurences.
            "binary" - uses a binary matrix.
            "sqrt" - uses a square root transform of the count matrix.
            "log" - uses a log+1 transform of the count matrix.
        
        Notes
        -----
        
        .. math::
        
            \Omega(\Phi(\theta, X)) = \sum_{i,j} W_{i,j} (\hat \phi_i - \hat \phi_j)
        """
        self.gene_names = gene_names
        self.gene_sets = gene_sets
        self.weighting = weighting
        
        self._set_prior_matrix_from_gene_sets()
        return
    
    def _weight_gene_matrix(self,) -> None:
        """Weight the gene gene matrix"""
        if self.weighting == "count":
            return
        elif self.weighting == "binary":
            self.gene_gene_matrix = self.gene_gene_matrix.bool().float()
        elif self.weighting == "sqrt":
            self.gene_gene_matrix = torch.sqrt(self.gene_gene_matrix)
        elif self.weighting == "log":
            self.gene_gene_matrix = torch.log1p(self.gene_gene_matrix)
        else:
            msg = f"{self.weighting} is not a valid weighting scheme."
            raise ValueError(msg)
        return    
    
    def _set_prior_matrix_from_gene_sets(self,) -> None:
        """Generate a prior matrix from a set of gene programs
        and gene names for the input variables.
        
        Returns
        -------
        None.
        Sets `prior_matrix` as a [n_hidden, n_genes] tensor where
        `1` values indicate membership of a gene within a latent var.
        """
        self.gene_set_names = sorted(list(self.gene_sets.keys()))
        
        # [n_programs, n_genes]
        P = torch.zeros(
            (len(self.gene_set_names), len(self.gene_names),)
        ).float()
        
        # cast to set for list comprehension speed
        gene_names = set(self.gene_names)
        
        for i, k in enumerate(self.gene_set_names):
            genes = self.gene_sets[k]
            bidx = torch.tensor(
                [x in genes for x in gene_names],
                dtype=torch.bool,
            ).float()
            P[i, :] = bidx
        
        self.prior_matrix = P
        print(P)
        # generate a [n_genes, n_genes] matrix of genes that
        # share a gene set using integer counts for the number
        # of gene sets where the two genes co-occur
        
        # G = [genes, programs] @ [programs, genes]
        # G[i, j] = # of programs where genes i,j co-occur
        G = P.T @ P
        self.gene_gene_matrix = G
        self.gene_gene_laplacian = adj2lap(G)
        self._weight_gene_matrix()
        logger.info("gene-gene adj matrix")
        logger.info(f"\t{self.gene_gene_matrix}")
        return
    
    def graph_penalty(
        self, 
        mu: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Compute the graph penalty given a set of attribution
        scores.
        
        Parameters
        ----------
        mu : torch.FloatTensor
            [Features,] mean attribution scores.
        
        Returns
        -------
        penalty : torch.FloatTensor
            graph penalty computed using the [Genes, Genes] matrix
            of shared ontology sets.
        """
        # computes the matrix product of attribution means
        # and the graph Laplacian of shape [genes, genes]
        # L = D - A, such that:
        #   L[i, j] = degree(vertex_i) if i == j
        #   L[i, j] = -1 if there is an edge v_i->v_j
        #   L[i, j] = 0 otherwise
        #
        # following the matrix multiplication, we perform
        # `(means,)` @ L which multiplies each col of `L`
        # with the mean attribution on each gene
        # the gradient on the gene `i` is multiplied with
        # it's corresponding degree, and all other gradients
        # that are adjacent to `i` are multiplied by `-1`.
        # Taking the sum across the products in the matmul
        # effectively computed the difference between gene `i`
        # and all of it's neighbors because `i` is scaled by 
        # the degree (e.g. # of neighbors) and all the neighbors
        # are now negative values
        mu = mu.view(-1, 1)
        L = self.gene_gene_laplacian.to(device=mu.device)
        penalty = mu.T @ L @ mu
        return penalty.view(-1)
    
    def sparse_penalty(
        self,
        attr_grad: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Compute a sparsity penalty on genes that are
        outside of any gene set
        
        Parameters
        ----------
        mu : torch.FloatTensor
            [Features,] mean attribution scores.
            
        Returns
        -------
        penalty : torch.FloatTensor
            sparsity penalty on genes that don't exist
            in any gene set.
        """
        return
    
    def grn_penalty(
        self,
        attr_grad: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Compute a sparsity penalty across GRNs such that
        only a few GRNs are receiving strong attribution scores
        for a given class."""
        return
        
    def __call__(
        self,
        attr_grad: torch.FloatTensor,
    ) -> torch.FloatTensor:
        # reduce to mean attributions on each gene
        mu = torch.mean(attr_grad, dim=0)        
        # compute the graph penalty on the attributions
        p_graph = self.graph_penalty(mu)
        # compute the sparsity penalty on the attributions
        #p_sparse = self.sparse_penalty(mu)
        # compute the total prior
        #prior = p_graph + p_sparse
        prior = p_graph
        return prior
