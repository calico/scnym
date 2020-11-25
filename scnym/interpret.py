'''Tools for interpreting trained scNym models'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import typing
import copy
import warnings


class Salience(object):
    '''
    Performs backpropogation to compute gradients on a target
    class with regards to an input.

    Notes
    -----
    Saliency analysis computes a gradient on a target class
    score :math:`f_i(x)` with regards to some input :math:`x`.


    .. math::

        S_i = \frac{\partial f_i(x)}{\partial x}
    '''

    def __init__(
        self,
        model: nn.Module,
        class_names: np.ndarray,
        gene_names: np.ndarray = None,
        layer_to_hook: int=None,
        verbose: bool=False,
    ) -> None:
        '''
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
        '''
        # ensure class names are unique for each output node
        if len(np.unique(class_names)) != len(class_names):
            msg = '`class_names` must all be unique.'
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
        '''Set up hooks to record gradients from the first linear
        layer into a target tensor.

        References
        ----------
        https://pytorch.org/docs/stable/nn.html#torch.nn.Module.register_backward_hook
        '''

        def _record_gradients(module, grad_in, grad_out):
            '''Record gradients of a layer with the correct input
            shape'''
            self.gradients = grad_in[1]
            if self.verbose:
                print([x.size() if x is not None else 'None' for x in grad_in])
                print('Hooked gradients to: ', module)

        for module in self.model.modules():
            if isinstance(module, nn.Linear) and module.in_features == len(self.gene_names):
                    module.register_backward_hook(_record_gradients)
        return
    
    def _hook_nth_layer_gradients(self, n: int):
        '''Set up hooks to record gradients from an arbitrary layer.

        References
        ----------
        https://pytorch.org/docs/stable/nn.html#torch.nn.Module.register_backward_hook
        '''

        def _record_gradients(module, grad_in, grad_out):
            '''Record gradients of a layer with the correct input
            shape'''
            self.gradients = grad_in[1]
            if self.verbose:
                print([x.size() if x is not None else 'None' for x in grad_in])
                print('Hooked gradients to: ', module)

        module = list(self.model.modules())[n]
        module.register_backward_hook(_record_gradients)
        return    

    def _guided_backprop_hooks(self):
        '''Set up forward and backward hook functions to perform
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
        '''
        def _record_relu_outputs(module, in_, out_):
            '''Store the outputs to each ReLU layer'''
            self.rectified_outputs.append(out_,)
            self.store_rectified_outputs.append(out_,)

        def _clamp_grad(module, grad_in, grad_out):
            '''Clamp ReLU gradients to [0, inf] and return a 
            new gradient to be used in subsequent outputs.
            '''
            self.store_grad.append(grad_in[0])

            grad = grad_in[0].clamp(min=0.0)
            self.store_clamped_grad.append(grad)

            # here we pop the outputs off to ensure that the
            # final output is always the current ReLU layer
            # we're investigating
            last_relu_output = self.rectified_outputs.pop()
            last_relu_output = copy.copy(
                last_relu_output
            )
            last_relu_output[last_relu_output>0] = 1
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
        '''Compute the saliency of a target class on an input
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
        '''
        if target_class not in self.class_names:
            msg = f'{target_class} is not in `.class_names`'
            raise ValueError(msg)

        target_idx = np.where(
            target_class == self.class_names
        )[0].astype(np.int)
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
        '''
        Rank genes by saliency for a target class and input.

        Passes **kwargs to `.get_saliency` and uses the output
        to rank genes.

        Returns
        -------
        ranked_genes : np.ndarray
            gene names with high saliency, ranked highest to
            lowest.
        '''
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
        verbose: bool = False,
    ) -> None:
        '''Performs integrated gradient computations for feature attribution
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
        '''
        self.model = copy.deepcopy(model)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            print('Model loaded on CUDA compute device.')
        self.model.zero_grad()
        for param in self.model.parameters():
            param.requires_grad = False        
        
        self.class_names = class_names
        self.gene_names = gene_names
        self.verbose = verbose
        
        if type(self.class_names) == np.ndarray:
            self.class_names = self.class_names.tolist()
                
        return
    
    
    def get_grad(
        self,
        x: torch.Tensor,
        target_class: str,
    ) -> (torch.Tensor, torch.Tensor):
        '''Get the gradient for a minibatch of observations with respect
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
        '''
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
    
    
    def _check_integration(
        self,
        integrated_grad: torch.Tensor,
    ) -> bool:
        '''Check that the approximation of the path integral is appropriate.
        If we used a sufficient number of steps in the Reimann sum, we should
        find that the gradient sum is roughly equivalent to the difference in
        class scores for the baseline vector and target vector.
        '''
        score_difference = self.raw_scores[-1] - self.raw_scores[0]
        check = torch.isclose(
            integrated_grad.sum(),
            score_difference,            
            rtol=0.1,
        )
        if not check:
            msg = 'integrated gradient magnitude does not match the difference in scores.\n'
            msg += f'magnitude {integrated_grad.sum().item()} vs. {score_difference.item()}.\n'
            msg += 'consider using more steps to estimate the path integral.'
            warnings.warn(msg)
        return check
    
    def get_integrated_gradient(
        self,
        x: torch.Tensor,
        target_class: str,
        M: int=300,
        baseline: torch.Tensor=None,
    ) -> torch.Tensor:
        '''Compute the integrated gradient for a single observation.
        
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
        '''
        if baseline is None:
            if self.verbose:
                print('Using the 0-vector as a baseline.')
            base = self.baseline_input = torch.zeros(
                (1, len(self.gene_names))
            ).float()
        else:
            base = self.baseline_input = baseline
            if base.dim() > 1 and base.size(0) != 1:
                msg = 'baseline must be a single gene expression vector'
                raise ValueError(msg)
            base = base.view(1, -1)
        
        self.target_class = target_class

        if x.dim() > 1 and x.size(0) == 1:
            # tensor has an empty batch dimension, flatten it
            x = x.view(-1)
        
        # create a batch of observations where each observation is
        # a single step along the path integral
        path = base.repeat((M, 1))
        
        # create a tensor marking the "step number" for each observation
        step = ( (x - base) / M).view(1, -1)
        step_coord = torch.arange(1, M+1).view(-1, 1).repeat((1, path.size(1)))
        
        # add the correct number of steps to fill the path tensor
        path += (step * step_coord)
        
        if self.verbose:
            print('baseline', base.size())
            print(base.sort())
            print('observation', x.size())
            print(x.sort())
            print()
            print('step : ', step.size())
            print(step)
            print('step_coord : ', step_coord.size())
            print(step_coord)
            print('path : ', path.size())
            print(path[0].sort())
            print('-'*3)
            print(path[-1].sort())
                
        # compute the gradient on the input at each step
        # along the path
        gradients = torch.zeros_like(path)
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
        integrated_grad = (x - base) * (gradients.sum(0) / M)
        
        self._check_integration(integrated_grad)
        
        return integrated_grad
    
    
class Tessaract(IntegratedGradient):
    
    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        """Tessaract finds a path from a source vector in feature
        space to a destination vector that maximizes the likelihood
        of observing each intermediate position using a trained
        classification model.
        """