'''Tools for interpreting trained scNym models'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import typing
import copy


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
