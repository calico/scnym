import numpy as np
from scipy import sparse
import os
import torch
import torch.nn.functional as F
from typing import Union
from .model import CellTypeCLF
from .dataprep import SingleCellDS
import tqdm


class Predicter(object):
    '''Predict cell types from expression data using `CellTypeCLF`.

    Attributes
    ----------
    model_weights : list
        paths to model weights for classification.
    labels : list
        str labels for output classes.
    n_cell_types : int
        number of output classes.
    n_genes : int
        number of input genes.
    models : list
        `nn.Module` for each set of weights in `.model_weights`.
    '''

    def __init__(
        self,
        model_weights: Union[str, list, tuple],
        n_genes: int = None,
        n_cell_types: int = None,
        labels: list = None,
        **kwargs,
    ) -> None:
        '''
        Predict cell types using pretrained weights for `CellTypeCLF`.

        Parameters
        ----------
        model_weights : str, list, tuple
            paths to pre-trained model weights. if more than one
            path to weights is provided, predicts using an ensemble
            of models.
        n_genes : int
            number of genes in the input frame.
        n_cell_types : int
            number of cell types in the output.
        labels : list
            string labels corresponding to each cell type output
        **kwargs passed to `model.CellTypeCLF`
        '''
        if type(model_weights) == str:
            self.model_weights = [model_weights]
        else:
            self.model_weights = model_weights
        self.labels = labels

        if n_cell_types is None:
            # get the number of output nodes from the pretrained model
            print('Assuming `n_cell_types` is the same as in the \
            pretrained model weights.')
            params = torch.load(self.model_weights[0], map_location='cpu')
            fkey = list(params.keys())[-1]
            self.n_cell_types = len(params[fkey])
        else:
            self.n_cell_types = n_cell_types

        # check that all the specified weights exist
        for weights in self.model_weights:
            if not os.path.exists(weights):
                raise FileNotFoundError()

        if n_genes is None:
            # get the number of input genes from the model weights
            print('Assuming `n_genes` is the same as in the \
            pretrained model weights.')
            params = torch.load(model_weights, map_location='cpu')
            fkey = list(params.keys())[0]
            self.n_genes = params[fkey].shape[1]
        else:
            self.n_genes = n_genes

        # Load each set of weights in `model_weights` into a model
        # to use in an ensemble prediction.
        self.models = []
        for weights in self.model_weights:
            model = CellTypeCLF(
                n_genes=self.n_genes,
                n_cell_types=self.n_cell_types,
                **kwargs,
            )
            model.load_state_dict(torch.load(weights, map_location='cpu'))

            if torch.cuda.is_available():
                model = model.cuda()

            self.models.append(model.eval())

        return

    def predict(
        self,
        X: Union[np.ndarray, sparse.csr.csr_matrix, torch.FloatTensor],
        output: str = None,
        batch_size: int=1024,
         **kwargs,
    ) -> (np.ndarray, list):
        '''
        Predict cell types given a matrix `X`.

        Parameters
        ----------
        X : np.ndarray, sparse.csr.csr_matrix, torch.FloatTensor
            [Cells, Genes]
        output : str
            additional output to include as an optional third tuple.
            ('prob', 'score').
        batch_size : int
            batch size to use for predictions.

        Returns
        -------
        predictions : np.ndarray
            [Cells,] ints of predicted class
        names : list
            [Cells,] str of predicted class names
        probabilities : np.ndarray
            [Cells, Types] probabilities (softmax outputs).

        Notes
        -----
        acceptable **kwarg for legacy compatibility --
        return_prob : bool
            return probabilities as an optional third output.
        '''
        if not X.shape[1] == self.n_genes:
            gs = (X.shape[1], self.n_genes)
            raise ValueError('%d genes in X, %d genes in model.' % gs)

        if 'return_prob' in kwargs:
            return_prob = kwargs['return_prob']
        else:
            return_prob = None
            
        if output not in ['prob', 'score'] and output is not None:
            msg = f'{output} is not a valid additional output.'
            raise ValueError(msg)

        # build a SingleCellDS so we can load cells onto the
        # GPU in batches
        ds = SingleCellDS(X=X, y=np.zeros(X.shape[0]))
        dl = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
        )

        # For each cell vector, compute a prediction
        # and a class probability vector.
        predictions = []
        scores = []
        probabilities = []

        # For each cell, compute predictions
        for data in tqdm.tqdm(dl, desc='Finding cell types'):
            
            X_batch = data['input']
            
            if torch.cuda.is_available():
                X_batch = X_batch.cuda()

            # take an average prediction across all models provided
            outs = []
            for model in self.models:
                out = model(X_batch)
                outs.append(out)
            outs = torch.stack(outs, dim=0)
            out = torch.mean(outs, dim=0)

            # save most likely prediction and output probabilities
            scores.append(
                out.detach().cpu().numpy()
            )
            
            _, pred = torch.max(out, 1)
            predictions.append(
                pred.detach().cpu().numpy()
            )

            probs = F.softmax(out, dim=1)
            probabilities.append(
                probs.detach().cpu().numpy()
            )

        predictions = np.concatenate(predictions, axis=0) # [Cells,]
        scores = np.concatenate(scores, axis=0)  # [Cells, Types]
        probabilities = np.concatenate(probabilities, axis=0)  # [Cells, Types]

        if self.labels is not None:
            names = []
            for i in range(len(predictions)):
                names += [self.labels[predictions[i]]]
        else:
            names = None

        # Parse the arguments to determine what to return
        # N.B. that `return_prob` here is to support legacy code
        # and may be removed in the future.
        if return_prob is True:
            return predictions, names, probabilities
        elif output is not None:
            if output == 'prob':
                return predictions, names, probabilities
            elif output == 'score':
                return predictions, names, scores
        else:
            return predictions, names
