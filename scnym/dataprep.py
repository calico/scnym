import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Callable


class SingleCellDS(Dataset):
    '''Dataset class for loading single cell profiles.

    Attributes
    ----------
    X : torch.FloatTensor
        [Cells, Genes] cell profiles.
    y : torch.LongTensor
        [Cells,] integer class labels.
    transform : Callable
        performs data transformation operations on a
        `sample` dict.
    '''

    def __init__(self,
                 X: np.ndarray,
                 y: np.ndarray,
                 transform: Callable = None,) -> None:
        '''
        Load single cell expression profiles.

        Parameters
        ----------
        X : np.ndarray
            [Cells, Genes] expression count matrix.
            scNym tools expect ln(Counts Per Million + 1).
        y : np.ndarray
            [Cells,] integer cell type labels.
        transform : Callable
            transform to apply to samples.

        Returns
        -------
        None.
        '''
        super(SingleCellDS, self).__init__()
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.transform = transform

        if not self.X.size(0) == self.y.size(0):
            sizes = (self.X.size(0), self.y.size(0))
            raise ValueError(
                'X rows %d not equal to y rows %d.' % sizes)
        return

    def __len__(self,) -> int:
        '''Return the number of examples in the data set.'''
        return self.X.size(0)

    def __getitem__(self, idx: int,) -> dict:
        '''Get a single cell expression profile and corresponding label.

        Parameters
        ----------
        idx : int
            index value in `range(len(self))`.

        Returns
        -------
        sample : dict
            'input' - torch.FloatTensor, input vector
            'output' - torch.LongTensor, target label
        '''
        # check if the idx value is valid given the dataset size
        if idx < 0 or idx > len(self):
            vals = (idx, len(self))
            raise ValueError(
                'idx %d is invalid for dataset with %d examples.' % vals)

        # retrieve relevant sample vector and associated label
        # store in a hash table for later manipulation and retrieval
        input_ = self.X[idx, ...]
        label = self.y[idx]

        sample = {'input': input_,
                  'output': label}

        # if a transformer was supplied, apply transformations
        # to the sample vector and label
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


def balance_classes(y: np.ndarray,
                    class_min: int = 128) -> np.ndarray:
    '''
    Perform class balancing by undersampling majority classes
    and oversampling minority classes, down to a minimum value.

    Parameters
    ----------
    y : np.ndarray
        class assignment indices.
    class_min : int
        minimum number of examples to use for a class.
        below this value, minority classes will be oversampled
        with replacement.

    Returns
    -------
    all_idx : np.ndarray
        indices for balanced classes. some indices may be repeated.
    '''
    # determine the size of the smallest class
    # if < `class_min`, we oversample to `class_min` samples.
    classes, counts = np.unique(y, return_counts=True)
    min_count = int(np.min(counts))
    if min_count < class_min:
        min_count = class_min

    # generate indices with equal representation of each class
    all_idx = []
    for i, c in enumerate(classes):
        class_idx = np.where(y == c)[0].astype('int')
        rep = counts[i] < min_count  # oversample minority classes
        if rep:
            print('Count for class %s is %d. Oversampling.' % (c, counts[i]))
        ridx = np.random.choice(class_idx, size=min_count, replace=rep)
        all_idx += [ridx]
    all_idx = np.concatenate(all_idx).astype('int')
    return all_idx


class LibrarySizeNormalize(object):
    '''Perform library size normalization.'''

    def __init__(self, counts_per_cell_after: int = int(1e6),) -> None:
        self.counts_per_cell_after = counts_per_cell_after
        return

    def __call__(self, sample: dict) -> dict:
        '''Perform library size normalization in-place
        on a sample dict.

        Parameters
        ----------
        sample : dict
            'input' - torch.FloatTensor, input vector
            'output' - torch.LongTensor, target label

        Returns
        -------
        sample : dict
            'input' - torch.FloatTensor, input vector
            'output' - torch.LongTensor, target label
        '''
        input_ = sample['input']
        size = torch.sum(input_)

        # get proportions of each feature per sample,
        # scale by `counts_per_cell_after`
        prop_input_ = input_ / size
        norm_input_ = prop_input_ * self.counts_per_cell_after
        sample['input'] = norm_input_
        return sample


class RandomCounts(object):
    '''Add random gene counts with Poisson frequency to
    a [Cells, Genes] expression matrix
    '''

    def __init__(self, rate: float = 1.,) -> None:
        '''Add random gene counts with Poisson frequency to
        a [Cells, Genes] expression matrix.

        Parameters
        ----------
        rate : float
            rate parameter for the Poisson noise distribution.

        Returns
        -------
        None.
        '''
        self.rate = rate
        return

    def __call__(self, sample: dict,) -> dict:
        '''
        Add Poisson noise to the sample matrix.

        Parameters
        ----------
        sample : dict
            'input' - torch.FloatTensor, input vector
            'output' - torch.LongTensor, target label

        Returns
        -------
        sample : dict
            'input' - torch.FloatTensor, input vector
            'output' - torch.LongTensor, target label
        '''
        X = sample['input']
        R = torch.from_numpy(
            np.random.poisson(lam=self.rate,
                              size=X.shape)).float()
        sample['input'] = X + R
        return sample


class RandomGainLoss(object):
    '''Randomly gain and lose counts with Poisson frequency'''

    def __init__(self, rate: float = 0.5,) -> None:
        '''Randomly gain and lose counts with Poisson frequency.

        Parameters
        ----------
        rate : float
            rate parameter for the Poisson noise distribution.

        Returns
        -------
        None.
        '''
        self.rate = rate
        return

    def __call__(self, sample: dict,) -> dict:
        '''
        Add and subtract Poisson noise.

        Parameters
        ----------
        sample : dict
            'input' - torch.FloatTensor, input vector
            'output' - torch.LongTensor, target label

        Returns
        -------
        sample : dict
            'input' - torch.FloatTensor, input vector
            'output' - torch.LongTensor, target label
        '''
        # generate Poisson sampled matrices of
        # additive and subtractive noise
        input_ = sample['input']  # [Genes,]
        loss = np.random.poisson(lam=self.rate, size=input_.size())
        gain = np.random.poisson(lam=self.rate, size=input_.size())

        # noise samples and truncate at `0` to ensure no values
        # fall into an undefined regime
        input_ = input_ - torch.from_numpy(loss).float()
        input_ = input_ + torch.from_numpy(gain).float()
        input_ = torch.nn.functional.relu(input_)

        sample['input'] = input_
        return sample
