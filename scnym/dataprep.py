import torch
import numpy as np
from scipy import sparse
from torch.utils.data import Dataset
from typing import Callable, Any, Union


class SingleCellDS(Dataset):
    '''Dataset class for loading single cell profiles.

    Attributes
    ----------
    X : np.ndarray, sparse.csr_matrix
        [Cells, Genes] cell profiles.
    y_labels : np.ndarray, sparse.csr_matrix
        [Cells,] integer class labels.
    y : torch.FloatTensor
        [Cells, Classes] one hot labels.
    transform : Callable
        performs data transformation operations on a
        `sample` dict.
    num_classes : int
        number of classes in the dataset. default `-1` infers
        the number of classes as `len(unique(y))`.
    '''

    def __init__(
        self,
        X: Union[sparse.csr.csr_matrix, np.ndarray],
        y: Union[sparse.csr.csr_matrix, np.ndarray],
        domain: Union[sparse.csr.csr_matrix, np.ndarray] = None,
        transform: Callable = None,
        num_classes: int=-1,
        num_domains: int=-1,
    ) -> None:
        '''
        Load single cell expression profiles.

        Parameters
        ----------
        X : np.ndarray, sparse.csr_matrix
            [Cells, Genes] expression count matrix.
            scNym tools expect ln(Counts Per Million + 1).
        y : np.ndarray, sparse.csr_matrix
            [Cells,] integer cell type labels.
        domain : np.ndarray, sparse.csr_matrix
            [Cells,] integer domain labels.
        transform : Callable
            transform to apply to samples.
        num_classes : int
            total number of classes for the task.
        num_domains : int
            total number of domains for the task.

        Returns
        -------
        None.
        '''
        super(SingleCellDS, self).__init__()
        
        # check types on input arrays
        if type(X) not in (np.ndarray, sparse.csr_matrix,):
            msg = f'X is type {type(X)}, must `np.ndarray` or `sparse.csr_matrix`'
            raise TypeError(msg)

        if type(y) not in (np.ndarray, sparse.csr_matrix,):
            msg = f'X is type {type(y)}, must `np.ndarray` or `sparse.csr_matrix`'
            raise TypeError(msg)
            
        if type(y) != np.ndarray:
            # densify labels
            y = y.toarray()
        
        self.X = X
        self.y_labels = torch.from_numpy(y).long()
        self.y = torch.nn.functional.one_hot(
            self.y_labels,
            num_classes=num_classes,
        ).float()
        
        self.dom_labels = domain
        if self.dom_labels is not None:
            self.dom = torch.nn.functional.one_hot(
                torch.from_numpy(self.dom_labels).long(),
                num_classes=num_domains,
            ).float()
        else:
            self.dom = np.zeros_like(self.y) - 1
        
        self.transform = transform

        if not self.X.shape[0] == self.y.shape[0]:
            sizes = (self.X.shape[0], self.y.shape[0])
            raise ValueError(
                'X rows %d not equal to y rows %d.' % sizes)
        return

    def __len__(self,) -> int:
        '''Return the number of examples in the data set.'''
        return self.X.shape[0]

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
        if type(idx) != int:
            raise TypeError(
                f'indices must be int, you passed {type(idx)}, {idx}'
            )
        
        # check if the idx value is valid given the dataset size
        if idx < 0 or idx > len(self):
            vals = (idx, len(self))
            raise ValueError(
                'idx %d is invalid for dataset with %d examples.' % vals)

        # retrieve relevant sample vector and associated label
        # store in a hash table for later manipulation and retrieval
        
        # input_ is either an `np.ndarray` or `sparse.csr.csr_matrix`
        input_ = self.X[idx, ...]
        # label is already a `torch.Tensor`
        label = self.y[idx]
        
        # if the corresponding vectors are sparse, convert them to dense
        # we perform this operation on a samplewise-basis to avoid
        # storing the whole count matrix in dense format
        if type(input_) != np.ndarray:
            input_ = input_.toarray()
            
        input_ = torch.from_numpy(input_).float()
        if input_.size(0) == 1:
            input_ = input_.squeeze()
        
        sample = {
            'input': input_,
            'output': label,
        }
        
        sample['domain'] = self.dom[idx]

        # if a transformer was supplied, apply transformations
        # to the sample vector and label
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


def balance_classes(
    y: np.ndarray,
    class_min: int = 256,
) -> np.ndarray:
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

    def __init__(
        self, 
        counts_per_cell_after: int = int(1e6),
        log1p: bool=True,
    ) -> None:
        self.counts_per_cell_after = counts_per_cell_after
        self.log1p = log1p
        return

    def __call__(
        self, 
        sample: dict,
    ) -> dict:
        '''Perform library size normalization in-place
        on a sample dict.

        Parameters
        ----------
        sample : dict
            'input' - torch.FloatTensor, input vector [N, C]
            'output' - torch.LongTensor, target label [N,]

        Returns
        -------
        sample : dict
            'input' - torch.FloatTensor, input vector [N, C]
            'output' - torch.LongTensor, target label [N,]
        '''
        input_ = sample['input']
        size = torch.sum(input_, dim=1).reshape(-1, 1)

        # get proportions of each feature per sample,
        # scale by `counts_per_cell_after`
        prop_input_ = input_ / size
        norm_input_ = prop_input_ * self.counts_per_cell_after
        if self.log1p:
            norm_input_ = torch.log1p(norm_input_)
        sample['input'] = norm_input_
        return sample
    
    
class ExpMinusOne(object):
    
    def __init__(
        self,
    ) -> None:
        '''Perform an exponential minus one transformation
        on an input vector'''
        return
    
    def __call__(
        self,
        sample: dict,
    ) -> dict:
        '''Perform an exponential minus one transformation
        on the sample input.'''
        sample['input'] = torch.expm1(
            sample['input'],
        )
        return sample

    
class MultinomialSample(object):
    '''Sample an mRNA abundance profile from a multinomial
    distribution parameterized by observations.
    '''
    
    def __init__(
        self,
        depth: tuple=(10000, 100000),
        depth_ratio: tuple=None,
    ) -> None:
        '''Sample an mRNA abundance profile from a multinomial 
        distribution parameterized by observations.
        
        Parameters
        ----------
        depth : tuple
            (min, max) depth for multinomial sampling.
        depth_ratio : tuple
            (min, max) ratio of profile depth for multinomial
            sampling. supercedes `depth`.
        
        Returns
        -------
        None.
        '''
        self.depth = depth
        self.depth_ratio = depth_ratio
        
        if self.depth_ratio is not None:
            self.depth = None
        
        return
    
    def __call__(
        self,
        sample: dict,
    ) -> dict:
        '''
        Sample an mRNA profile from a multinomial
        parameterized by observations.
        
        Parameters
        ----------
        sample : dict
            'input' - torch.FloatTensor, input vector [N, C]
            'output' - torch.LongTensor, target label [N,]

        Returns
        -------
        sample : dict
            'input' - torch.FloatTensor, input vector [N, C]
            'output' - torch.LongTensor, target label [N,]
            
        Notes
        -----
        We perform multinomial sampling with a call to `np.random.multinomial`
        for each observation. This may be faster in the future using the native
        `torch.distributions.Multinomial`, but right now the sampling procedure
        is incredibly slow. The implementation below is ~100X slower than our 
        `numpy` calls.
        
            ```
            multi = torch.distributions.Multinomial(
              total_count=d,
              probs=p,
            )
        
            m = multi.sample()
            m = m.float()
            ```
            
        Follow:
        https://github.com/pytorch/pytorch/issues/11931
        '''
        # input is a torch.FloatTensor
        # we assume x is NOT log-transformed
        # cast to float64 to preserve precision of proportions
        x = sample['input'].to(torch.float64)
        size = torch.sum(x, dim=1).detach().cpu().numpy()
        
        # generate a relative abundance profile
        p = (x / torch.sum(x, dim=1).reshape(-1, 1))
        # normalize to ensure roundoff errors don't
        # give us p.sum() > 1
        idx = torch.where(p.sum(1) > 1)
        for i in idx[0]:
            p[i, :] = p[i, :] / np.min([p[i, :].sum(), 1.])
        # sample a sequencing depth
        if self.depth_ratio is None:
            # tile the specified depth for all cells
            depth = np.tile(
                np.array(self.depth).reshape(1, -1),
                (x.size(0), 1)
            ).astype(np.int)
        else:
            # compute a range of depths based on the library size
            # of each observation
            depth = np.concatenate(
                [
                    np.floor(self.depth_ratio[0] * size).reshape(-1, 1),
                    np.ceil(self.depth_ratio[1] * size).reshape(-1, 1),
                ],
                axis = 1,
            ).astype(np.int)
        
        # sample from a multinomial
        # np.random.multinomial is ~100X faster than the native
        # torch.distributions.Multinomial, implemented in Notes
        m = np.zeros(x.size())
        for i in range(x.size(0)):
            
            d = int(
                np.random.choice(
                    np.arange(depth[i, 0], depth[i, 1]),
                    size=1,
                )
            ) 
            
            m[i, :] = np.random.multinomial(
                d,
                pvals=p[i, :].detach().cpu().numpy(),
            )
        m = torch.from_numpy(m).float()
        m = m.to(device=x.device)
        output = {
            'input'  : m,
            'output' : sample['output'],
        }
        return output
    
    
class GeneMasking(object):
    
    def __init__(
        self,
        p_drop: float=0.1,
        p_apply: float=0.5,
        sample_p_drop: bool=False,
    ) -> None:
        '''Mask a subset of genes in the gene expression vector
        with zeros. This may simulate a failed detection event.
        This mask is applied to `p_apply`*100% of input vectors.
        
        Parameters
        ----------
        p_drop : float
            proportion of genes to mask with zeros.
        p_apply : float
            proportion of samples to mask.
        sample_p_drop : bool
            sample the proportion of genes to drop from
            `Unif(0, p_drop)`.
        
        Returns
        -------
        None.
        '''
        self.p_drop = p_drop
        self.p_apply = p_apply
        self.sample_p_drop = sample_p_drop
        return

    def __call__(
        self,
        sample: dict,
    ) -> dict:
        '''Mask a subset of genes.'''
        do_apply = np.random.random()
        if do_apply > self.p_apply:
            # no-op
            return sample
        
        # input is a torch.FloatTensor
        x = sample['input'].clone()
        
        if self.sample_p_drop:
            p_drop = np.random.random() * self.p_drop
        else:
            p_drop = self.p_drop
        
        # mask a proportion `p` of genes with `0`
        # assume x [N, Genes]
        n_genes = x.size(1)
        for i in range(x.size(0)):
            idx = np.random.choice(
                np.arange(n_genes),
                size=int(np.floor(n_genes*p_drop)),
                replace=False,
            ).astype(np.int)
            x[i, idx] = 0
        
        sample['input'] = x
        return sample
    

class InputDropout(object):
    
    def __init__(
        self,
        p_drop: float=0.1,
    ) -> None:
        '''Randomly mask `p_drop` genes.
        
        Parameters
        ----------
        p_drop : float
            proportion of genes to mask.
            
        Returns
        -------
        None
        '''
        self.p_drop = p_drop
        return
    
    def __call__(
        self,
        sample: dict,
    ) -> dict:
        sample['input'] = torch.nn.functional.dropout(
            sample['input'], 
            p=self.p_drop, 
            inplace=False,
        )
        return sample
    
    
class PoissonSample(object):
    '''Sample a gene expression profile based on gene-specific
    Poisson distributions'''
    
    def __init__(
        self,
        depth_range: tuple=(0.5, 2.),
    ) -> None:
        '''Sample a gene expression profile based on gene-specific
        Poisson distributions.
        
        Parameters
        ----------
        depth_range : tuple
            (min_factor, max_factor) for scaling the rate of the Poisson
            that samples are drawn from. Scaling down produces sparser 
            profiles, scaling up produces less sparse profiles.
        
        Returns
        -------
        None.
        
        Notes
        -----
        Treats a raw gene count as an estimate of the rate for a Poisson
        distribution.
        '''
        self.depth_range = depth_range
        return
    
    def __call__(
        self,
        sample: dict,
    ) -> dict:
        # input is a torch.FloatTensor
        # we assume x is NOT log-transformed
        x = sample['input'].to(torch.float64)
        # sample a scale factor for the rate in the specified interval
        # Unif(r1, r2) = Unif(0, 1) * (r1 - r2) + r2
        r = torch.rand(x.size(0)).to(device=x.device)
        r = r * (self.depth_range[0] - self.depth_range[1]) + self.depth_range[1]
        
        P = torch.distributions.Poisson(
            rate=x*r.view(-1, 1),
        )
        x_poisson = P.sample()
        
        assert x.size() == x_poisson.size()
        
        sample['input'] = x_poisson.float()
        return sample


'''Implement MixUp training'''


def mixup(
    a: torch.FloatTensor, 
    b: torch.FloatTensor,
    gamma: torch.FloatTensor,
) -> torch.FloatTensor:
    '''Perform a MixUp operation.
    This is effectively just a weighted average, where
    `gamma = 0.5` yields the mean of `a` and `b`.

    Parameters
    ----------
    a : torch.FloatTensor
        [Batch, C] first sample matrix.
    b : torch.FloatTensor
        [Batch, C] second sample matrix.
    gamma : torch.FloatTensor
        [Batch,] MixUp coefficient.
        
    Returns
    -------
    m : torch.FloatTensor
        [Batch, C] mixed sample matrix.
    '''
    return gamma*a + (1-gamma)*b


class SampleMixUp(object):

    def __init__(
        self, 
        alpha: float=0.2,
        keep_dominant_obs: bool=False,
    ) -> None:
        '''Perform a MixUp operation on a sample batch.

        Parameters
        ----------
        alpha : float
            alpha parameter of the Beta distribution.
        keep_dominant_obs : bool
            use max(gamma, 1-gamma) for each pair of samples
            so the identity of the dominant observation can be
            associated with the mixed sample.

        Returns
        -------
        None.

        References
        ----------
        mixup: Beyond Empirical Risk Minimization
        Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz
        arXiv:1710.09412

        Notes
        -----
        Zhang et. al. note alpha [0.1, 0.4] improve performance on CIFAR-10,
        while larger values of alpha induce underfitting.
        '''
        self.alpha = alpha
        self.beta = torch.distributions.beta.Beta(
            self.alpha, self.alpha,
        )
        self.keep_dominant_obs = keep_dominant_obs
        return

    def __call__(
        self, 
        sample: dict,
    ) -> dict:
        '''Perform a MixUp operation on the sample.

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
        if self.alpha == 0.:
            # mixup is deactivated, return the original
            # sample without mixing
            return sample
        
        input_ = sample['input']
        output = sample['output']

        # randomly permute the input and output
        ridx = torch.randperm(input_.size(0))
        r_input_ = input_[ridx]
        r_output = output[ridx]

        # perform the mixup operation between the source
        # data and the rearranged data -- random pairs
        gamma = self.beta.sample((input_.size(0),))
        if self.keep_dominant_obs:
            gamma, _ = torch.max(
                torch.stack([gamma, 1 - gamma,], dim=1,),
                dim=1,
            )
        gamma = gamma.reshape(-1, 1)
        # move gamma weights to the same device as the 
        # inputs
        gamma = gamma.to(device=input_.device)
        
        mix_input_ = mixup(input_, r_input_, gamma=gamma)
        mix_output = mixup(output, r_output, gamma=gamma)

        sample['input'] = mix_input_
        sample['output'] = mix_output
        
        # if there are additional tensors in sample, also mix
        # them up
        other_keys = [
            k for k in sample.keys() if k not in ('input', 'output')
        ]
        for k in other_keys:
            if type(sample[k]) == torch.Tensor:
                sample[k] = mixup(sample[k], sample[k][ridx], gamma=gamma)
        
        # add the randomization index to the sample in case
        # it's useful downstream
        sample['random_idx'] = ridx
        
        return sample


#################################################
# Define augmentation series
#################################################

from torchvision import transforms

def identity(x: Any) -> Any:
    '''Identity function'''
    return x

AUGMENTATION_SCHEMES = {
    'log1p_drop' : transforms.Compose([
        ExpMinusOne(),
        InputDropout(p_drop=0.1,),
        LibrarySizeNormalize(log1p=True),
    ]),
    'log1p_mask' : transforms.Compose([
        ExpMinusOne(),
        GeneMasking(
            p_drop=0.1,
            p_apply=0.5,
        ),
        LibrarySizeNormalize(log1p=True),
    ]),
    'log1p_poisson' : transforms.Compose([
        ExpMinusOne(),
        PoissonSample(),
        LibrarySizeNormalize(log1p=True),
    ]),
    'log1p_poisson_drop' : transforms.Compose([
        ExpMinusOne(),
        PoissonSample(depth_range=(0.1, 2.)),
        InputDropout(p_drop=0.1),
        LibrarySizeNormalize(log1p=True),
    ]),
    'None' : identity,
    'none' : identity,
    None : identity,
}
