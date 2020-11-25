'''
Utility functions
'''
import torch
import numpy as np
import anndata
from scipy import sparse
import pandas as pd
import tqdm
from scipy import stats
import scanpy as sc
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from sklearn.metrics.pairwise import euclidean_distances
from typing import Union, Callable


def make_one_hot(
    labels: torch.LongTensor, 
    C=2,
) -> torch.FloatTensor:
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.

    Parameters
    ----------
    labels : torch.LongTensor or torch.cuda.LongTensor
        [N, 1], where N is batch size.
        Each value is an integer representing correct classification.
    C : int
        number of classes in labels.

    Returns
    -------
    target : torch.FloatTensor or torch.cuda.FloatTensor
        [N, C,], where C is class number. One-hot encoded.
    '''
    if labels.ndimension() < 2:
        labels = labels.unsqueeze(1)
    one_hot = torch.zeros([labels.size(0), C, ],
                          dtype=torch.float32, device=labels.device)
    target = one_hot.scatter_(1, labels, 1)

    return target


def l1_layer0(
    model: torch.nn.Module,
) -> torch.FloatTensor:
    '''Compute l1 norm for the first input layer of
    a `CellTypeCLF` model.

    Parameters
    ----------
    model : torch.nn.Module
        CellTypeCLF model with `.classif` module.

    Returns
    -------
    l1_reg : torch.FloatTensor
        [1,] l1 norm for the first layer parameters.
    '''
    # get the parameters of the first classification layer
    layer0 = list(model.classif.modules())[1]
    params = layer0.parameters()
    l1_reg = None

    # compute the l1_norm
    for W in params:
        if l1_reg is None:
            l1_reg = W.norm(1)
        else:
            l1_reg = l1_reg + W.norm(1)
    return l1_reg


def append_categorical_to_data(
    X: Union[np.ndarray, sparse.csr.csr_matrix],
    categorical: np.ndarray,
) -> (Union[np.ndarray, sparse.csr.csr_matrix], np.ndarray):
    '''Convert `categorical` to a one-hot vector and append
    this vector to each sample in `X`.

    Parameters
    ----------
    X : np.ndarray, sparse.csr.csr_matrix
        [Cells, Features]
    categorical : np.ndarray
        [Cells,]

    Returns
    -------
    Xa : np.ndarray
        [Cells, Features + N_Categories]
    categories : np.ndarray
        [N_Categories,] str category descriptors.
    '''
    # `pd.Categorical(xyz).codes` are int values for each unique
    # level in the vector `xyz`
    labels = pd.Categorical(categorical)
    idx = np.array(labels.codes)
    idx = torch.from_numpy(idx.astype('int32')).long()
    categories = np.array(labels.categories)

    one_hot_mat = make_one_hot(
        idx,
        C=len(categories),
    )
    one_hot_mat = one_hot_mat.numpy()
    assert X.shape[0] == one_hot_mat.shape[0], \
        'dims unequal at %d, %d' % (X.shape[0], one_hot_mat.shape[0])
    # append one hot vector to the [Cells, Features] matrix
    if sparse.issparse(X):
        X = sparse.hstack([X, one_hot_mat])
    else:
        X = np.concatenate([X, one_hot_mat], axis=1)
    return X, categories


def get_adata_asarray(
    adata: anndata.AnnData,
) -> Union[np.ndarray, sparse.csr.csr_matrix]:
    '''Get the gene expression matrix `.X` of an
    AnnData object as an array rather than a view.
    
    Parameters
    ----------
    adata : anndata.AnnData
        [Cells, Genes] AnnData experiment.
    
    Returns
    -------
    X : np.ndarray, sparse.csr.csr_matrix
        [Cells, Genes] `.X` attribute as an array
        in memory.
    
    Notes
    -----
    Returned `X` will match the type of `adata.X` view.
    '''
    if sparse.issparse(adata.X):
        X = sparse.csr.csr_matrix(adata.X)
    else:
        X = np.array(adata.X)
    return X


def build_classification_matrix(
    X: Union[np.ndarray, sparse.csr.csr_matrix],
    model_genes: np.ndarray,
    sample_genes: np.ndarray,
    gene_batch_size: int=512,
) -> Union[np.ndarray, sparse.csr.csr_matrix]:
    '''
    Build a matrix for classification using only genes that overlap
    between the current sample and the pre-trained model.

    Parameters
    ----------
    X : np.ndarray, sparse.csr_matrix
        [Cells, Genes] count matrix.
    model_genes : np.ndarray
        gene identifiers in the order expected by the model.
    sample_genes : np.ndarray
        gene identifiers for the current sample.
    gene_batch_size : int
        number of genes to copy between arrays per batch.
        controls a speed vs. memory trade-off.

    Returns
    -------
    N : np.ndarray, sparse.csr_matrix
        [Cells, len(model_genes)] count matrix.
        Values where a model gene was not present in the sample are left
        as zeros. `type(N)` will match `type(X)`.
    '''
    # check types
    if type(X) not in (np.ndarray, sparse.csr.csr_matrix):
        msg = f'X is type {type(X)}, must `np.ndarray` or `sparse.csr_matrix`'
        raise TypeError(msg)
    n_cells = X.shape[0]
    # check if gene names already match exactly
    if len(model_genes) == len(sample_genes):
        if np.all(model_genes == sample_genes):
            print('Gene names match exactly, returning input.')
            return X
    
    # instantiate a new [Cells, model_genes] matrix where columns
    # retain the order used during training
    if type(X) == np.ndarray:
        N = np.zeros((n_cells, len(model_genes)))
    else:
        # use sparse matrices if the input is sparse
        N = sparse.lil_matrix((n_cells, len(model_genes),))

    # map gene indices from the model to the sample genes
    model_genes_indices = []
    sample_genes_indices = []
    common_genes = 0
    for i, g in tqdm.tqdm(enumerate(sample_genes), desc='mapping genes'):
        if np.sum(g==model_genes) > 0:
            model_genes_indices.append(
                int(np.where(g==model_genes)[0])
            )
            sample_genes_indices.append(
                i,
            )
            common_genes += 1
            
    # copy the data in batches to the new array to avoid memory overflows
    gene_idx = 0
    n_batches = int(np.ceil(N.shape[1] / gene_batch_size))
    for b in tqdm.tqdm(range(n_batches), desc='copying gene batches'):
        model_batch_idx = model_genes_indices[gene_idx:gene_idx+gene_batch_size]
        sample_batch_idx = sample_genes_indices[gene_idx:gene_idx+gene_batch_size]
        N[:, model_batch_idx] = X[:, sample_batch_idx]
        gene_idx += gene_batch_size
            
    if sparse.issparse(N):
        # convert to `csr` from `csc`
        N = sparse.csr_matrix(N)
    print('Found %d common genes.' % common_genes)
    return N


def knn_smooth_pred_class(
    X: np.ndarray,
    pred_class: np.ndarray,
    grouping: np.ndarray = None,
    k: int = 15,
) -> np.ndarray:
    '''
    Smooths class predictions by taking the modal class from each cell's
    nearest neighbors.

    Parameters
    ----------
    X : np.ndarray
        [N, Features] embedding space for calculation of nearest neighbors.
    pred_class : np.ndarray
        [N,] array of unique class labels.
    groupings : np.ndarray
        [N,] unique grouping labels for i.e. clusters.
        if provided, only considers nearest neighbors *within the cluster*.
    k : int
        number of nearest neighbors to use for smoothing.

    Returns
    -------
    smooth_pred_class : np.ndarray
        [N,] unique class labels, smoothed by kNN.

    Examples
    --------
    >>> smooth_pred_class = knn_smooth_pred_class(
    ...     X = X,
    ...     pred_class = raw_predicted_classes,
    ...     grouping = louvain_cluster_groups,
    ...     k = 15,)

    Notes
    -----
    scNym classifiers do not incorporate neighborhood information.
    By using a simple kNN smoothing heuristic, we can leverage neighborhood
    information to improve classification performance, smoothing out cells
    that have an outlier prediction relative to their local neighborhood.
    '''
    if grouping is None:
        # do not use a grouping to restrict local neighborhood
        # associations, create a universal pseudogroup `0`.
        grouping = np.zeros(X.shape[0])

    smooth_pred_class = np.zeros_like(pred_class)
    for group in np.unique(grouping):
        # identify only cells in the relevant group
        group_idx = np.where(grouping == group)[0].astype('int')
        X_group = X[grouping == group, :]
        # if there are < k cells in the group, change `k` to the
        # group size
        if X_group.shape[0] < k:
            k_use = X_group.shape[0]
        else:
            k_use = k
        # compute a nearest neighbor graph and identify kNN
        nns = NearestNeighbors(n_neighbors=k_use,).fit(X_group)
        dist, idx = nns.kneighbors(X_group)

        # for each cell in the group, assign a class as
        # the majority class of the kNN
        for i in range(X_group.shape[0]):
            classes = pred_class[group_idx[idx[i, :]]]
            uniq_classes, counts = np.unique(classes, return_counts=True)
            maj_class = uniq_classes[int(np.argmax(counts))]
            smooth_pred_class[group_idx[i]] = maj_class
    return smooth_pred_class



class RBFWeight(object):
    
    def __init__(
        self,
        alpha: float=None,
    ) -> None:
        '''Generate a set of weights based on distances to a point
        with a radial basis function kernel.
        
        Parameters
        ----------
        alpha : float
            radial basis function parameter. inverse of sigma
            for a standard Gaussian pdf.
        
        Returns
        -------
        None.
        '''
        self.alpha = alpha
        return
    
    def set_alpha(
        self,
        X: np.ndarray,
        n_max: int=None,
        dm: np.ndarray=None,
    ) -> None:
        '''Set the alpha parameter of a Gaussian RBF kernel
        as the median distance between points in an array of
        observations.
        
        Parameters
        ----------
        X : np.ndarray
            [N, P] matrix of observations and features.
        n_max : int
            maximum number of observations to use for median
            distance computation.
        dm : np.ndarray, optional
            [N, N] distance matrix for setting the RBF kernel parameter.
            speeds computation if pre-computed.            
        
        Returns
        -------
        None. Sets `self.alpha`.
    
        References
        ----------
        A Kernel Two-Sample Test
        Arthur Gretton, Karsten M. Borgwardt, Malte J. Rasch, 
        Bernhard Schölkopf, Alexander Smola.
        JMLR, 13(Mar):723−773, 2012.
        http://jmlr.csail.mit.edu/papers/v13/gretton12a.html        
        '''
        if n_max is None:
            n_max = X.shape[0]

        if dm is None:
            # compute a distance matrix from observations
            if X.shape[0] > n_max:
                ridx = np.random.choice(
                    X.shape[0], 
                    size=n_max, 
                    replace=False,
                )
                X_p = X[ridx, :]
            else:
                X_p = X

            dm = euclidean_distances(X_p,)
        
        upper = dm[np.triu_indices_from(dm, k=1)]

        # overwrite_input = True saves memory by overwriting
        # the upper indices in the distance matrix array during
        # median computation
        sigma = np.median(
            upper,
            overwrite_input=True,
        )
        self.alpha = 1./(2*(sigma**2))
        return

    def __call__(
        self,
        distances: np.ndarray,
    ) -> np.ndarray:
        '''Generate a set of weights based on distances to a point
        with a radial basis function kernel.

        Parameters
        ----------
        distances : np.ndarray
            [N,] distances used to generate weights.

        Returns
        -------
        weights : np.ndarray
            [N,] weights from the radial basis function kernel.

        Notes
        -----
        We weight distances with a Gaussian RBF.

        .. math::

            f(r) = \exp -(\alpha r)^2

        '''
        # check that alpha parameter is set
        if self.alpha is None:
            msg = 'must set `alpha` attribute before computing weights.\n'
            msg += 'use `.set_alpha() method to estimate from data.'
            raise ValueError(msg)
        
        # generate weights with an RBF kernel
        weights = np.exp(-(self.alpha * distances)**2)
        return weights


def knn_smooth_pred_class_prob(
    X: np.ndarray,
    pred_probs: np.ndarray,
    names: np.ndarray,    
    grouping: np.ndarray = None,
    k: Union[Callable,int] = 15,
    dm: np.ndarray=None,
    **kwargs,
) -> np.ndarray:
    '''
    Smooths class predictions by taking the modal class from each cell's
    nearest neighbors.

    Parameters
    ----------
    X : np.ndarray
        [N, Features] embedding space for calculation of nearest neighbors.
    pred_probs : np.ndarray
        [N, C] array of class prediction probabilities.
    names : np.ndarray,
        [C,] names of predicted classes in `pred_probs`.
    groupings : np.ndarray
        [N,] unique grouping labels for i.e. clusters.
        if provided, only considers nearest neighbors *within the cluster*.
    k : int
        number of nearest neighbors to use for smoothing.
    dm : np.ndarray, optional
        [N, N] distance matrix for setting the RBF kernel parameter.
        speeds computation if pre-computed.

    Returns
    -------
    smooth_pred_class : np.ndarray
        [N,] unique class labels, smoothed by kNN.

    Examples
    --------
    >>> smooth_pred_class = knn_smooth_pred_class_prob(
    ...     X = X,
    ...     pred_probs = predicted_class_probs,
    ...     grouping = louvain_cluster_groups,
    ...     k = 15,)

    Notes
    -----
    scNym classifiers do not incorporate neighborhood information.
    By using a simple kNN smoothing heuristic, we can leverage neighborhood
    information to improve classification performance, smoothing out cells
    that have an outlier prediction relative to their local neighborhood.
    '''
    if grouping is None:
        # do not use a grouping to restrict local neighborhood
        # associations, create a universal pseudogroup `0`.
        grouping = np.zeros(X.shape[0])

    smooth_pred_probs = np.zeros_like(pred_probs)        
    smooth_pred_class = np.zeros(pred_probs.shape[0], dtype='object')
    for group in np.unique(grouping):
        # identify only cells in the relevant group
        group_idx = np.where(grouping == group)[0].astype('int')
        X_group = X[grouping == group, :]
        y_group = pred_probs[grouping == group, :]
        # if k is a Callable, use it to define k for this group
        if callable(k):
            k_use = k(X_group.shape[0])
        else:
            k_use = k
        
        # if there are < k cells in the group, change `k` to the
        # group size
        if X_group.shape[0] < k_use:
            k_use = X_group.shape[0]

        # set up weights using a radial basis function kernel
        rbf = RBFWeight()
        rbf.set_alpha(
            X=X_group,
            n_max=None,
            dm=dm,
        )
        
        if 'dm' in kwargs:
            del kwargs['dm']
        # fit a nearest neighbor regressor
        nns = KNeighborsRegressor(
            n_neighbors=k_use,
            weights=rbf,
            **kwargs,
        ).fit(X_group, y_group)
        smoothed_probs = nns.predict(X_group)

        smooth_pred_probs[group_idx, :] = smoothed_probs
        g_classes = names[np.argmax(smoothed_probs, axis=1)]
        smooth_pred_class[group_idx] = g_classes
        
    return smooth_pred_class


def argmax_pred_class(grouping: np.ndarray,
                      prediction: np.ndarray,
                      ):
    '''Assign class to elements in groups based on the
    most common predicted class for that group.

    Parameters
    ----------
    grouping : np.ndarray
        [N,] partition values defining groups to be classified.
    prediction : np.ndarray
        [N,] predicted values for each element in `grouping`.

    Returns
    -------
    assigned_classes : np.ndarray
        [N,] class labels based on the most common class assigned
        to elements in the group partition.

    Examples
    --------
    >>> grouping = np.array([0,0,0,1,1,1,2,2,2,2])
    >>> prediction = np.array(['A','A','A','B','A','B','C','A','B','C'])
    >>> argmax_pred_class(grouping, prediction)
    np.ndarray(['A','A','A','B','B','B','C','C','C','C',])

    Notes
    -----
    scNym classifiers do not incorporate neighborhood information.
    This simple heuristic leverages cluster information obtained by
    an orthogonal method and assigns all cells in a given cluster
    the majority class label within that cluster.
    '''
    assert grouping.shape[0] == prediction.shape[0], \
        '`grouping` and `prediction` must be the same length'
    groups = sorted(list(set(grouping.tolist())))

    assigned_classes = np.zeros(grouping.shape[0], dtype='object')

    for i, group in enumerate(groups):
        classes, counts = np.unique(prediction[grouping == group],
                                    return_counts=True)
        majority_class = classes[np.argmax(counts)]
        assigned_classes[grouping == group] = majority_class
    return assigned_classes


def compute_entropy_of_mixing(
    X: np.ndarray,
    y: np.ndarray,
    n_neighbors: int,
    n_iters: int=None,
    **kwargs,
) -> np.ndarray:
    '''Compute the entropy of mixing among groups given
    a distance matrix.
    
    Parameters
    ----------
    X : np.ndarray
        [N, P] feature matrix.
    y : np.ndarray
        [N,] group labels.
    n_neighbors : int
        number of nearest neighbors to draw for each iteration 
        of the entropy computation.
    n_iters : int
        number of iterations to perform.
        if `n_iters is None`, uses every point.
        
    Returns
    -------
    entropy_of_mixing : np.ndarray
        [n_iters,] entropy values for each iteration.

    Notes
    -----
    The entropy of batch mixing is computed by sampling `n_per_sample` 
    cells from a local neighborhood in the nearest neighbor graph
    and contructing a probability vector based on their group membership.
    The entropy of this probability vector is computed as a metric of
    intermixing between groups.
    
    If groups are more mixed, the probability vector will have higher
    entropy, and vice-versa.
    '''
    # build nearest neighbor graph
    n_neighbors = min(n_neighbors, X.shape[0])
    nn = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric='euclidean',
        **kwargs,
    )
    nn.fit(X)
    nn_idx = nn.kneighbors(return_distance=False)
    
    # define query points
    if n_iters is not None:
        # don't duplicate points when sampling
        n_iters = min(n_iters, X.shape[0])
    
    if (n_iters is None) or (n_iters == X.shape[0]):
        # sample all points
        query_points = np.arange(X.shape[0])
    else:
        # subset random query points for entropy
        # computation
        assert n_iters < X.shape[0]
        query_points = np.random.choice(
            X.shape[0],
            size=n_iters,
            replace=False,
        )
    
    entropy_of_mixing = np.zeros(len(query_points))
    for i, ridx in enumerate(query_points): 
        # get the nearest neighbors of a point
        nn_y = y[nn_idx[ridx, :]]
        
        nn_y_p = np.zeros(len(np.unique(y)))
        for j, v in enumerate(np.unique(y)):
            nn_y_p[j] = sum(nn_y == v)
        nn_y_p = nn_y_p / nn_y_p.sum()
        
        # use base 2 to return values in bits rather
        # than the default nats
        H = stats.entropy(nn_y_p)
        entropy_of_mixing[i] = H
    return entropy_of_mixing


'''Find new cell state based on scNym confidence scores'''

from sklearn.metrics import calinski_harabasz_score
def _optimize_clustering(adata, resolution: list=[0.1, 0.2, 0.3, 0.5, 1.0]):
    scores = []
    for r in resolution:
        sc.tl.leiden(adata, resolution=r)
        s = calinski_harabasz_score(adata.obsm['X_scnym'], adata.obs['leiden'])
        scores.append(s)
    cl_opt_df = pd.DataFrame({'resolution': resolution, 'score': scores})
    best_idx = np.argmax(cl_opt_df['score'])
    res = cl_opt_df.iloc[best_idx, 0]
    sc.tl.leiden(adata, resolution=res)
    print('Best resolution: ', res)
    return cl_opt_df


def find_low_confidence_cells(
    adata: anndata.AnnData,
    confidence_threshold: float=0.5,
    confidence_key: str='Confidence',
    use_rep: str='X_scnym',
    n_neighbors: int=15,
) -> pd.DataFrame:
    '''Find cells with low confidence predictions and suggest a potential
    number of cell states within the low confidence cell population.
    
    Parameters
    ----------
    adata : anndata.AnnData
        [Cells, Genes] experiment containing an scNym embedding and scNym
        confidence scores.
    confidence_threshold : float
        threshold for low confidence cells.
    confidence_key : str
        key in `adata.obs` containing confidence scores.
    use_rep : str
        tensor in `adata.obsm` containing the scNym embedding.
    n_neighbors : int
        number of nearest neighbors to use for NN graph construction
        prior to community detection.
    
    Returns
    -------
    None.
    Adds `adata.uns["scNym_low_confidence_cells"]`, a `dict` containing
    keys `"cluster_optimization", "n_clusters", "embedding"`.
    Adds key to `adata.obs["scNym_low_confidence_cluster"]`.
    
    Notes
    -----
    '''
    # identify low confidence cells
    adata.obs['scNym Discovery'] = (
        adata.obs[confidence_key] < confidence_threshold
    ).astype(bool)
    low_conf_bidx = adata.obs['scNym Discovery']
    
    # embed low confidence cells
    lc_ad = adata[adata.obs['scNym Discovery'], :].copy()
    sc.pp.neighbors(lc_ad, use_rep=use_rep, n_neighbors=n_neighbors)
    sc.tl.umap(lc_ad, min_dist=0.3)
    
    cl_opt_df = _optimize_clustering(lc_ad)
    
    lc_embed = lc_ad.obs.copy()
    for k in range(1, 3):
        lc_embed[f'UMAP{k}'] = lc_ad.obsm['X_umap'][:, k-1]
    
    # set the outputs
    adata.uns['scNym_low_confidence_cells'] = {
        'cluster_optimization' : cl_opt_df,
        'n_clusters' : len(np.unique(lc_ad.obs['leiden'])),
        'embedding': lc_embed,
    }
    adata.obs['scNym_low_confidence_cluster'] = 'High Confidence'
    adata.obs.loc[
        low_conf_bidx,
        'scNym_low_confidence_cluster',
    ] = lc_ad.obs['leiden'].apply(lambda x : f'Low Confidence {x}')
    return