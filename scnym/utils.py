'''
Utility functions
'''
import torch
import numpy as np
import pandas as pd
import tqdm
from sklearn.neighbors import NearestNeighbors


def make_one_hot(labels: torch.LongTensor, C=2) -> torch.FloatTensor:
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


def l1_layer0(model: torch.nn.Module) -> torch.FloatTensor:
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


def append_categorical_to_data(X: np.ndarray,
                               categorical: np.ndarray,
                               ) -> (np.ndarray, np.ndarray):
    '''Convert `categorical` to a one-hot vector and append
    this vector to each sample in `X`.

    Parameters
    ----------
    X : np.ndarray
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

    one_hot_mat = make_one_hot(idx,
                               C=len(categories))
    one_hot_mat = one_hot_mat.numpy()
    assert X.shape[0] == one_hot_mat.shape[0], \
        'dims unequal at %d, %d' % (X.shape[0], one_hot_mat.shape[0])
    # append one hot vector to the [Cells, Features] matrix
    X = np.concatenate([X, one_hot_mat], axis=1)
    return X, categories


def build_classification_matrix(X: np.ndarray,
                                model_genes: np.ndarray,
                                sample_genes: np.ndarray,) -> np.ndarray:
    '''
    Build a matrix for classification using only genes that overlap
    between the current sample and the pre-trained model.

    Parameters
    ----------
    X : np.ndarray
        [Cells, Genes] count matrix.
    model_genes : np.ndarray
        gene identifiers in the order expected by the model.
    sample_genes : np.ndarray
        gene identifiers for the current sample.

    Returns
    -------
    N : np.ndarray
        [Cells, len(model_genes)] count matrix.
        Values where a model gene was not present in the sample are left
        as zeros.
    '''
    n_cells = X.shape[0]
    # instantiate a new [Cells, model_genes] matrix where columns
    # retain the order used during training
    N = np.zeros((n_cells, len(model_genes)))

    # populate `N` using only genes that are shared between the trained
    # model and the sample matrix. leave other columns as `0`.
    common_genes = 0
    for i, g in tqdm.tqdm(enumerate(model_genes), 'Building matrix'):
        if g in sample_genes:
            j = np.where(sample_genes == g)[0].astype('int')
            N[:, i] = X[:, j].squeeze()
            common_genes += 1
    print('Found %d common genes.' % common_genes)
    return N


def knn_smooth_pred_class(X: np.ndarray,
                          pred_class: np.ndarray,
                          grouping: np.ndarray = None,
                          k: int = 15,) -> np.ndarray:
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
