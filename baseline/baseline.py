'''Baseline classification models used for comparison to scNym

This module contains all baseline models implemented in python.
'''
import numpy as np
import pandas as pd
import anndata
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV # platt scaling for SVMs
from scipy import stats
# scanpy for `harmony_clf`
import scanpy as sc
import scanpy.external as sce
import scvi # note -- requires `scvi-tools`
import pickle


############################################
# scmap-cell-exact classification scoring
############################################


def scmap_cell_classif(
    train_adata: anndata.AnnData, 
    test_adata: anndata.AnnData, 
    cell_type_col: str,
    n_neighbors: int=10,
    n_features: int=1000,
    return_prob: bool=False,
    **kwargs,
) -> np.ndarray:
    '''Train a kNN classifier with cosine distances per scmap-cell.
    Instead of an approximate kNN search, this version uses the exact
    kNN search provided in `scikit-learn`.
    
    Parameters
    ----------
    train_adata : anndata.AnnData
        [Cells, Genes] for training.
    test_adata : anndata.AnnData
        [Cells, Genes] for testing
    cell_type_col : str
        column in `.obs` containing cell types for `adata` objects.
    n_neighbors : int
        number of nearest neighbors to use.
    n_features : int
        features to select with the `M3Drop` scheme.
        authors recommend 1000.
    return_prob : bool
        return probabilities in addition to predictions, in that
        order.
    
    Returns
    -------
    predictions : np.ndarray
        [Cells,] predicted labels.
    probabilities : pd.DataFrame, optional
        [Cells, Classes] probability matrix. classes are column names.
        
    Notes
    -----
    The top `n_features` genes are selected using the M3Drop method.
    Default values for `n_features` and `n_neighbors` are set based on the
    hyperparameter optimization presented in the original `scmap` paper.
    
    References
    ----------
    scmap: projection of single-cell RNA-seq data across data sets
    Vladimir Yu Kiselev, Andrew Yiu & Martin Hemberg
    https://www.nature.com/articles/nmeth.4644
    '''
    knn = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        metric='cosine',
        n_jobs=-1,
    )
    
    if n_features is None:
        n_features = train_adata.shape[1]
    
    if n_features < train_adata.shape[1]:
        # fit "M3Drop" model to estimate genes with higher than expected dropout
        if type(train_adata.X) != np.ndarray:
            X = train_adata.X.toarray()
        expr = np.expm1(X).mean(0)
        # percentage of dropouts for each gene
        drop = (1 - (X>0).mean(0)) * 100

        expr = np.log2(expr + 1)
        drop = np.log2(drop)

        lm = stats.linregress(expr, drop)
        pred = lm.slope*expr + lm.intercept
        res = drop - pred
        selection_idx = np.argsort(res)[-n_features:]  
        selected_genes = train_adata.var_names[selection_idx]
    else:
        selected_genes = train_adata.var_names
    
    # remove genes not expressed in the test data
    # per the scmap-cell paper
    selected_genes = np.intersect1d(
        np.array(test_adata.var_names),
        selected_genes,
    )
    
    train_adata = train_adata[:, selected_genes].copy()
    test_adata = test_adata[:, selected_genes].copy()
    
    knn.fit(train_adata.X, train_adata.obs[cell_type_col])
    predictions = knn.predict(test_adata.X)
    probabilities = knn.predict_proba(test_adata.X)
    
    probabilities = pd.DataFrame(
        probabilities,
        columns=knn.classes_,
    )
    
    if return_prob:
        r = (predictions, probabilities)
    else:
        r = predictions
    return r


def vanilla_svm(
    train_adata: anndata.AnnData, 
    test_adata: anndata.AnnData, 
    cell_type_col: str,
    return_prob: bool=False,
    **kwargs,
) -> np.ndarray:
    '''Train and predict with a linear support vector machine.
    
    Parameters
    ----------
    train_adata : anndata.AnnData
        [Cells, Genes] for training.
    test_adata : anndata.AnnData
        [Cells, Genes] for testing.
    cell_type_col : str
        column in `.obs` containing cell types for `adata` objects.        
    return_prob : bool
        return probabilities in addition to predictions, in that
        order. SVMs do not have native probablistic predictions,
        so we scale the decision confidence for a pseudoprobability.

    Returns
    -------
    predictions : np.ndarray
        [Cells,] predicted labels for test set.
        
    Notes
    -----
    A linear support vector machine (SVM) was shown to be the top performing
    baseline in a benchmarking study (see References).
    We set the default SVM hyperparameter values based on this 
    benchmarking paper.
    
    SVMs do not offer probabilistic outputs natively. To approximate a probablistic
    output, we use the Platt Scaling method.
    
    References
    ----------
    A comparison of automatic cell identification methods for \
    single-cell RNA sequencing data.
    Abdelaal, T. et al. 
    Genome Biology 20, 194 (2019).
    
    Probabilistic Outputs for Support Vector Machines and Comparisons \
    to Regularized Likelihood Methods. 
    Platt, J. C.
    Advances in Large Margin Classifiers 61–74 (MIT Press, 1999).
    '''
    model = LinearSVC(**kwargs)
    model.fit(
        train_adata.X,
        train_adata.obs[cell_type_col],
    )
    
    predictions = model.predict(
        test_adata.X,
    )
    
    # SVMs do not have native probabilistic prediction
    # Rather, we scale the model confidence scores 
    # derived from the distance to the support vectors
    # to generate a pseudoprobability with Platt scaling
    #
    # See the relevant `sklearn` documentation
    # https://bit.ly/2FJX21r
    
    # use CalibratedClassifierCV to perform platt scaling
    calib = CalibratedClassifierCV(
        model,
        method='sigmoid', # corresponds to Platt Scaling
        cv=5,
    )
    calib.fit(
        train_adata.X, 
        train_adata.obs[cell_type_col],
    )
    
    # use decision_function
    probabilities = calib.predict_proba(test_adata.X)
    
    probabilities = pd.DataFrame(
        probabilities,
        columns=calib.classes_,
    )
    
    if return_prob:
        r = (predictions, probabilities)
    else:
        r = predictions
    
    return r


def harmony_clf(
    train_adata: anndata.AnnData, 
    test_adata: anndata.AnnData, 
    cell_type_col: str,
    return_prob: bool=False,
    **kwargs,    
) -> (np.ndarray, anndata.AnnData):
    '''Perform `harmony` data integration, followed by SVM classification.
    
    Parameters
    ----------
    train_adata : anndata.AnnData
        [Cells, Genes] for training.
    test_adata : anndata.AnnData
        [Cells, Genes] for testing.
    cell_type_col : str
        column in `.obs` containing cell types for `adata` objects.        
    return_prob : bool
        return probabilities in addition to predictions, in that
        order. SVMs do not have native probablistic predictions,
        so we scale the decision confidence for a pseudoprobability.        
        
    Returns
    -------
    predictions : np.ndarray
        [Cells,] predicted labels for test set.
    probabiliteis : np.ndarray, optional.
        [Cells, Classes]
    joint : anndata.AnnData
        [Cells, Genes] of joined training and test data.
        `.obs['batch']` specifies train (`'0'`) or test (`'1'`) data.
        `.obsm['X_pca_harmony']` contains the harmonized embedding.
        
    Notes
    -----
    Data are first integrated using the `harmonypy` implementation of 
    the Harmony algorithm. An SVM is then trained on the integrated latent
    variables.
    
    See Also
    --------
    vanilla_svm
    '''
    # `.concatenate` automatically adds a batch variable
    joint = train_adata.concatenate(test_adata)
    
    # fit unharmonized PCA
    sc.pp.highly_variable_genes(joint, n_top_genes=4000)
    sc.pp.pca(joint)
    
    # creates "harmonized" PCA embedding
    # `joint.obsm['X_pca_harmony']`
    sce.pp.harmony_integrate(
        joint,
        key='batch',
    )
    
    # train an SVM
    X_train = joint.obsm['X_pca_harmony'][joint.obs['batch']=='0']
    y_train = np.array(joint.obs[cell_type_col])[joint.obs['batch']=='0']
    X_test  = joint.obsm['X_pca_harmony'][joint.obs['batch']=='1']
    y_test  = np.array(joint.obs[cell_type_col])[joint.obs['batch']=='1']
    
    model = LinearSVC(**kwargs)
    model.fit(
        X_train,
        y_train,
    )
    
    predictions = model.predict(
        X_test,
    )
    
    # use CalibratedClassifierCV to perform platt scaling
    calib = CalibratedClassifierCV(
        model,
        method='sigmoid', # corresponds to Platt Scaling
        cv=5,
    )
    calib.fit(
        train_adata.X, 
        train_adata.obs[cell_type_col],
    )
    
    # use decision_function
    probabilities = calib.predict_proba(test_adata.X)
    
    probabilities = pd.DataFrame(
        probabilities,
        columns=calib.classes_,
    )
    
    if return_prob:
        r = (predictions, probabilities, joint)
    else:
        r = (predictions, joint)
    
    return r


def isinteger(x) -> bool:
    '''Check if values in `x` are integers, regardless of
    dtype. Returns elementwise `bool`.
    '''
    # check if X is sparse
    if type(x) == np.ndarray:
        return np.equal(np.mod(x, 1), 0)
    else:
        return np.all(np.equal(np.mod(x.data, 1), 0))


def scanvi(
    train_adata: anndata.AnnData,
    test_adata: anndata.AnnData,
    cell_type_col: str,
    n_per_class: int=100,
    **kwargs,
) -> (np.ndarray, pd.DataFrame, anndata.AnnData, scvi.model.scanvi.SCANVI):
    '''Use SCANVI to transfer annotations.
    
    Parameters
    ----------
    train_adata : anndata.AnnData
        [Cells, Genes] for training.
    test_adata : anndata.AnnData
        [Cells, Genes] for testing.
    cell_type_col : str
        column labeling ground truth cell types in
        `train_adata` and `test_adata`.
    n_per_class : int
        number of training examples per class. scANVI authors
        recommend `100` (default).
    
    Returns
    -------
    predictions : np.ndarray
        [Cells,] cell type label predictions.
    probabilities : pd.DataFrame
        [Cells, Class] probabilities.
        classes are column labels.
    adata : anndata.AnnData
        [Cells, Genes] concatenation of `train_adata` and `test_adata`
        with the learned scANVI latent space in `.obsm['X_scANVI']`.
        class predictions are in `.obs['C_scANVI']`.
    lvae : scvi.model.scanvi.SCANVI
        a trained scANVI model object.
    
    Notes
    -----
    This implementation exactly follows the working example
    from the `scvi` authors.
    
    https://www.scvi-tools.org/en/0.7.0-alpha.4/user_guide/notebooks/harmonization.html
    '''
    # check that train_adata and test_adata contain
    # raw counts
    tr_int = isinteger(train_adata.X)
    te_int = isinteger(test_adata.X)
    if not (tr_int and te_int):
        # check if the raw counts are in `.raw`
        tr_int = isinteger(train_adata.raw.X)
        te_int = isinteger(test_adata.raw.X)
        if tr_int and te_int:
            train_adata = train_adata.copy()
            test_adata = test_adata.copy()
            # set raw counts to `X`
            train_adata.X = train_adata.raw[:, train_adata.var_names].X
            test_adata.X = test_adata.raw[:, test_adata.var_names].X
        else:
            msg = 'Integer raw counts not found.'
            raise ValueError(msg)
    else:
        # raw counts are already set to X
        pass
    
    # `.concatenate()` creates batch labels in `.obs['batch']`
    adata = train_adata.concatenate(test_adata)
    
    # store raw counts in a new layer
    # normalize and select highly variable genes
    #
    # scVI uses only a set of highly variable genes
    # to perform data integration in the latent space
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata  # keep full dimension safe
    sc.pp.highly_variable_genes(
        adata, 
        flavor="seurat_v3", 
        n_top_genes=2000, 
        layer="counts", 
        batch_key="batch",
        subset=True
    )
    
    # assign cell type labels for training
    # scANVI uses a special token `"Unknown"` for cells that are not labeled
    # the authors recommend using 100 cells per cell type
    # from the training set to balance classes
    adata.obs[cell_type_col] = pd.Categorical(
        adata.obs[cell_type_col],
    )
    
    labels = np.repeat("Unknown", adata.shape[0])
    labels = labels.astype("<U43")
    for x in np.unique(adata.obs[cell_type_col]):
        idx = np.where((adata.obs[cell_type_col] == x) & (adata.obs["batch"] == "0"))[
            0
        ]
        sampled = np.random.choice(
            idx, 
            np.min([n_per_class, len(idx)])
        )
        labels[sampled] = adata.obs[cell_type_col][sampled]

    adata.obs["celltype_scanvi"] = labels
    
    # setup scANVI for training
    scvi.data.setup_anndata(
        adata, 
        layer="counts",
        batch_key="batch", 
        labels_key="celltype_scanvi", 
    )
    
    # fit the semi-supervised scANVI model
    lvae = scvi.model.SCANVI(
        adata,
        "Unknown",
        use_cuda=True,
        n_latent=30,
        n_layers=2,
    )
    
    lvae.train(n_epochs_semisupervised=100)
    
    # extract labels
    adata.obs["C_scANVI"] = lvae.predict(adata)
    adata.obsm["X_scANVI"] = lvae.get_latent_representation(adata)
    
    predictions = np.array(adata.obs.loc[adata.obs['batch']=='1', "C_scANVI"])
    # returns a [Cells, Classes] data frame with class
    # names as column labels and cell barcodes as indices
    probabilities = lvae.predict(adata, soft=True)
    probabilities = probabilities.loc[adata.obs['batch']=='1']
    # scANVI will add the "Unknown" token as a class, usually
    # with very low probability
    # here we drop it, then renorm probabilities to == 1
    probabilities = probabilities.drop(columns=['Unknown'])
    probabilities = probabilities / np.tile(
        np.array(probabilities.sum(1)).reshape(-1, 1), (1, probabilities.shape[1])
    )
    # check probability normalization
    eq1 = np.allclose(
        probabilities.sum(1),
        np.ones(probabilities.shape[0]),
    )
    if not eq1:
        msg = 'Not all sum(probabilities) are close to 1.'
        n = np.sum(probabilities.sum(1)!=1.)
        msg += f'{n} cells have probabilities != 1.'
        raise ValueError(msg)

    
    r = (
        predictions, 
        probabilities, 
        adata, 
        lvae,
    )
    
    return r
