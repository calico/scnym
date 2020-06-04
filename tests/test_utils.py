import pytest
import os
import os.path as osp
import sys
import numpy as np
import torch
import anndata
import scipy.sparse as sparse

sys.path.append('../')

def test_build_classification_matrix_dense():
    '''Builds a classification matrix from two
    matrices with gene names that are not aligned.
    
    Tests using dense matrices.
    '''
    import scnym
    
    B = np.random.random((100, 10))
    
    # create dummy gene names where the order of 
    # genes in `B` is permuted
    A_genes = np.arange(10)
    B_genes = np.random.permutation(A_genes)
    
    # build the classification matrix
    X = scnym.utils.build_classification_matrix(
        X=B,
        model_genes=A_genes,
        sample_genes=B_genes,
    )
    
    # X should have the genes of B in the order of A
    for i, g in enumerate(A_genes):
        j = int(np.where(B_genes == g)[0])
        assert np.all(X[:, i] == B[:, j])
    return
    

def test_build_classification_matrix_sparse():
    '''Builds a classification matrix from two
    matrices with gene names that are not aligned.
    
    Tests using sparse matrices.
    '''
    import scnym
    
    # generate a sparse matrix with ~10% of elements filled
    B = np.zeros((100, 10))
    ridx = np.random.choice(B.size, size=100, replace=True)
    B.flat[ridx] = 1
    B = sparse.csr_matrix(B)
    
    # create dummy gene names where the order of 
    # genes in `B` is permuted
    A_genes = np.arange(10)
    B_genes = np.random.permutation(A_genes)
    
    # build the classification matrix
    X = scnym.utils.build_classification_matrix(
        X=B,
        model_genes=A_genes,
        sample_genes=B_genes,
    )
    assert sparse.issparse(X)
    
    # X should have the genes of B in the order of A
    for i, g in enumerate(A_genes):
        j = int(np.where(B_genes == g)[0])
        assert np.all(X[:, i].toarray() == B[:, j].toarray())
    return


def test_get_adata_asarray():
    '''Tests getting anndata main matrix as an array in memory'''
    # test getting a dense matrix
    import scnym
    
    adata = anndata.AnnData(
        X = np.random.random((100, 100))
    )
    X = scnym.utils.get_adata_asarray(adata=adata)
    assert type(X) == np.ndarray
    
    # test getting a sparse matrix
    A = np.zeros((100, 100))
    ridx = np.random.choice(A.size, size=1000, replace=True)
    A.flat[ridx] = 1
    A = sparse.csr_matrix(A)
    adata = anndata.AnnData(
        X = A
    )
    X = scnym.utils.get_adata_asarray(adata=adata)
    assert sparse.issparse(X)
    return