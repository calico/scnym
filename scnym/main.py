'''Train scNym models and identify cell type markers'''
import numpy as np
import pandas as pd
import os
import os.path as osp
import scanpy.api as sc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from typing import Union
import copy

from .model import CellTypeCLF, CellTypeCLFConditional
from .dataprep import SingleCellDS, balance_classes
from .trainer import Trainer
from .predict import Predicter
from . import utils

#########################################################
# Train scNym classification models
#########################################################


def train_cv(X: np.ndarray,
             y: np.ndarray,
             batch_size: int,
             n_epochs: int,
             weight_decay: float,
             ModelClass: nn.Module,
             fold_indices: list,
             out_path: str,
             n_genes: int = None,
             **kwargs) -> (np.ndarray, np.ndarray):
    '''Perform training using a provided set of training/hold-out
    sample indices.

    Parameters
    ----------
    X : np.ndarray
        [Cells, Genes] of log1p transformed normalized values.
        log1p and normalization performed using scanpy defaults.
    y : np.ndarray
        [Cells,] integer class labels.
    n_epochs : int
        number of epochs for training.
    ModelClass : nn.Module
        a model class for construction classification models.
    batch_size : int
        batch size for training.        
    fold_indices : list
        elements are 2-tuple, with training indices and held-out.
    out_path : str
        top level path for saving fold outputs.
    n_genes : int
        number of genes in the input. Not necessarily `X.shape[1]` if
        the input matrix has been concatenated with other features.

    Returns
    -------
    fold_eval_acc : np.ndarray
        evaluation accuracies for each fold.
    fold_eval_losses : np.ndarray
        loss values for each fold.
    '''
    n_cell_types = len(np.unique(y))
    if n_genes is None:
        n_genes = X.shape[1]

    fold_eval_losses = np.zeros(len(fold_indices))
    fold_eval_acc = np.zeros(len(fold_indices))

    # Perform training on each fold specified in `fold_indices`
    for f in range(len(fold_indices)):
        print('Training tissue independent, fold %d.' % f)
        fold_out_path = osp.join(out_path, 'fold' + str(f).zfill(2))

        os.makedirs(fold_out_path, exist_ok=True)

        traintest_idx = fold_indices[f][0].astype('int')
        val_idx = fold_indices[f][1].astype('int')

        # Set aside 10% of the traintest data for model selection in `test_idx`
        train_idx = np.random.choice(traintest_idx,
                                     size=int(
                                         np.floor(0.9 * len(traintest_idx))),
                                     replace=False).astype('int')
        test_idx = np.setdiff1d(traintest_idx, train_idx).astype('int')
        # save indices to CSVs for later retrieval
        np.savetxt(osp.join(fold_out_path, 'train_idx.csv'), train_idx)
        np.savetxt(osp.join(fold_out_path, 'test_idx.csv'), test_idx)
        np.savetxt(osp.join(fold_out_path, 'val_idx.csv'), val_idx)

        # Generate training and model selection Datasets and Dataloaders
        X_train = X[train_idx, :]
        y_train = y[train_idx]

        X_test = X[test_idx, :]
        y_test = y[test_idx]

        train_ds = SingleCellDS(X_train, y_train,)
        test_ds = SingleCellDS(X_test, y_test,)

        train_dl = DataLoader(train_ds,
                              batch_size=batch_size,
                              shuffle=True,)
        test_dl = DataLoader(test_ds,
                             batch_size=batch_size,
                             shuffle=True,)

        dataloaders = {'train': train_dl,
                       'val': test_dl, }

        # Build a cell type classification model and transfer to CUDA
        model = ModelClass(n_genes=n_genes,
                           n_cell_types=n_cell_types,
                           **kwargs)
        if torch.cuda.is_available():
            model = model.cuda()

        # Set up loss criterion and the model optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adadelta(model.parameters(),
                                         weight_decay=weight_decay)

        print('Training...')
        T = Trainer(model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    dataloaders=dataloaders,
                    out_path=fold_out_path,
                    n_epochs=n_epochs,
                    reg_criterion=None,
                    exp_name='fold' + str(f).zfill(2),
                    verbose=False,)
        T.train()
        print('Training complete.')
        print()

        # Perform model evaluation using the best set of weights on the
        # totally unseen, held out data.
        print('Evaluating tissue independent, fold %d.' % f)
        model = ModelClass(n_genes=n_genes,
                           n_cell_types=n_cell_types,
                           **kwargs,)
        model.load_state_dict(
            torch.load(osp.join(fold_out_path, '00_best_model_weights.pkl'))
        )
        model.eval()

        if torch.cuda.is_available():
            model = model.cuda()

        # Build a DataLoader for validation
        X_val = X[val_idx, :]
        y_val = y[val_idx]
        val_ds = SingleCellDS(X_val, y_val,)
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,)

        # Without recording any gradients to speed things up,
        # predict classes for all held out data and evaluate metrics.
        with torch.no_grad():
            loss = 0.
            running_corrects = 0.
            running_total = 0.
            all_predictions = []
            all_labels = []
            for data in val_dl:
                input_ = data['input']
                label_ = data['output']

                if torch.cuda.is_available():
                    input_ = input_.cuda()
                    label_ = label_.cuda()

                # Perform forward pass and compute predictions as the
                # most likely class
                output = model(input_)
                _, predictions = torch.max(output, 1)
                corrects = torch.sum(predictions.detach() == label_.detach())

                l = criterion(output, label_)
                loss += float(l.detach().cpu().numpy())

                running_corrects += float(corrects.item())
                running_total += float(label_.size(0))

                all_labels.append(label_.detach().cpu().numpy())
                all_predictions.append(predictions.detach().cpu().numpy())

            norm_loss = loss / len(val_dl)
            print('EVAL LOSS: ', norm_loss)
            print('EVAL ACC : ', running_corrects/running_total)
        fold_eval_acc[f] = running_corrects/running_total
        fold_eval_losses[f] = norm_loss

        all_predictions = np.concatenate(all_predictions)
        all_labels = np.concatenate(all_labels)
        np.savetxt(
            osp.join(fold_out_path, 'predictions.csv'),
            all_predictions)
        np.savetxt(
            osp.join(fold_out_path, 'labels.csv'),
            all_labels)

        PL = np.stack([all_predictions, all_labels], 0)
        print('Predictions | Labels')
        print(PL.T[:15, :])
    return fold_eval_acc, fold_eval_losses


def train_all(X: np.ndarray,
              y: np.ndarray,
              batch_size: int,
              n_epochs: int,
              weight_decay: float,
              ModelClass: nn.Module,
              out_path: str,
              n_genes: int = None,
              **kwargs) -> (float, float):
    '''Perform training using all provided samples.

    Parameters
    ----------
    X : np.ndarray
        [Cells, Genes] of log1p transformed normalized values.
        log1p and normalization performed using scanpy defaults.
    y : np.ndarray
        [Cells,] integer class labels.
    n_epochs : int
        number of epochs for training.
    ModelClass : nn.Module
        a model class for construction classification models.
    batch_size : int
        batch size for training.
    out_path : str
        top level path for saving fold outputs.
    n_genes : int
        number of genes in the input. Not necessarily `X.shape[1]` if
        the input matrix has been concatenated with other features.

    Returns
    -------
    test_loss : float
        best loss on the testing set used for model selection.
    test_acc : float
        best accuracy on the testing set used for model selection.
    '''
    n_cell_types = len(np.unique(y))
    if n_genes is None:
        n_genes = X.shape[1]

    # Prepare a unique output directory
    all_out_path = osp.join(out_path, 'all_data')
    os.makedirs(all_out_path, exist_ok=True)

    # Generate training and model selection indices
    train_idx = np.random.choice(
        np.arange(X.shape[0]),
        size=int(np.floor(0.9*X.shape[0])),
        replace=False).astype('int')
    test_idx = np.setdiff1d(np.arange(X.shape[0]), train_idx).astype('int')

    # Generate training and model selection Datasets and Dataloaders
    X_train = X[train_idx, :]
    y_train = y[train_idx]

    X_test = X[test_idx, :]
    y_test = y[test_idx]

    train_ds = SingleCellDS(X_train, y_train,)
    test_ds = SingleCellDS(X_test, y_test,)

    train_dl = DataLoader(train_ds,
                          batch_size=batch_size,
                          shuffle=True,)
    test_dl = DataLoader(test_ds,
                         batch_size=batch_size,
                         shuffle=True,)

    dataloaders = {'train': train_dl,
                   'val': test_dl, }

    # Build a cell type classification model and transfer to CUDA
    model = ModelClass(n_genes=n_genes,
                       n_cell_types=n_cell_types,
                       **kwargs)
    if torch.cuda.is_available():
        model = model.cuda()

    # Set up loss criterion and the model optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adadelta(model.parameters(),
                                     weight_decay=weight_decay)

    # Fit the model
    print('Training...')
    T = Trainer(model=model,
                criterion=criterion,
                optimizer=optimizer,
                dataloaders=dataloaders,
                out_path=all_out_path,
                n_epochs=n_epochs,
                reg_criterion=None,
                exp_name='all_data',
                verbose=False,)
    T.train()
    print('Training complete.')
    print()

    # Perform model evaluation on the validation set
    # This is the best we can do when fitting to all the data.
    model = ModelClass(n_genes=n_genes,
                       n_cell_types=n_cell_types,
                       **kwargs,)
    model.load_state_dict(
        torch.load(osp.join(all_out_path, '00_best_model_weights.pkl'))
    )
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    with torch.no_grad():
        loss = 0.
        running_corrects = 0.
        running_total = 0.
        all_predictions = []
        all_labels = []
        for data in test_dl:
            input_ = data['input']
            label_ = data['output']

            if torch.cuda.is_available():
                input_ = input_.cuda()
                label_ = label_.cuda()

            # Perform forward pass and compute predictions as the
            # most likely class
            output = model(input_)
            _, predictions = torch.max(output, 1)
            corrects = torch.sum(predictions.detach() == label_.detach())

            l = criterion(output, label_)
            loss += float(l.detach().cpu().numpy())

            running_corrects += float(corrects.item())
            running_total += float(label_.size(0))

            all_labels.append(label_.detach().cpu().numpy())
            all_predictions.append(predictions.detach().cpu().numpy())

        test_loss = loss / len(test_dl)
        test_acc = running_corrects/running_total
        print('FINAL EVAL LOSS: ', test_loss)
        print('FINAL EVAL ACC : ', test_acc)

    np.savetxt(
        osp.join(all_out_path, 'test_loss_acc.csv'),
        np.array([test_loss, test_acc]).reshape(2, 1),
        delimiter=',')

    return test_loss, test_acc


def train_tissue_independent_cv(X: np.ndarray,
                                metadata: pd.DataFrame,
                                out_path: str,
                                balanced_classes: bool = False,
                                batch_size: int = 256,
                                n_epochs: int = 200,
                                lower_group: str = 'cell_ontology_class',
                                **kwargs,):
    '''
    Trains a cell type classifier that is independent of tissue origin

    Parameters
    ----------
    X : np.ndarray
        [Cells, Genes] of log1p transformed normalized values.
        log1p and normalization performed using scanpy defaults.
    metadata : pd.DataFrame
        [Cells, Features] data with `upper_group` and `lower_group` columns.
    out_path : str
        path for saving trained model weights and evaluation performance.
    balanced_classes : bool, optional
        perform class balancing by undersampling majority classes.
    batch_size : int
        batch size for training.
    n_epochs : int
        number of epochs for training.
    lower_group : str
        column in `metadata` corresponding to output classes. i.e. cell types.

    Returns
    -------
    None.

    Notes
    -----
    Passes `kwargs` to `CellTypeCLF`.
    '''

    print('TRAINING TISSUE INDEPENDENT CLASSIFIER')
    print('-'*20)
    print()

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    # identify all the `lower_group` levels and create
    # an integer class vector corresponding to unique levels
    celltypes = sorted(list(set(metadata[lower_group])))
    print('There are %d %s in the experiment.\n' %
          (len(celltypes), lower_group))
    for t in celltypes:
        print(t)
    y = pd.Categorical(metadata[lower_group]).codes
    y = y.astype('int32')
    labels = pd.Categorical(metadata[lower_group]).categories
    # save mapping of levels : integer values as a CSV
    out_df = pd.DataFrame({'label': labels, 'code': np.arange(len(labels))})
    out_df.to_csv(osp.join(out_path, 'celltype_label.csv'))

    if balanced_classes:
        print('Performing class balancing by undersampling...')
        balanced_idx = balance_classes(y)
        X = X[balanced_idx, :]
        y = y[balanced_idx]
        print('Class balancing complete.')

    # generate k-fold cross-validation split indices
    # & vectors for metrics evaluated at each fold.
    kf = StratifiedKFold(n_splits=5, shuffle=True)
    kf_indices = list(kf.split(X, y))

    # Perform training on each fold specified in `kf_indices`
    fold_eval_acc, fold_eval_losses = train_cv(
        X=X,
        y=y,
        batch_size=batch_size,
        n_epochs=n_epochs,
        ModelClass=CellTypeCLF,
        fold_indices=kf_indices,
        out_path=out_path,
        **kwargs)

    # Save the per-fold results to CSVs
    print('Fold eval losses')
    print(fold_eval_losses)
    print('Fold eval accuracy')
    print(fold_eval_acc)
    print('Mean %f Std %f' % (fold_eval_losses.mean(), fold_eval_losses.std()))
    np.savetxt(osp.join(out_path, 'fold_eval_losses.csv',), fold_eval_losses)
    np.savetxt(osp.join(out_path, 'fold_eval_acc.csv',), fold_eval_acc)

    # Train a model using all available data (after class balancing)
    val_loss, val_acc = train_all(
        X=X,
        y=y,
        batch_size=batch_size,
        n_epochs=n_epochs,
        ModelClass=CellTypeCLF,
        out_path=out_path,
        **kwargs)

    return


def train_tissue_dependent_cv(X: np.ndarray,
                              metadata: pd.DataFrame,
                              out_path: str,
                              balanced_classes: bool = False,
                              batch_size: int = 256,
                              n_epochs: int = 200,
                              upper_group: str = 'tissue',
                              lower_group: str = 'cell_ontology_class',
                              **kwargs,) -> None:
    '''
    Trains a cell type classifier conditioned on tissue of origin.

    Parameters
    ----------
    X : np.ndarray
        [Cells, Genes] of log1p transformed, normalized values.
        log1p and normalization performed using scanpy defaults.
    metadata : pd.DataFrame
        [Cells, Features] data with 'tissue' and 'cell_ontology_class' columns.
    out_path : str
        path for saving trained model weights and evaluation performance.
    balanced_classes : bool, optional
        perform class balancing by undersampling majority classes.
    batch_size : int
        batch size for training.
    upper_group : str
        column in `metadata` with subsets for training `lower_group`
        classifiers independently. i.e. tissues.
    lower_group : str
        column in `metadata` corresponding to output classes. i.e. cell types.

    Returns
    -------
    None.

    Notes
    -----
    Appends a one-hot vector indicating the tissue of origin to each input.
    Length of the one-hot vector is equal to `len(set(metadata[upper_group]))`.
    '''

    print('TRAINING TISSUE DEPENDENT CLASSIFIER')
    print('-'*20)
    print()

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    celltypes = sorted(list(set(metadata[lower_group])))
    tissues = sorted(list(set(metadata[upper_group])))

    n_tissues = len(tissues)
    n_genes = X.shape[1]

    print('There are %d %s in the experiment.' % (len(celltypes), lower_group))
    print('There are %d tissues in the experiment.\n' % len(tissues))
    print()
    print('%s levels:' % lower_group)
    for t in celltypes:
        print(t)

    # identify all the `lower_group` levels and create
    # an integer class vector corresponding to unique levels
    y = pd.Categorical(metadata[lower_group]).codes
    y = y.astype('int32')
    labels = pd.Categorical(metadata[lower_group]).categories
    out_df = pd.DataFrame({'label': labels, 'code': np.arange(len(labels))})
    out_df.to_csv(osp.join(out_path, 'celltype_label.csv'))

    # Make `upper_group` one-hot matrix
    tissue_labels = pd.Categorical(metadata[upper_group])
    tissue_idx = np.array(tissue_labels.codes)
    tissue_idx = torch.from_numpy(tissue_idx.astype('int32')).long()
    tissue_categories = np.array(tissue_labels.categories)

    np.savetxt(osp.join(out_path, 'tissue_categories.csv'),
               tissue_categories,
               delimiter=',', fmt='%s')

    one_hot_mat = utils.make_one_hot(tissue_idx, C=len(tissues))
    one_hot_mat = one_hot_mat.numpy()
    assert X.shape[0] == one_hot_mat.shape[0], \
        'dims unequal at %d, %d' % (X.shape[0], one_hot_mat.shape[0])

    # append `upper_group` one hot vector
    # to the [Cells, Genes] matrix.
    print('Input [Cells, Genes] array shape: ', X.shape)
    X = np.concatenate([X, one_hot_mat], axis=1)
    print('One hot matrix appended to [Cells, Genes] array.')
    print('New dimensions: ', X.shape)

    if balanced_classes:
        print('Performing class balancing by undersampling...')
        balanced_idx = balance_classes(y)
        X = X[balanced_idx, :]
        y = y[balanced_idx]
        print('Class balancing complete.')

    # generate k-fold cross-validation split indices
    # & vectors for metrics evaluated at each fold.
    kf = StratifiedKFold(n_splits=5, shuffle=True)
    kf_indices = list(kf.split(X, y))

    # Perform training on each fold specified in `kf_indices`
    fold_eval_acc, fold_eval_losses = train_cv(
        X=X,
        y=y,
        batch_size=batch_size,
        n_epochs=n_epochs,
        n_genes=n_genes,    # set n_genes and n_tissues for Conditional model
        n_tissues=n_tissues,
        ModelClass=CellTypeCLFConditional,
        fold_indices=kf_indices,
        out_path=out_path,
        **kwargs)

    print('Fold eval losses')
    print(fold_eval_losses)
    print('Fold eval accuracy')
    print(fold_eval_acc)
    print('Mean %f Std %f' % (fold_eval_losses.mean(), fold_eval_losses.std()))
    np.savetxt(osp.join(out_path, 'fold_eval_losses.csv',), fold_eval_losses)
    np.savetxt(osp.join(out_path, 'fold_eval_acc.csv',), fold_eval_acc)

    # Train a model using all available data (after class balancing)
    val_loss, val_acc = train_all(
        X=X,
        y=y,
        batch_size=batch_size,
        n_epochs=n_epochs,
        n_genes=n_genes,    # set n_genes and n_tissues for Conditional model
        n_tissues=n_tissues,
        ModelClass=CellTypeCLFConditional,
        out_path=out_path,
        **kwargs)
    return


def train_one_tissue_cv(X: np.ndarray,
                        metadata: pd.DataFrame,
                        out_path: str,
                        balanced_classes: bool = False,
                        batch_size: int = 256,
                        n_epochs: int = 200,
                        upper_group: str = 'tissue',
                        lower_group: str = 'cell_ontology_class',
                        **kwargs,) -> None:
    '''
    Trains a cell type classifier for a single tissue

    Parameters
    ----------
    X : np.ndarray
        [Cells, Genes] of log1p transformed, normalized values.
        log1p and normalization performed using scanpy defaults.
    metadata : pd.DataFrame
        [Cells, Features] data with `upper_group` and `lower_group` columns.
    out_path : str
        path for saving trained model weights and evaluation performance.
    balanced_classes : bool, optional
        perform class balancing by undersampling majority classes.
    upper_group : str
        column in `metadata` with subsets for training `lower_group`
        classifiers independently. i.e. tissues.
    lower_group : str
        column in `metadata` corresponding to output classes. i.e. cell types.

    Returns
    -------
    None.
    '''

    tissue_str = str(list(metadata[upper_group])[0]).lower()
    print('TRAINING %s DEPENDENT CLASSIFIER FOR: ' %
          upper_group.upper(), tissue_str.upper())
    print('-'*20)
    print()

    celltypes = sorted(list(set(metadata[lower_group])))
    print('There are %d %s in the experiment.\n' %
          (len(celltypes), lower_group))
    for t in celltypes:
        print(t)
    print('')
    y = pd.Categorical(metadata[lower_group]).codes
    y = y.astype('int32')
    labels = pd.Categorical(metadata[lower_group]).categories
    out_df = pd.DataFrame({'label': labels, 'code': np.arange(len(labels))})
    out_df.to_csv(osp.join(out_path, 'celltype_label.csv'))

    if balanced_classes:
        print('Performing class balancing by undersampling...')
        balance_idx = balance_classes(y,)
        X = X[balance_idx, :]
        y = y[balance_idx]
        print('Class balancing complete.')

    kf = StratifiedKFold(n_splits=5, shuffle=True)
    kf_indices = list(kf.split(X, y))

    # Perform training on each fold specified in `kf_indices`
    fold_eval_acc, fold_eval_losses = train_cv(
        X=X,
        y=y,
        batch_size=batch_size,
        n_epochs=n_epochs,
        ModelClass=CellTypeCLF,
        fold_indices=kf_indices,
        out_path=out_path,
        **kwargs)

    print('Fold eval losses')
    print(fold_eval_losses)
    print('Fold eval accuracy')
    print(fold_eval_acc)
    print('Mean %f Std %f' % (fold_eval_losses.mean(), fold_eval_losses.std()))
    np.savetxt(osp.join(out_path, 'fold_eval_losses.csv',), fold_eval_losses)
    np.savetxt(osp.join(out_path, 'fold_eval_acc.csv',), fold_eval_acc)

    # Train a model using all available data (after class balancing)
    val_loss, val_acc = train_all(
        X=X,
        y=y,
        batch_size=batch_size,
        n_epochs=n_epochs,
        ModelClass=CellTypeCLF,
        out_path=out_path,
        **kwargs)
    return


def train_tissue_specific_cv(X: np.ndarray,
                             metadata: pd.DataFrame,
                             out_path: str,
                             balanced_classes: dict = None,
                             upper_group: str = 'tissue',
                             lower_group: str = 'cell_ontology_class',
                             **kwargs,) -> None:
    '''
    Train tissue dependent cell type classifiers for each
    tissue in the experiment

    Parameters
    ----------
    X : np.ndarray
        [Cells, Genes] of log1p transformed, normalized values.
        log1p and normalization performed using scanpy defaults.
    metadata : pd.DataFrame
        [Cells, Features] data with 'tissue' and 'cell_ontology_class' columns.
    out_path : str
        path for saving trained model weights and evaluation performance.
    balanced_classes : dict
        keyed by tissue name.
        values are booleans specifying whether to perform class
        balancing for a given tissue.
    upper_group : str
        column in `metadata` with subsets for training `lower_group`
        classifiers independently. i.e. tissues.
    lower_group : str
        column in `metadata` corresponding to output classes. i.e. cell types.
    '''
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    # Identify all the `upper_group` levels present in the data
    tissues = sorted(list(set(metadata[upper_group])))
    print('There are %d %s in the experiment.' % (len(tissues), upper_group))
    for t in tissues:
        print(t)

    # For each `upper_group` level, train a classification model
    for i, t in enumerate(tissues):
        tissue_bidx = metadata[upper_group] == t

        tissue_meta = metadata.loc[tissue_bidx, :]
        tissue_X = X[tissue_bidx, :]

        tissue_out_path = osp.join(out_path, str(t))
        if not os.path.exists(tissue_out_path):
            os.mkdir(tissue_out_path)

        if balanced_classes is None:
            bc = False
        else:
            # determine if we should balance classes for this tissue
            bc = balanced_classes.get(t, False)

        train_one_tissue_cv(tissue_X,
                            tissue_meta,
                            tissue_out_path,
                            balanced_classes=bc,
                            upper_group=upper_group,
                            lower_group=lower_group,
                            **kwargs)

    return

#########################################################
# Identify cell type markers
#########################################################


def optimize_class(model: nn.Module,
                   class_idx: int,
                   n_genes: int,
                   n_epochs: int = 10000,
                   lr: float = 10.0,
                   reg_strength: float = 1e-4,
                   verbose: bool = True,
                   log_iter: int = 1000,
                   _init_vector: np.ndarray = None,
                   ) -> (torch.FloatTensor, list, float):
    '''
    Optimize the input to a model to maximize the probability
    of a target class.

    Parameters
    ----------
    model : nn.Module
        pretrained classification model.
    class_idx : int
        class index to optimize.
    n_genes : int
        number of genes as input to the model.
    n_epochs : int
        number of epochs for optimization.
    lr : float
        learning rate.
    reg_strength : float
        L1 norm regularization strength.
    verbose : bool
        print losses during training.
    _init_vector : np.ndarray
        [n_genes,] vector to use for model initialization.
        if `None`, uses a unit Gaussian to initialize.

    Returns
    -------
    optimum : torch.FloatTensor
        [n_genes,] vector that optimizes the target class probability.
    losses : list
        loss for every `log_iter` iterations of input optimization.
    min_loss : float
        minimum identified loss value, corresponds to the loss at 
        `optimum`.

    See Also
    --------
    find_cell_type_markers
    '''
    print('Making input with %d genes' % n_genes)
    print('Using l1_strength: ', reg_strength)
    if _init_vector is None:
        # Initialize an input vector with Gaussian noise
        input_ = torch.rand(n_genes).float().unsqueeze(0)
    else:
        # Prepare the supplied input vector as a torch.FloatTensor
        if type(_init_vector) == np.ndarray:
            input_ = torch.from_numpy(_init_vector).float().unsqueeze(0)
        elif type(_init_vector) == torch.FloatTensor:
            if _init_vector.size(0) != 1:
                # add an empty batch dimension if it's not present
                input_ = _init_vector.unsqueeze(0)
            else:
                input_ = _init_vector
        else:
            raise ValueError('_init_vector type %s is invalid.' %
                             str(type(_init_vector)))
    if torch.cuda.is_available():
        input_ = input_.cuda()

    # Collect gradients on the input
    input_.requires_grad = True
    print(input_.size())
    # Set up an optimizer to take gradient steps on the input
    optimizer = torch.optim.Adadelta([input_], lr=lr)

    # add a ReLU to the beginning of the model to generate
    # only non-negative inputs
    relu_model = nn.Sequential(nn.ReLU(),
                               model)

    # initialize best loss as a high value
    min_loss = 1e6
    optimum = copy.copy(input_)
    losses = []

    for epoch in range(n_epochs):
        # compute new gradients and take a step down
        # the gradient
        optimizer.zero_grad()
        output = relu_model(input_)
        sm_output = F.softmax(output, dim=1)
        l1 = torch.norm(input_[0, :], 1)
        loss = -sm_output[0, class_idx] + reg_strength*l1
        if loss < min_loss:
            optimum = copy.copy(input_)
            min_loss = loss.detach().cpu().item()
        loss.backward()
        optimizer.step()

        if verbose and epoch % log_iter == 0:
            print('Epoch %d Loss %f' % (epoch, loss.detach().cpu().item()))
            print(input_)
        if epoch % log_iter == 0:
            losses.append(loss.detach().cpu().item())
    if verbose:
        print('Optimal activation')
        print(optimum)
        print('Best Loss')
        print(min_loss)
        print('Number non-zero genes')
        print(torch.sum(optimum > 0.01))
    return optimum, losses, min_loss


def find_cell_type_markers(X,
                           metadata,
                           model_path: str,
                           out_path: str,
                           genes_to_use: list = None,
                           upper_group: str = 'tissue',
                           lower_group: str = 'cell_ontology_class',
                           reg_strength: float = 1e4,
                           **kwargs,) -> None:
    '''
    Find the maximally activating input for each cell type class.

    Parameters
    ----------
    X : np.ndarray
        [Cells, Genes] of log1p transformed, normalized values.
        log1p and normalization performed using scanpy defaults.
    metadata : pd.DataFrame
        [Cells, Features] data with 'tissue' and 'cell_ontology_class' columns.
    model_path : str
        path to pretrained model weights.
    out_path : str
        path for outputs.
    genes_to_use : list
        gene str names to use for training.
        `len(genes_to_use) == X.shape[1]`
    upper_group : str
        column in `metadata` with subsets for training `lower_group`
        classifiers independently. i.e. tissues.
    lower_group : str
        column in `metadata` with output classes. i.e. cell types.
    reg_strength : float
        L1 norm regularization strength.

    Returns
    -------
    None.

    Notes
    -----
    We identify markers for each output class using an input 
    optimization approach. For each output class, we instantiate
    an input cell profile vector using random noise, and optimize
    the vector by gradient descent to maximize the probability
    on the target output class. 

    By including l_1 regularization, we can encourage optimal 
    input vectors to be sparse. This sparsity enables us to
    identify marker genes for the target class. Here, we perform
    the optimization across 10 separate initializations to assess
    the consistency of genes we discover.

    See Also
    --------
    optimize_class
    '''
    # Identify relevant output classes
    celltypes = sorted(list(set(metadata[lower_group])))
    n_cell_types = len(celltypes)
    n_genes = X.shape[1]

    # Instantiate a pre-trained model
    model = CellTypeCLF(n_genes=n_genes,
                        n_cell_types=n_cell_types,
                        **kwargs)
    print('Loading pretrained model...')
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    if torch.cuda.is_available():
        model = model.cuda()
    print('Model loaded.')
    model.eval()

    # For each output class, perform input optimization
    optima = []
    losses = []
    for class_idx in range(len(celltypes)):
        cell_type_optima = []
        cell_type_losses = []
        print('Optimizing for %s' % str(celltypes[class_idx]))
        for rep in range(10):
            # Perform 10 independent iterations of class optimization
            # to evaluate the consistency of identified optima
            optimum, losses, best_loss = optimize_class(
                model=model,
                class_idx=class_idx,
                n_genes=n_genes,
                reg_strength=reg_strength)

            cell_type_optima.append(
                optimum.detach().cpu().numpy())
            cell_type_losses.append(losses)

        # Output the cell type optima as a matrix and save the
        # associated loss values to CSV
        cell_type_optima = np.concatenate(cell_type_optima, axis=0)
        optima.append(cell_type_optima)

        cell_type_losses = np.stack(losses, axis=0)
        np.savetxt(
            osp.join(out_path,
                     (str(class_idx).zfill(3)
                      + '_'
                      + str(celltypes[class_idx])
                      .replace(' ', '_')
                      .upper()
                      + '_losses.csv')
                     ),
            cell_type_losses,
            delimiter=',')
        np.savetxt(
            osp.join(out_path,
                     (str(class_idx).zfill(3)
                      + '_'
                      + str(celltypes[class_idx])
                      .replace(' ', '_')
                      .upper()
                      + '_optima.csv')),
            cell_type_optima,
            delimiter=',')

    # Save a single large optima matrix across all cell types
    optima = np.stack(optima, axis=0)  # [CellTypes, Replicates, Genes]
    optima = np.reshape(optima, (n_cell_types*10, n_genes)
                        )  # [CellType*Replicates, Genes]
    df = pd.DataFrame(optima)
    if genes_to_use is not None:
        df.columns = genes_to_use
    df.insert(0, lower_group, np.repeat(celltypes, 10))
    df.to_csv(
        osp.join(out_path, '%s_optima.csv' %
                 lower_group.replace(' ', '_').lower()),
        index=False)
    return

#########################################################
# Predict cell types with a trained model
#########################################################


def predict_cell_types(X: np.ndarray,
                       model_path: str,
                       out_path: str,
                       upper_groups: Union[list, np.ndarray] = None,
                       lower_group_labels: list = None,
                       **kwargs) -> None:
    '''Predict cell types using a pretrained model

    Parameters
    ----------
    X : np.ndarray
        [Cells, Genes] of log1p transformed, normalized values.
        log1p and normalization performed using scanpy defaults.
    model_path : str
        path to a set of pretrained model weights.
    out_path : str
        path for prediction outputs.
    upper_groups : list, np.ndarray
        [Cells,] iterable of str specifying the `upper_group` for each cell.
        if provided, assumes an `upper_group` conditional model.
        if `None`, assumes an `upper_group` independent model.
    lower_group_labels : list
        str labels corresponding to output nodes of the model.

    Returns
    -------
    None.

    Notes
    -----
    `**kwargs` passed to `scnym.predict.Predicter`.
    '''
    if upper_groups is not None:
        print('Assuming conditional model.')

        X, categories = utils.append_categorical_to_data(X, upper_groups)
        np.savetxt(osp.join(out_path, 'category_names.csv'),
                   categories,
                   fmt='%s',
                   delimiter=',')
    else:
        print('Assuming independent model')

    # Intantiate a prediction object, which handles batch processing
    P = Predicter(model_weights=model_path,
                  n_genes=X.shape[1],
                  n_cell_types=None,  # infer cell type # from weights
                  labels=lower_group_labels,
                  **kwargs)

    predictions, names, scores = P.predict(X, output='score')
    
    probabilities = F.softmax(torch.from_numpy(scores), dim=1)
    probabilities = probabilities.cpu().numpy()

    np.savetxt(osp.join(out_path, 'predictions_idx.csv'),
               predictions, delimiter=',')
    np.savetxt(osp.join(out_path, 'probabilities.csv'),
               probabilities, delimiter=',')
    np.savetxt(osp.join(out_path, 'raw_scores.csv'),
               scores, delimiter=',')
    if names is not None:
        np.savetxt(osp.join(out_path, 'predictions_names.csv'),
                   names, delimiter=',', fmt='%s')
    return


#########################################################
# main()
#########################################################

def main():
    import configargparse
    parser = configargparse.ArgParser(
        description='Train cell type classifiers',
        default_config_files=['./configs/default_config.txt'])
    parser.add_argument('command', type=str,
                        help='action to perform. \
                       ["train_tissue_independent", \
                       "train_tissue_dependent", \
                       "train_tissue_specific", \
                       "find_cell_type_markers", \
                       "predict_cell_types"]')
    parser.add_argument('-c', is_config_file=True, required=False,
                        help='path to a configuration file.')
    parser.add_argument('--input_counts', type=str, required=True,
                        help='path to input data [Cells, Genes] counts. \
                        [npy, csv, h5ad, loom]')
    parser.add_argument('--input_gene_names', type=str, required=True,
                        help='path to gene names for the input data.')
    parser.add_argument('--training_gene_names', type=str, required=False,
                        help='path to training data gene names. \
                        required for prediction.')
    parser.add_argument('--training_metadata', type=str, required=True,
                        help='CSV metadata for training. Requires `upper_group` and `lower_group` columns. \
                       necessary for prediction to provide cell type names.')
    parser.add_argument('--lower_group', type=str, required=True,
                        default='cell_ontology_class',
                        help='column in `metadata` with to output labels. \
                        i.e. cell types.')
    parser.add_argument('--upper_group', type=str, required=True,
                        default='tissue',
                        help='column in `metadata` with to subsets for independent training. \
                        i.e. tissues.')
    parser.add_argument('--out_path', type=str, required=True,
                        help='path for output files')
    parser.add_argument('--genes_to_use', type=str, default=None,
                        help='path to a text file of genes to use for training. \
                       must be a subset of genes in `training_gene_names`')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch size for training')
    parser.add_argument('--n_epochs', type=int, default=256,
                        help='number of epochs for training')
    parser.add_argument('--init_dropout', type=float, default=0.3,
                        help='initial dropout to perform on gene inputs')
    parser.add_argument('--n_hidden', type=int, default=128,
                        help='number of hidden units in the classifier')
    parser.add_argument('--n_layers', type=int, default=2,
                        help='number of hidden layers in the model')
    parser.add_argument('--residual', action='store_true',
                        help='use residual layers in the model')
    parser.add_argument('--model_path', type=str, default=None,
                        help='path to pretrained model weights \
                        for class marker identification.')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='weight decay applied by the optimizer')
    parser.add_argument('--l1_reg', type=float, default=1e-4,
                        help='l1 regularization strength \
                        for class marker identification')
    parser.add_argument('--balance_classes', type=bool, default=True,
                        help='perform class balancing')
    args = parser.parse_args()

    print(args)
    print(parser.format_values())

    COMMANDS = [
        'train_tissue_independent',
        'train_tissue_dependent',
        'train_tissue_specific',
        'predict_cell_types',
        'find_cell_type_markers',
    ]

    if args.command not in COMMANDS:
        raise ValueError('%s is not a valid command.' % args.command)

    #####################################
    # LOAD DATA
    #####################################

    if osp.splitext(args.input_counts)[-1] == '.npy':
        print('Assuming sparse matrix...')
        X_raw = np.load(args.input_counts, allow_pickle=True)
        X_raw = X_raw.item()
    elif osp.splitext(args.input_counts)[-1] == '.csv':
        X_raw = np.loadtxt(args.input_counts, delimiter=',')
    elif osp.splitext(args.input_counts)[-1] == '.h5ad':
        adata = sc.read_h5ad(args.input_counts)
        X_raw = adata.X
    elif osp.splitext(args.input_counts)[-1] == '.loom':
        adata = sc.read_loom(args.input_counts)
        X_raw = adata.X
    else:
        raise ValueError('unrecognized file type %s for `input_counts`' %
                         osp.splitext(args.input_counts)[-1])

    if type(X_raw) != np.ndarray:
        # If X_raw is not np.ndarray, we assume it's part of the
        # scipy.sparse family and use the `.toarray()` method
        # to densify the matrix.
        print('Assuming sparse matrix, densifying...')
        try:
            X_raw = X_raw.toarray()
        except RuntimeError:
            print(
                'Counts object was neither `np.ndarray` or matrix with `.toarray()` method.')
            print('Ensure that the counts you load are either a dense `np.ndarray` \
                or part of the `scipy.sparse` family of matrices.')

    print('Loaded data.')
    print('%d cells and %d genes in raw data.' % X_raw.shape)
    gene_names = np.loadtxt(args.input_gene_names, dtype='str')
    print('Loaded gene names for the raw data. %d genes.' % len(gene_names))

    if args.genes_to_use is not None:
        genes_to_use = np.loadtxt(args.genes_to_use, dtype='str')
        print('Using a subset of %d genes as specified in \n %s.' %
              (len(genes_to_use), args.genes_to_use))
    else:
        genes_to_use = gene_names

    if args.genes_to_use is not None:
        # Filter the input matrix to use only the specified genes
        print('Using %d genes for classification.' % len(genes_to_use))
        gnl = gene_names.tolist()
        keep_idx = np.array([gnl.index(x) for x in genes_to_use])
        X = X_raw[:, keep_idx]
    else:
        # leave all genes in the matrix
        X = X_raw

    # Load metadata and identify output classes
    metadata = pd.read_csv(args.training_metadata,)
    lower_groups = sorted(list(set(metadata[args.lower_group])))

    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)

    sub_dirs = ['tissues', 'tissue_independent',
                'tissue_dependent', 'tissue_ind_class_optimums']
    for sd in sub_dirs:
        if not os.path.exists(osp.join(args.out_path, sd)):
            os.mkdir(osp.join(args.out_path, sd))

    #####################################
    # TISSUE INDEPENDENT CLASSIFIERS
    #####################################

    if args.command == 'train_tissue_independent':
        train_tissue_independent_cv(X,
                                    metadata,
                                    osp.join(args.out_path,
                                             'tissue_independent'),
                                    balanced_classes=args.balance_classes,
                                    batch_size=args.batch_size,
                                    n_epochs=args.n_epochs,
                                    init_dropout=args.init_dropout,
                                    lower_group=args.lower_group,
                                    n_hidden=args.n_hidden,
                                    n_layers=args.n_layers,
                                    weight_decay=args.weight_decay,
                                    residual=args.residual)

    #####################################
    # TISSUE DEPENDENT CLASSIFIERS
    #####################################

    if args.command == 'train_tissue_dependent':
        train_tissue_dependent_cv(X,
                                  metadata,
                                  osp.join(args.out_path, 'tissue_dependent'),
                                  balanced_classes=args.balance_classes,
                                  batch_size=args.batch_size,
                                  n_epochs=args.n_epochs,
                                  upper_group=args.upper_group,
                                  lower_group=args.lower_group,
                                  n_hidden=args.n_hidden,
                                  n_layers=args.n_layers,
                                  weight_decay=args.weight_decay,
                                  residual=args.residual)

    #####################################
    # TISSUE SPECIFIC CLASSIFIERS
    #####################################

    if args.command == 'train_tissue_specific':

        tissue_specific_bal = {
            k: False for k in set(metadata[args.upper_group])}
        if args.balance_classes:
            for k in tissue_specific_bal:
                tissue_specific_bal[k] = True

        train_tissue_specific_cv(X,
                                 metadata,
                                 osp.join(args.out_path, 'tissues'),
                                 balanced_classes=tissue_specific_bal,
                                 batch_size=args.batch_size,
                                 n_epochs=args.n_epochs,
                                 upper_group=args.upper_group,
                                 lower_group=args.lower_group,
                                 n_hidden=args.n_hidden,
                                 n_layers=args.n_layers,
                                 weight_decay=args.weight_decay,
                                 residual=args.residual)

    #####################################
    # INPUT OPTIMIZATION
    #####################################

    if args.command == 'find_cell_type_markers':
        if args.model_path is None:
            raise ValueError('`model_path` required.')
        model_path = args.model_path
        find_cell_type_markers(X,
                               metadata,
                               model_path=model_path,
                               out_path=osp.join(
                                   args.out_path, 'tissue_ind_class_optimums'),
                               genes_to_use=genes_to_use,
                               reg_strength=args.l1_reg,
                               upper_group=args.upper_group,
                               lower_group=args.lower_group,
                               n_hidden=args.n_hidden,
                               n_layers=args.n_layers,
                               residual=args.residual)

    #####################################
    # PRETRAINED MODEL PREDICTION
    #####################################

    if args.command == 'predict_cell_types':
        if args.model_path is None:
            raise ValueError('`model_path` required.')
        if args.training_gene_names is None:
            raise ValueError('must supply `training_gene_names`.')
        training_genes = np.loadtxt(
            args.training_gene_names, delimiter=',', dtype='str').tolist()

        X = utils.build_classification_matrix(
            X=X,
            model_genes=training_genes,
            sample_genes=gene_names)

        predict_cell_types(X,
                           model_path=args.model_path,
                           out_path=args.out_path,
                           lower_group_labels=lower_groups,
                           n_hidden=args.n_hidden,
                           n_layers=args.n_layers,
                           residual=args.residual)

#########################################################
# __main__
#########################################################


if __name__ == '__main__':

    main()
