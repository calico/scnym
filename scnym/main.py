'''Train scNym models and identify cell type markers'''
import numpy as np
import pandas as pd
from scipy import sparse
import os
import os.path as osp
import scanpy as sc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from typing import Union
import copy
import itertools
from functools import partial

from .model import CellTypeCLF, CellTypeCLFConditional

from .dataprep import SingleCellDS, SampleMixUp, balance_classes
from .dataprep import AUGMENTATION_SCHEMES
from .trainer import Trainer, SemiSupervisedTrainer
from .trainer import cross_entropy, get_class_weight
from .trainer import InterpolationConsistencyLoss, ICLWeight, MixMatchLoss, DANLoss

from .predict import Predicter
from . import utils

# define optimizer map for cli selection
OPTIMIZERS = {
    'adadelta': torch.optim.Adadelta,
    'adam': torch.optim.Adam,
    'adamw': torch.optim.AdamW,
    'sgd': torch.optim.SGD,
}

#########################################################
# Train scNym classification models
#########################################################


def repeater(data_loader):
    """Use `itertools.repeat` to infinitely loop through
    a dataloader.
    
    Parameters
    ----------
    data_loader : torch.utils.data.DataLoader
        data loader class.
    
    Yields
    ------
    data : Iterable
        batches from `data_loader`.
        
    Credit
    ------
    https://bit.ly/2z0LGm8
    """
    for loader in itertools.repeat(data_loader):
        for data in loader:
            yield data


def fit_model(
    X: Union[np.ndarray, sparse.csr.csr_matrix],
    y: np.ndarray,
    traintest_idx: Union[np.ndarray, tuple],
    val_idx: np.ndarray,
    batch_size: int,
    n_epochs: int,
    lr: float,
    optimizer_name: str,
    weight_decay: float,
    ModelClass: nn.Module,
    out_path: str,
    n_genes: int = None,
    mixup_alpha: float = None,
    unlabeled_counts: np.ndarray = None,
    unsup_max_weight: float = 2.,
    unsup_mean_teacher: bool = False,
    ssl_method: str='mixmatch',
    ssl_kwargs: dict={},
    weighted_classes: bool = False,
    balanced_classes: bool = False,
    input_domain: np.ndarray=None,
    unlabeled_domain: np.ndarray=None,
    pretrained: str=None,
    patience: int=None,
    save_freq: int=None,
    tensorboard: bool=True,
    **kwargs,
) -> (float, float):
    '''Fit an scNym model given a set of observations and labels.
    
    Parameters
    ----------
    X : np.ndarray
        [Cells, Genes] of log1p transformed normalized values.
        log1p and normalization performed using scanpy defaults.
    y : np.ndarray
        [Cells,] integer class labels.
    traintest_idx : np.ndarray
        [Int,] indices to use for training and early stopping.
        a single array will be randomly partitioned, OR a tuple
        of `(train_idx, test_idx)` can be passed.
    val_idx : np.ndarray
        [Int,] indices to hold-out for final model evaluation.
    n_epochs : int
        number of epochs for training.
    lr : float
        learning rate.
    optimizer_name : str
        optimizer to use. {"adadelta", "adam"}.
    weight_decay : float
        weight decay to apply to model weights.
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
    mixup_alpha : float
        alpha parameter for an optional MixUp augmentation during training.
    unlabeled_counts : np.ndarray
        [Cells', Genes] of log1p transformed normalized values for 
        unlabeled observations.
    unsup_max_weight : float
        maximum weight for the unsupervised loss term.
    unsup_mean_teacher : bool
        use a mean teacher for pseudolabel generation.
    ssl_method : str
        semi-supervised learning method to use.
    ssl_kwargs : dict
        arguments passed to the semi-supervised learning loss.
    balanced_classes : bool
        perform class balancing by undersampling majority classes.
    weighted_classes : bool
        weight loss for each class based on relative abundance of classes
        in the training data.
    input_domain : np.ndarray
        [Cells,] integer domain labels for training data.
    unlabeled_domain : np.ndarray
        [Cells',] integer domain labels for unlabeled data.
    pretrained : str
        path to a pretrained model for initialization.
        default: `None`.
    patience : int
        number of epochs to wait before early stopping.
        `None` deactivates early stopping.
    save_freq : int
        frequency in epochs for saving model checkpoints.
        if `None`, saves >=5 checkpoints per model.
    tensorboard : bool
        save logs to tensorboard.
        
    Returns
    -------
    test_acc : float
        classification accuracy on the test set.
    test_loss : float
        supervised loss on the test set.
    '''
    # count the number of cell types available
    n_cell_types = len(np.unique(y))
    if n_genes is None:
        n_genes = X.shape[1]
        
    if type(traintest_idx) != tuple:
        # Set aside 10% of the traintest data for model selection in `test_idx`
        train_idx = np.random.choice(
            traintest_idx,
             size=int(
                 np.floor(0.9 * len(traintest_idx))),
             replace=False,
        ).astype('int')
        test_idx = np.setdiff1d(traintest_idx, train_idx).astype('int')
    elif type(traintest_idx) == tuple and len(traintest_idx) == 2:
        # use the user provided train/test split
        train_idx = traintest_idx[0]
        test_idx  = traintest_idx[1]
    else:
        # the user supplied an invalid argument
        msg = '`traintest_idx` of type {type(traintest_idx)}\n'
        msg += 'and length {len(traintest_idx)} is invalid.'
        raise ValueError(msg)
        
    # save indices to CSVs for later retrieval
    np.savetxt(osp.join(out_path, 'train_idx.csv'), train_idx)
    np.savetxt(osp.join(out_path, 'test_idx.csv'), test_idx)
    np.savetxt(osp.join(out_path, 'val_idx.csv'), val_idx)

    # balance or weight classes if applicable
    if balanced_classes and weighted_classes:
        msg = 'balancing AND weighting classes is not useful.'
        msg += '\nPick one mode of accounting for class imbalances.'
        raise ValueError(msg)
    elif balanced_classes and not weighted_classes:
        print('Setting up a stratified sampler...')
        # we sample classes with weighted likelihood, rather than
        # a uniform likelihood of sampling
        # we use the inverse of the class count as a weight
        # this is normalized in `WeightedRandomSample`
        classes, counts = np.unique(y[train_idx], return_counts=True)
        sample_weights = 1./counts
        
        # `WeightedRandomSampler` is kind of funny and takes a weight
        # **per example** in the training set, rather than per class.
        # here we assign the appropriate class weight to each sample
        # in the training set.
        weight_per_example = sample_weights[y[train_idx]]
        
        # we instantiate the sampler with the relevant weight for
        # each observation and set the number of total samples to the
        # number of samples in our training set
        # `WeightedRandomSampler` will sample indices from a multinomial
        # with probabilities computed from the normalized vector
        # of `weights_per_example`.
        sampler = torch.utils.data.sampler.WeightedRandomSampler(
            weight_per_example, 
            len(y[train_idx]),
        )
        class_weight = None
    elif weighted_classes and not balanced_classes:
        # compute class weights
        # class weights amplify the loss of some classes and reduce
        # the loss of others, inversely proportional to the class
        # frequency
        print('Weighting classes for training...')
        class_weight = get_class_weight(y[train_idx])
        print(class_weight)
        print()
        sampler = None
    else:
        print('Not weighting classes and not balancing classes.')
        class_weight = None
        sampler = None

    # Generate training and model selection Datasets and Dataloaders
    X_train = X[train_idx, :]
    y_train = y[train_idx]

    X_test = X[test_idx, :]
    y_test = y[test_idx]
    
    # count the number of domains
    if (input_domain is None) and (unlabeled_domain is None):
        n_domains = 2
    elif (input_domain is not None) and (unlabeled_domain is not None):
        n_domains = int(np.max([
            input_domain.max(), unlabeled_domain.max(),
        ])) + 1
    else:
        msg = 'domains supplied for only one set of data'
        raise ValueError(msg)
    print(f'Found {n_domains} unique domains.')
    
    if input_domain is not None:
        d_train = input_domain[train_idx]
        d_test  = input_domain[test_idx]
    else:
        d_train = None
        d_test  = None

    train_ds = SingleCellDS(
        X=X_train,
        y=y_train,
        num_classes=len(np.unique(y)),
        domain = d_train,
        num_domains = n_domains,
    )
    test_ds = SingleCellDS(
        X_test,
        y_test,
        num_classes=len(np.unique(y)),
        domain = d_test,
        num_domains = n_domains,
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True if sampler is None else False,
        sampler=sampler,
        drop_last=True,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=True,
    )

    dataloaders = {
        'train': train_dl,
        'val': test_dl,
    }

    # Define batch transformers
    batch_transformers = {}
    if mixup_alpha is not None and ssl_method != 'mixmatch':
        print('Using MixUp as a batch transformer.')
        batch_transformers['train'] = SampleMixUp(alpha=mixup_alpha)

    # Build a cell type classification model and transfer to CUDA
    model = ModelClass(
        n_genes=n_genes,
        n_cell_types=n_cell_types,
        **kwargs,
    )
    
    if pretrained is not None:
        # initialize with supplied weights
        model.load_state_dict(
            torch.load(
                pretrained,
                map_location='cpu',
            )
        )
    
    if torch.cuda.is_available():
        model = model.cuda()

    # Set up loss criterion and the model optimizer
    # here we use our own cross_entropy loss to handle
    # discrete probability distributions rather than
    # categorical predictions
    if class_weight is None:
        criterion = cross_entropy
    else:
        criterion = partial(
            cross_entropy,
            class_weight=torch.from_numpy(class_weight).float(),
        )
        
    opt_callable = OPTIMIZERS[optimizer_name.lower()]
    
    if opt_callable != torch.optim.SGD:
        optimizer = opt_callable(
            model.parameters(),
            weight_decay=weight_decay,
            lr=lr,
        )
        scheduler = None
    else:
        # use SGD as the optimizer with momentum
        # and a learning rate scheduler
        optimizer = opt_callable(
            model.parameters(),
            weight_decay=weight_decay,
            lr=lr,
            momentum=0.9,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=n_epochs,
            eta_min=lr/10000,
        )
        

    # Build the relevant trainer object for either supervised
    # or semi-supervised learning with interpolation consistency
    trainer_kwargs = {
        'model': model,
        'criterion': criterion,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'dataloaders': dataloaders,
        'out_path': out_path,
        'batch_transformers': batch_transformers,
        'n_epochs': n_epochs,
        'min_epochs': n_epochs//20,
        'save_freq': max(n_epochs//5, 1) if save_freq is None else save_freq,
        'reg_criterion': None,
        'exp_name': osp.basename(out_path),
        'verbose': False,
        'tb_writer' : osp.join(out_path, 'tblog') if tensorboard else None,
        'patience': patience,
    }

    if unlabeled_counts is None:
        # perform fully supervised training
        T = Trainer(**trainer_kwargs)
    else:
        # perform semi-supervised training
        unsup_dataset = SingleCellDS(
            X=unlabeled_counts,
            y=np.zeros(unlabeled_counts.shape[0]),
            num_classes=len(np.unique(y)),
            domain=unlabeled_domain,
            num_domains=n_domains,
        )
        
        # Build a semi-supervised data loader that infinitely samples
        # unsupervised data for interpolation consistency.
        # This allows us to loop through the labeled data iterator
        # without running out of unlabeled batches.        
        unsup_dataloader = DataLoader(
            unsup_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )
        unsup_dataloader = repeater(unsup_dataloader)

        # Set up the unsupervised loss
        if ssl_method.lower() == 'ict':
            print('Using ICT for semi-supervised learning')                
            USL = InterpolationConsistencyLoss(
                alpha=mixup_alpha if mixup_alpha is not None else 0.3,
                unsup_criterion=nn.MSELoss(),
                sup_criterion=criterion,
                decay_coef=ssl_kwargs.get('decay_coef', 0.997),
                mean_teacher=unsup_mean_teacher,
            )

        elif ssl_method.lower() == 'mixmatch':
            print('Using MixMatch for semi-supervised learning')
            # we want the raw MSE per sample here, rather than the average
            # so we set `reduction='none'`.
            # this allows us to scale the weight of individual examples
            # based on pseudolabel confidence.
            unsup_criterion_name = ssl_kwargs.get('unsup_criterion', 'mse')
            if unsup_criterion_name.lower() == 'mse':
                unsup_criterion = nn.MSELoss(reduction='none')
            elif unsup_criterion_name.lower() in ('crossentropy', 'ce'):
                unsup_criterion = partial(
                    cross_entropy,
                    reduction='none',
                )
            USL = MixMatchLoss(
                alpha=mixup_alpha if mixup_alpha is not None else 0.3,
                unsup_criterion=unsup_criterion,
                sup_criterion=criterion,
                decay_coef=ssl_kwargs.get('decay_coef', 0.997),
                mean_teacher=unsup_mean_teacher,
                augment=AUGMENTATION_SCHEMES[ssl_kwargs.get('augment', 'log1p_drop')],
                n_augmentations=ssl_kwargs.get('n_augmentations', 1),
                T=ssl_kwargs.get('T', 0.5),
                augment_pseudolabels=ssl_kwargs.get('augment_pseudolabels', True),
                pseudolabel_min_confidence=ssl_kwargs.get('pseudolabel_min_confidence', 0.0),
            )
        else:
            msg = f'{ssl_method} is not a valid semi-supervised learning method.\n'
            msg += 'must be one of {"ict", "mixmatch"}'
            raise ValueError(msg)

        # set up the weight schedule
        # we define a number of epochs for ramping, a number to wait
        # ("burn_in_epochs") before we start the ramp up, and a maximum
        # coefficient value
        weight_schedule = ICLWeight(
            ramp_epochs=ssl_kwargs.get('ramp_epochs', max(n_epochs//4, 1)),
            max_unsup_weight=unsup_max_weight,
            burn_in_epochs = ssl_kwargs.get('burn_in_epochs', 20),
            sigmoid = ssl_kwargs.get('sigmoid', False),
        )
        # don't let early stopping save checkpoints from before the SSL 
        # ramp up has started
        trainer_kwargs['min_epochs'] = max(
            trainer_kwargs['min_epochs'],
            weight_schedule.burn_in_epochs + weight_schedule.ramp_epochs // 5,
        )
        
        # if min_epochs are manually specified, use that number instead
        if ssl_kwargs.get('min_epochs', None) is not None:
            trainer_kwargs['min_epochs'] = ssl_kwargs['min_epochs']
        
        # let the model save weights even if the ramp is 
        # longer than the total epochs we'll train for
        trainer_kwargs['min_epochs'] = min(
            trainer_kwargs['min_epochs'],
            trainer_kwargs['n_epochs']-1,
        )
        
        dan_criterion = ssl_kwargs.get('dan_criterion', None)
        if dan_criterion is not None:
            # initialize the DAN Loss
            
            dan_criterion = DANLoss(
                model=model,
                dan_criterion=cross_entropy,
                use_conf_pseudolabels=ssl_kwargs.get('dan_use_conf_pseudolabels', False),
                scale_loss_pseudoconf=ssl_kwargs.get('dan_scale_loss_pseudoconf', False),
                n_domains = n_domains,
            )

            # setup the DANN learning rate schedule
            dan_weight = ICLWeight(
                ramp_epochs=ssl_kwargs.get('dan_ramp_epochs', max(n_epochs//4, 1)),
                max_unsup_weight=ssl_kwargs.get('dan_max_weight', 1.),
                burn_in_epochs = ssl_kwargs.get('dan_burn_in_epochs', 0),
                sigmoid = ssl_kwargs.get('sigmoid', True),
            )
            # add DANN parameters to the optimizer
            optimizer.add_param_group({
                'params': dan_criterion.dann.domain_clf.parameters(),
                'name': 'domain_classifier',
            })
        else:
            dan_weight = None

        # initialize the trainer
        T = SemiSupervisedTrainer(
            unsup_dataloader=unsup_dataloader,
            unsup_criterion=USL,
            unsup_weight=weight_schedule,
            dan_criterion=dan_criterion,
            dan_weight=dan_weight,
            **trainer_kwargs,
        )

    print('Training...')
    T.train()
    print('Training complete.')
    print()

    # Perform model evaluation using the best set of weights on the
    # totally unseen, held out data.
    print('Evaluating model.')
    model = ModelClass(
        n_genes=n_genes,
        n_cell_types=n_cell_types,
        **kwargs,
    )
    model.load_state_dict(
        torch.load(
            osp.join(out_path, '00_best_model_weights.pkl'),
        )
    )
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    # Build a DataLoader for validation
    X_val = X[val_idx, :]
    y_val = y[val_idx]
    val_ds = SingleCellDS(
        X_val, 
        y_val,
        num_classes=len(np.unique(y)),            
    )
    val_dl = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False, 
    )

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

            label_ = data['output']  # one-hot

            if torch.cuda.is_available():
                input_ = input_.cuda()
                label_ = label_.cuda()

            # make an integer version of labels for convenience
            int_label_ = torch.argmax(label_, 1)

            # Perform forward pass and compute predictions as the
            # most likely class
            output = model(input_)
            _, predictions = torch.max(output, 1)

            corrects = torch.sum(
                predictions.detach()== int_label_.detach(),
            )

            l = criterion(output, label_)
            loss += float(l.detach().cpu().numpy())

            running_corrects += float(corrects.item())
            running_total += float(label_.size(0))

            all_labels.append(
                int_label_.detach().cpu().numpy()
            )

            all_predictions.append(
                predictions.detach().cpu().numpy()
            )

        norm_loss = loss / len(val_dl)
        acc = running_corrects/running_total
        print('EVAL LOSS: ', norm_loss)
        print('EVAL ACC : ', acc)

    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)
    np.savetxt(
        osp.join(out_path, 'predictions.csv'),
        all_predictions)
    np.savetxt(
        osp.join(out_path, 'labels.csv'),
        all_labels)

    PL = np.stack([all_predictions, all_labels], 0)
    print('Predictions | Labels')
    print(PL.T[:15, :])
    return acc, norm_loss


def train_cv(
    X: Union[np.ndarray, sparse.csr.csr_matrix],
    y: np.ndarray,
    batch_size: int,
    n_epochs: int,
    lr: float,
    optimizer_name: str,    
    weight_decay: float,
    ModelClass: nn.Module,
    fold_indices: list,
    out_path: str,
    n_genes: int = None,
    mixup_alpha: float = None,
    unlabeled_counts: np.ndarray = None,
    unsup_max_weight: float = 2.,
    unsup_mean_teacher: bool = False,
    ssl_method: str='mixmatch',
    ssl_kwargs: dict={},
    weighted_classes: bool = False,
    balanced_classes: bool = False,
    **kwargs,
) -> (np.ndarray, np.ndarray):
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
    weight_decay : float
        weight decay to apply to model weights.
    lr : float
        learning rate.
    optimizer_name : str
        optimizer to use. {"adadelta", "adam"}.        
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
    mixup_alpha : float
        alpha parameter for an optional MixUp augmentation during training.
    unsup_max_weight : float
        maximum weight for the unsupervised loss term.
    unsup_mean_teacher : bool
        use a mean teacher for pseudolabel generation.
    ssl_method : str
        semi-supervised learning method to use.
    ssl_kwargs : dict
        arguments passed to the semi-supervised learning loss.
    balanced_classes : bool
        perform class balancing by undersampling majority classes.
    weighted_classes : bool
        weight loss for each class based on relative abundance of classes
        in the training data.

    Returns
    -------
    fold_eval_acc : np.ndarray
        evaluation accuracies for each fold.
    fold_eval_losses : np.ndarray
        loss values for each fold.
    '''
    fold_eval_losses = np.zeros(len(fold_indices))
    fold_eval_acc = np.zeros(len(fold_indices))

    # Perform training on each fold specified in `fold_indices`
    for f in range(len(fold_indices)):
        print('Training tissue independent, fold %d.' % f)
        fold_out_path = osp.join(out_path, 'fold' + str(f).zfill(2))

        os.makedirs(fold_out_path, exist_ok=True)

        traintest_idx = fold_indices[f][0].astype('int')
        val_idx = fold_indices[f][1].astype('int')

        acc, loss = fit_model(
            X=X,
            y=y,
            traintest_idx=traintest_idx,
            val_idx=val_idx,
            out_path=fold_out_path,
            batch_size=batch_size,
            n_epochs=n_epochs,
            ModelClass=ModelClass,
            n_genes=n_genes,
            lr=lr,
            optimizer_name=optimizer_name,            
            weight_decay=weight_decay,
            mixup_alpha=mixup_alpha,
            unlabeled_counts=unlabeled_counts,
            unsup_max_weight=unsup_max_weight,
            unsup_mean_teacher=unsup_mean_teacher,
            ssl_method=ssl_method,
            ssl_kwargs=ssl_kwargs,    
            weighted_classes=weighted_classes,
            balanced_classes=balanced_classes,
            **kwargs,
        )
        
        fold_eval_losses[f] = loss
        fold_eval_acc[f] = acc
    return fold_eval_acc, fold_eval_losses


def train_all(
    X: Union[np.ndarray, sparse.csr.csr_matrix],
    y: np.ndarray,
    batch_size: int,
    n_epochs: int,
    ModelClass: nn.Module,
    out_path: str,
    n_genes: int = None,
    lr: float=1.0,
    optimizer_name: str='adadelta',
    weight_decay: float = None,
    mixup_alpha: float = None,
    unlabeled_counts: np.ndarray = None,
    unsup_max_weight: float = 2.,
    unsup_mean_teacher: bool = False,
    ssl_method: str='mixmatch',
    ssl_kwargs: dict={},    
    weighted_classes: bool = False,
    balanced_classes: bool = False,
    **kwargs,
) -> (float, float):
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
    lr : float
        learning rate.
    optimizer_name : str
        optimizer to use. {"adadelta", "adam"}.        
    weight_decay : float
        weight decay to apply to model weights.
    balanced_classes : bool
        perform class balancing by undersampling majority classes.
    weighted_classes : bool
        weight loss for each class based on relative abundance of classes
        in the training data.      

    Returns
    -------
    loss : float
        best loss on the testing set used for model selection.
    acc : float
        best accuracy on the testing set used for model selection.
    '''
    # Prepare a unique output directory
    all_out_path = osp.join(out_path, 'all_data')
    if not osp.exists(all_out_path):
        os.mkdir(all_out_path)

    # Generate training and model selection indices
    traintest_idx = np.random.choice(
        np.arange(X.shape[0]),
        size=int(np.floor(0.9*X.shape[0])),
        replace=False,
    ).astype('int')
    val_idx = np.setdiff1d(
        np.arange(X.shape[0]), traintest_idx,
    ).astype('int')

    acc, loss = fit_model(
        X=X,
        y=y,
        traintest_idx=traintest_idx,
        val_idx=val_idx,
        batch_size=batch_size,
        n_epochs=n_epochs,
        ModelClass=ModelClass,
        out_path=all_out_path,
        n_genes=n_genes,
        lr=lr,
        optimizer_name=optimizer_name,        
        weight_decay=weight_decay,
        mixup_alpha=mixup_alpha,
        unlabeled_counts=unlabeled_counts,
        unsup_max_weight=unsup_max_weight,
        unsup_mean_teacher=unsup_mean_teacher,
        ssl_method=ssl_method,
        ssl_kwargs=ssl_kwargs,    
        weighted_classes=weighted_classes,
        balanced_classes=balanced_classes,
        **kwargs,
    )

    np.savetxt(
        osp.join(all_out_path, 'test_loss_acc.csv'),
        np.array([loss, acc]).reshape(2, 1),
        delimiter=',')

    return loss, acc


def train_tissue_independent_cv(
    X: Union[np.ndarray, sparse.csr.csr_matrix],
    metadata: pd.DataFrame,
    out_path: str,
    balanced_classes: bool = False,
    weighted_classes: bool = False,
    batch_size: int = 256,
    n_epochs: int = 200,
    lower_group: str = 'cell_ontology_class',
    **kwargs,
) -> None:
    '''
    Trains a cell type classifier that is independent of tissue origin

    Parameters
    ----------
    X : np.ndarray
        [Cells, Genes] of log1p transformed, normalized values.
        log1p and normalization performed using scanpy defaults.
    metadata : pd.DataFrame
        [Cells, Features] data with `upper_group` and `lower_group` columns.
    out_path : str
        path for saving trained model weights and evaluation performance.
    balanced_classes : bool
        perform class balancing by undersampling majority classes.
    weighted_classes : bool
        weight loss for each class based on relative abundance of classes
        in the training data.
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

    # identify all the `lower_group` levels and create
    # an integer class vector corresponding to unique levels
    y = pd.Categorical(metadata[lower_group]).codes
    y = y.astype('int32')
    labels = pd.Categorical(metadata[lower_group]).categories
    # save mapping of levels : integer values as a CSV
    out_df = pd.DataFrame({'label': labels, 'code': np.arange(len(labels))})
    out_df.to_csv(osp.join(out_path, 'celltype_label.csv'))

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
        balanced_classes=balanced_classes,
        weighted_classes=weighted_classes,
        **kwargs,
    )

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
        balanced_classes=balanced_classes,
        weighted_classes=weighted_classes,
        **kwargs,
    )

    return


def train_tissue_dependent_cv(
    X: Union[np.ndarray, sparse.csr.csr_matrix],
    metadata: pd.DataFrame,
    out_path: str,
    balanced_classes: bool = False,
    weighted_classes: bool = False,
    batch_size: int = 256,
    n_epochs: int = 200,
    upper_group: str = 'tissue',
    lower_group: str = 'cell_ontology_class',
    **kwargs,
) -> None:
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
    weighted_classes : bool
        weight loss for each class based on relative abundance of classes
        in the training data.        
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
        balanced_classes=balanced_classes,
        weighted_classes=weighted_classes,
        **kwargs,
    )

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
        balanced_classes=balanced_classes,
        weighted_classes=weighted_classes,
        **kwargs,
    )
    return


def train_one_tissue_cv(
    X: Union[np.ndarray, sparse.csr.csr_matrix],
    metadata: pd.DataFrame,
    out_path: str,
    balanced_classes: bool = False,
    weighted_classes: bool = False,
    batch_size: int = 256,
    n_epochs: int = 200,
    upper_group: str = 'tissue',
    lower_group: str = 'cell_ontology_class',
    **kwargs,
) -> None:
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
    weighted_classes : bool
        weight loss for each class based on relative abundance of classes
        in the training data.        
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
        weighted_classes=weighted_classes,
        balanced_classes=balanced_classes,
        **kwargs,
    )

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
        weighted_classes=weighted_classes,
        balanced_classes=balanced_classes,        
        **kwargs,
    )
    return


def train_tissue_specific_cv(
    X: Union[np.ndarray, sparse.csr.csr_matrix],
    metadata: pd.DataFrame,
    out_path: str,
    balanced_classes: dict = None,
    weighted_classes: bool = None,
    upper_group: str = 'tissue',
    lower_group: str = 'cell_ontology_class',
    **kwargs,
) -> None:
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
    weighted_classes : bool
        weight loss for each class based on relative abundance of classes
        in the training data.        
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

        train_one_tissue_cv(
            tissue_X,
            tissue_meta,
            tissue_out_path,
            balanced_classes=bc,
            weighted_classes=weighted_classes,
            upper_group=upper_group,
            lower_group=lower_group,
            **kwargs,
        )

    return

#########################################################
# Predict cell types with a trained model
#########################################################


def predict_cell_types(
    X: Union[np.ndarray, sparse.csr.csr_matrix],
    model_path: str,
    out_path: str,
    upper_groups: Union[list, np.ndarray] = None,
    lower_group_labels: list = None,
    **kwargs,
) -> None:
    '''Predict cell types using a pretrained model

    Parameters
    ----------
    X : np.ndarray, sparse.csr.csr_matrix
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
    P = Predicter(
        model_weights=model_path,
        n_genes=X.shape[1],
        n_cell_types=None,  # infer cell type # from weights
        labels=lower_group_labels,
        **kwargs,
    )

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
# utilities
#########################################################


def load_data(
    path: str,
) -> Union[np.ndarray, sparse.csr.csr_matrix]:
    '''Load a counts matrix from a file path.

    Parameters
    ----------
    path : str
        path to [npy, csv, h5ad, loom] file.

    Returns
    -------
    X : np.ndarray
        [Cells, Genes] matrix.
    '''
    if osp.splitext(path)[-1] == '.npy':
        print('Assuming sparse matrix...')
        X_raw = np.load(path, allow_pickle=True)
        X_raw = X_raw.item()
    elif osp.splitext(path)[-1] == '.csv':
        X_raw = np.loadtxt(path, delimiter=',')
    elif osp.splitext(path)[-1] == '.h5ad':
        adata = sc.read_h5ad(path)
        X_raw = utils.get_adata_asarray(adata=adata)
    elif osp.splitext(path)[-1] == '.loom':
        adata = sc.read_loom(path)
        X_raw = utils.get_adata_asarray(adata=adata)
    else:
        raise ValueError('unrecognized file type %s for counts' %
                         osp.splitext(path)[-1])

    return X_raw

#########################################################
# main()
#########################################################


def main():
    import configargparse
    import yaml
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
    parser.add_argument(
        '--input_domain_group',
        type=str,
        help='column in `training_metadata` that specifies domain of origin for each training observation.',
        required=False,
        default=None,
    )
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
    parser.add_argument('--track_running_stats', type=bool, default=True,
                       help='track running statistics in batch normalization layers')
    parser.add_argument('--model_path', type=str, default=None,
                        help='path to pretrained model weights \
                        for class marker identification.')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='weight decay applied by the optimizer')
    parser.add_argument('--lr', type=float, default=1.0,
                        help='learning rate for the optimizer.')
    parser.add_argument('--optimizer', type=str, default='adadelta',
                       help='optimizer to use. {adadelta, adam}.')
    parser.add_argument('--l1_reg', type=float, default=1e-4,
                        help='l1 regularization strength \
                        for class marker identification')
    parser.add_argument('--weight_classes', type=bool, default=False,
                        help='weight loss based on relative class abundance.')
    parser.add_argument('--balance_classes', type=bool, default=False,
                        help='perform class balancing.')
    parser.add_argument('--mixup_alpha', type=float, default=None,
                        help='alpha parameter for MixUp training. \
                        if set performs MixUp, otherwise does not.')
    parser.add_argument('--unlabeled_counts', type=str, default=None,
                        help='path to unlabeled data [Cells, Genes]. \
                       [npy, csv, h5ad, loom]. \
                       if provided, uses interpolation consistency training.')
    parser.add_argument('--unlabeled_genes', type=str, default=None,
                        help='path to gene names for the unlabeled data.\
                       if not provided, assumes same as `input_counts`.')
    parser.add_argument(
        '--unlabeled_domain',
        type=str,
        help='path to a CSV of integer domain labels for each data point in `unlabeled_counts`.',
        required=False,
        default=None,
    )
    parser.add_argument('--unsup_max_weight', type=float, default=2.,
                        help='maximum weight for the unsupervised component of IC training.')
    parser.add_argument('--unsup_mean_teacher', action='store_true',
                        help='use a mean teacher for IC training.')
    parser.add_argument(
        '--ssl_method',
        type=str,
        default='mixmatch',
        help='semi-supervised learning method to use. {"mixmatch", "ict"}.',
    )
    parser.add_argument(
        '--ssl_config',
        type=str,
        default=None,
        help='path to a YAML configuration file of kwargs for the SSL method.'
    )
    args = parser.parse_args()

    print(args)
    print(parser.format_values())

    COMMANDS = [
        'train_tissue_independent',
        'train_tissue_dependent',
        'train_tissue_specific',
        'predict_cell_types',
    ]

    if args.command not in COMMANDS:
        raise ValueError('%s is not a valid command.' % args.command)

    #####################################
    # LOAD DATA
    #####################################

    X_raw = load_data(args.input_counts)

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
    lower_groups = np.unique(metadata[args.lower_group]).tolist()
    
    # load domain labels if applicable
    if args.input_domain_group is not None:
        if args.input_domain_group not in metadata.columns:
            msg = f'{args.input_domain_group} is not a column in `training_metadata`'
            raise ValueError(msg)
        else:
            input_domain = np.array(metadata[args.input_domain_group])
    else:
        input_domain = None

    # Load any provided unlabeled data for semi-supervised learning
    if args.unlabeled_counts is not None:
        unlabeled_counts = load_data(args.unlabeled_counts)
        print('%d cells, %d genes in unlabeled data.' % unlabeled_counts.shape)
        
        # parse any semi-supervised learning specific parameters
        if args.ssl_config is not None:
            print(f'Loading Semi-Supervised Learning parameters for {args.ssl_method}')
            with open(args.ssl_config, 'r') as f:
                ssl_kwargs = yaml.load(f, Loader=yaml.Loader)
            print('SSL kwargs:')
            for k, v in ssl_kwargs.items():
                print(f'{k}\t\t:\t\t{v}')
            print()
        else:
            ssl_kwargs = {}
        
    else:
        unlabeled_counts = None
        ssl_kwargs = {}

    if args.unlabeled_genes is not None and unlabeled_counts is not None:
        # Contruct a matrix using the unlabeled counts where columns
        # correspond to the same gene in `input_counts`.
        print('Subsetting unlabeled counts to genes used for training...')
        unlabeled_genes = np.loadtxt(
            args.unlabeled_genes,
            delimiter=',',
            dtype='str',
        )
        unlabeled_counts = utils.build_classification_matrix(
            X=unlabeled_counts,
            model_genes=genes_to_use,
            sample_genes=unlabeled_genes,
        )
        if args.unlabeled_domain is not None:
            unlabeled_domain = np.loadtxt(
                args.unlabeled_domain,
            ).astype(np.int)
        else:
            unlabeled_domain = None
    else:
        unlabeled_domain = None

    # prepare output paths
    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)

    sub_dirs = [
        'tissues', 
        'tissue_independent_no_dropout',
        'tissue_dependent', 
        'tissue_ind_class_optimums',
    ]
    for sd in sub_dirs:
        if not os.path.exists(osp.join(args.out_path, sd)):
            os.mkdir(osp.join(args.out_path, sd))

    #####################################
    # TISSUE INDEPENDENT CLASSIFIERS
    #####################################

    if args.command == 'train_tissue_independent':
        train_tissue_independent_cv(
            X,
            metadata,
            osp.join(args.out_path, 'tissue_independent'),
            balanced_classes=args.balance_classes,
            weighted_classes=args.weight_classes,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            init_dropout=args.init_dropout,
            lower_group=args.lower_group,
            n_hidden=args.n_hidden,
            n_layers=args.n_layers,
            lr=args.lr,
            optimizer_name=args.optimizer,
            weight_decay=args.weight_decay,
            residual=args.residual,
            track_running_stats=args.track_running_stats,
            mixup_alpha=args.mixup_alpha,
            unlabeled_counts=unlabeled_counts,
            unsup_max_weight=args.unsup_max_weight,
            unsup_mean_teacher=args.unsup_mean_teacher,
            ssl_method=args.ssl_method,
            ssl_kwargs=ssl_kwargs,
            input_domain=input_domain,
            unlabeled_domain=unlabeled_domain,
        )

    #####################################
    # TISSUE DEPENDENT CLASSIFIERS
    #####################################

    if args.command == 'train_tissue_dependent':
        train_tissue_dependent_cv(
            X,
            metadata,
            osp.join(args.out_path, 'tissue_dependent'),
            balanced_classes=args.balance_classes,
            weighted_classes=args.weight_classes,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            upper_group=args.upper_group,
            lower_group=args.lower_group,
            n_hidden=args.n_hidden,
            n_layers=args.n_layers,
            lr=args.lr,
            optimizer_name=args.optimizer,
            weight_decay=args.weight_decay,
            residual=args.residual,
            track_running_stats=args.track_running_stats,
            mixup_alpha=args.mixup_alpha,
            unlabeled_counts=unlabeled_counts,
            unsup_max_weight=args.unsup_max_weight,
            unsup_mean_teacher=args.unsup_mean_teacher,
            ssl_method=args.ssl_method,
            ssl_kwargs=ssl_kwargs,
            input_domain=input_domain,
            unlabeled_domain=unlabeled_domain,
        )

    #####################################
    # TISSUE SPECIFIC CLASSIFIERS
    #####################################

    if args.command == 'train_tissue_specific':

        tissue_specific_bal = {
            k: False for k in set(metadata[args.upper_group])}
        if args.balance_classes:
            print(args.balance_classes)
            for k in tissue_specific_bal:
                tissue_specific_bal[k] = True
            print('Balancing classes on a tissue specific basis')
            print(tissue_specific_bal)

        train_tissue_specific_cv(
            X,
            metadata,
            osp.join(args.out_path, 'tissues'),
            balanced_classes=tissue_specific_bal,
            weighted_classes=args.weight_classes,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            upper_group=args.upper_group,
            lower_group=args.lower_group,
            n_hidden=args.n_hidden,
            n_layers=args.n_layers,
            lr=args.lr,
            optimizer_name=args.optimizer,
            weight_decay=args.weight_decay,
            residual=args.residual,
            track_running_stats=args.track_running_stats,
            unlabeled_counts=unlabeled_counts,
            unsup_max_weight=args.unsup_max_weight,
            unsup_mean_teacher=args.unsup_mean_teacher,
            ssl_method=args.ssl_method,
            input_domain=input_domain,
            unlabeled_domain=unlabeled_domain,
         )

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
            sample_genes=gene_names,
        )

        predict_cell_types(
            X,
            model_path=args.model_path,
            out_path=args.out_path,
            lower_group_labels=lower_groups,
            n_hidden=args.n_hidden,
            n_layers=args.n_layers,
            residual=args.residual,
        )


#########################################################
# __main__
#########################################################


if __name__ == '__main__':

    main()
