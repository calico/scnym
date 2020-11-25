"""
Classify cell identities using scNym

scnym_api() is the main API endpoint for users.
This function allows for training and prediction using scnym_train() 
and scnym_predict(). Both of these functions will be infrequently 
accessed by users.

get_pretrained_weights() is a wrapper function that downloads pretrained
weights from our cloud storage bucket.
atlas2target() downloads preprocessed reference datasets and concatenates
them onto a user supplied target dataset.
"""
from typing import Optional, Union, List
from anndata import AnnData
import scanpy as sc
import numpy as np
import pandas as pd
import torch
import os
import os.path as osp
import copy
import pickle
import warnings
import itertools
import pprint
# for fetching pretrained weights, all in standard lib
import requests
import json
import urllib
# for data splits
from sklearn.model_selection import StratifiedKFold
# from scnym
from . import utils
from . import model
from . import main
from . import predict
from . import dataprep

# Define constants

TEST_URL = 'https://storage.googleapis.com/calico-website-mca-storage/kang_2017_stim_pbmc.h5ad'
WEIGHTS_JSON = 'https://storage.googleapis.com/calico-website-scnym-storage/link_tables/pretrained_weights.json'
REFERENCE_JSON = 'https://storage.googleapis.com/calico-website-scnym-storage/link_tables/cell_atlas.json'

ATLAS_ANNOT_KEYS = {
    'human': 'celltype',
    'mouse': 'cell_ontology_class',
    'rat': 'cell_ontology_class',
}

TASKS = (
    'train',
    'predict',
)

# Define configurations

CONFIGS = {
    'default' : {
        'n_epochs': 100,
        'patience': 40,
        'lr': 1.0,
        'optimizer_name': 'adadelta',
        'weight_decay': 1e-4,
        'batch_size': 256,
        'mixup_alpha': 0.3,
        'unsup_max_weight': 1.,
        'unsup_mean_teacher': False,
        'ssl_method': 'mixmatch',
        'ssl_kwargs': {
            'augment_pseudolabels': False,
            'augment': 'log1p_drop',
            'unsup_criterion': 'mse',
            'n_augmentations': 1,
            'T': 0.5,
            'ramp_epochs': 100,
            'burn_in_epochs': 0,
            'dan_criterion': True,
            'dan_ramp_epochs': 20,
            'dan_max_weight': 0.1,
            'min_epochs': 20,
        },
        'model_kwargs' : {
            'n_hidden': 256,
            'n_layers': 2,
            'init_dropout': 0.0,
            'residual': False,
        },
        'tensorboard': False,
    },
}

CONFIGS['no_new_identity'] = copy.deepcopy(CONFIGS['default'])
CONFIGS['no_new_identity']['description'] = (
    'Train scNym models with MixMatch and a domain adversary, assuming no new cell types in the target data.'
)

CONFIGS['new_identity_discovery'] = copy.deepcopy(CONFIGS['default'])
CONFIGS['new_identity_discovery']['ssl_kwargs']['pseudolabel_min_confidence'] = 0.9
CONFIGS['new_identity_discovery']['ssl_kwargs']['dan_use_conf_pseudolabels'] = True
CONFIGS['new_identity_discovery']['description'] = (
    'Train scNym models with MixMatch and a domain adversary, using pseudolabel thresholding to allow for new cell type discoveries.'
)

CONFIGS['no_dan'] = copy.deepcopy(CONFIGS['default'])
CONFIGS['no_dan']['ssl_kwargs']['dan_max_weight'] = 0.0
CONFIGS['no_dan']['ssl_kwargs']['dan_ramp_epochs'] = 1
CONFIGS['no_dan']['description'] = (
    'Train scNym models with MixMatch but no domain adversary. May be useful if class imbalance is very large.'
)

CONFIGS['no_ssl'] = copy.deepcopy(CONFIGS['default'])
CONFIGS['no_ssl']['ssl_kwargs']['dan_max_weight'] = 0.0
CONFIGS['no_ssl']['ssl_kwargs']['dan_ramp_epochs'] = 1
CONFIGS['no_ssl']['ssl_kwargs']['unsup_max_weight'] = 0.0
CONFIGS['no_ssl']['description'] = (
    'Train scNym models with MixMatch but no domain adversary. May be useful if class imbalance is very large.'
)


UNLABELED_TOKEN = 'Unlabeled'


def scnym_api(
    adata: AnnData,
    task: str='train',
    groupby: str=None,
    out_path: str='./scnym_outputs',
    trained_model: str=None,
    config: Union[dict, str]='new_identity_discovery',
    key_added: str='scNym',
    copy: bool=False,
) -> Optional[AnnData]:
    """
    scNym: Semi-supervised adversarial neural networks for 
    single cell classification [Kimmel2020]_.
    
    scNym is a cell identity classifier that transfers annotations from one
    single cell experiment to another. The model is implemented as a neural
    network that employs MixMatch semi-supervision and a domain adversary to
    take advantage of unlabeled data during training. scNym offers superior
    performance to many baseline single cell identity classification methods.
    
    Parameters
    ----------
    adata
        Annotated data matrix used for training or prediction.
        If `"scNym_split"` in `.obs_keys()`, uses the cells annotated 
        `"train", "val"` to select data splits.
    task
        Task to perform, either "train" or "predict".
        If "train", uses `adata` as labeled training data.
        If "predict", uses `trained_model` to infer cell identities for
        observations in `adata`.
    groupby
        Column in `adata.obs` that contains cell identity annotations.
        Values of `"Unlabeled"` indicate that a given cell should be used
        only as unlabeled data during training.
    out_path
        Path to a directory for saving scNym model weights and training logs.
    trained_model
        Path to the output directory of an scNym training run
        or a string specifying a pretrained model.
        If provided while `task == "train"`, used as an initialization. 
    config
        Configuration name or dictionary of configuration of parameters.
        Pre-defined configurations:
            "new_identity_discovery" - Default. Employs pseudolabel thresholding to 
            allow for discovery of new cell identities in the target dataset using
            scNym confidence scores.
            "no_new_identity" - Assumes all cells in the target data belong to one
            of the classes in the training data. Recommended to improve performance
            when this assumption is valid.
    key_added
        Key added to `adata.obs` with scNym predictions if `task=="predict"`.            
    copy
        copy the AnnData object before predicting cell types.
            
    Returns
    -------
    Depending on `copy`, returns or updates `adata` with the following fields.

    `X_scnym` : :class:`~numpy.ndarray`, (:attr:`~anndata.AnnData.obsm`, shape=(n_samples, n_hidden), dtype `float`)
        scNym embedding coordinates of data.
    `scNym` : (`adata.obs`, dtype `str`)
        scNym cell identity predictions for each observation.
    `scNym_train_results` : :class:`~dict`, (:attr:`~anndata.AnnData.uns`)
        results of scNym model training.
    
    Examples
    --------
    >>> import scanpy as sc    
    >>> from scnym.api import scnym_api, atlas2target
    
    **Loading Data and preparing labels**
    
    >>> adata = sc.datasets.kang17()
    >>> target_bidx = adata.obs['stim']=='stim'
    >>> adata.obs['cell'] = np.array(adata.obs['cell'])
    >>> adata.obs.loc[target_bidx, 'cell'] = 'Unlabeled'
    
    **Train an scNym model**
    
    >>> scnym_api(
    ...   adata=adata,
    ...   task='train',
    ...   groupby='clusters',
    ...   out_path='./scnym_outputs',
    ...   config='no_new_identity',
    ... )
    
    **Predict cell identities with the trained scNym model**
    
    >>> path_to_model = './scnym_outputs/'
    >>> scnym_api(
    ...   adata=adata,
    ...   task='predict',
    ...   groupby='scNym',
    ...   trained_model=path_to_model,
    ...   config='no_new_identity',
    ... )
    
    **Perform semi-supervised training with an atlas**
    
    >>> joint_adata = atlas2target(
    ...   adata=adata,
    ...   species='mouse',
    ...   key_added='annotations',
    ... )
    >>> scnym_api(
    ...   adata=joint_adata,
    ...   task='train',
    ...   groupby='annotations',
    ...   out_path='./scnym_outputs',
    ...   config='no_new_identity',
    ... )
    """
    if task not in TASKS:
        msg = f'{task} is not a valid scNym task.\n'
        msg += f'must be one of {TASKS}'
        raise ValueError(msg)
            
    # check configuration arguments and choose a config
    if type(config) == str:
        if config not in CONFIGS.keys():
            msg = f'{config} is not a predefined configuration.\n'
            msg += f'must be one of {CONFIGS.keys()}.'
            raise ValueError(msg)
        else:
            config = CONFIGS[config]
    elif type(config) != dict:
        msg = f'`config` was a {type(config)}, must be dict or str.'
        raise TypeError(msg)
    else:
        # config is a dictionary of parameters
        # add or update default parameters based on these
        dconf = CONFIGS['default']
        for k in config.keys():
            dconf[k] = config[k]
        config = dconf

    # check for CUDA
    if torch.cuda.is_available():
        print('CUDA compute device found.')
    else:
        print('No CUDA device found.')
        print('Computations will be performed on the CPU.')
        print('Add a CUDA compute device to improve speed dramatically.\n')
    
    if not osp.exists(out_path):
        os.makedirs(out_path, exist_ok=True)
    
    # add args to `config`
    config['out_path']  = out_path
    config['groupby']   = groupby
    config['key_added'] = key_added
    config['trained_model'] = trained_model
    
    ################################################
    # check that there are no duplicate genes in the input object
    ################################################
    n_genes = adata.shape[1]
    n_unique_genes = len(np.unique(adata.var_names))
    if n_genes != n_unique_genes:
        msg = 'Duplicate Genes Error\n'
        msg += 'Not all genes passed to scNym were unique.\n'
        msg += f'{n_genes} genes are present but only {n_unique_genes} unique genes were detected.\n'
        msg += 'Please use unique gene names in your input object.\n'
        msg += 'This can be achieved by running `adata.var_names_make_unique()`'
        raise ValueError(msg)
        
    ################################################
    # check that `adata.X` are log1p(CPM) counts
    ################################################
    # we can't directly check if cells were normalized to CPM because
    # users may have filtered out genes *a priori*, so the cell sum
    # may no longer be ~= 1e6.
    # however, we can check that our assumptions about log normalization
    # are true.
    
    # check that the min/max are within log1p(CPM) range
    x_max = np.max(adata.X) > np.log1p(1e6)
    x_min = np.min(adata.X) < 0.
    
    # check to see if a user accidently provided raw counts
    if type(adata.X) == np.ndarray:
        int_counts = np.equal(np.mod(adata.X, 1), 0)
    else:
        int_counts = np.all(np.equal(np.mod(adata.X.data, 1), 0))    
    
    if x_max or x_min or int_counts:
        msg = 'Normalization error\n'
        msg += '`adata.X` does not appear to be log(CountsPerMillion+1) normalized data.\n'
        msg += 'Please replace `adata.X` with log1p(CPM) values.\n'
        msg += '>>> # starting from raw counts in `adata.X`\n'
        msg += '>>> sc.pp.normalize_total(adata, target_sum=1e6))\n'
        msg += '>>> sc.pp.log1p(adata)'
        raise ValueError(msg)
        
    ################################################
    # check inputs and launch the appropriate task
    ################################################
    
    if task == 'train':
        # pass parameters to training routine
        if groupby not in adata.obs.columns:
            msg = f'{groupby} is not a variable in `adata.obs`'
            raise ValueError(msg)
            
        scnym_train(
            adata=adata,
            config=config,
        )
    else:
        # check that a pre-trained model was specified or 
        # provided for prediction
        if trained_model is None:
            msg = 'must provide a path to a trained model for prediction.'
            raise ValueError(msg)
        if not os.path.exists(trained_model) and 'pretrained_' not in trained_model:
            msg = 'path to the trained model does not exist.'
            raise FileNotFoundError(msg)
        # predict identities
        config['model_weights'] = trained_model
        scnym_predict(
            adata=adata,
            config=config,
        )
    return


def scnym_train(
    adata: AnnData,
    config: dict,
) -> None:
    '''Train an scNym model. 
    
    Parameters
    ----------
    adata : AnnData
        [Cells, Genes] experiment containing annotated
        cells to train on.
    config : dict
        configuration options.
    
    Returns
    -------
    None.
    Saves model outputs to `config["out_path"]` and adds model results
    to `adata.uns["scnym_train_results"]`.
    
    Notes
    -----
    This method should only be directly called by advanced users.
    Most users should use `scnym_api`.
    
    See Also
    --------
    scnym_api
    '''
    # determine if unlabeled examples are present
    n_unlabeled = np.sum(
        adata.obs[config['groupby']] == UNLABELED_TOKEN
    )
    if n_unlabeled == 0:
        print('No unlabeled data was found.')
        print(f'Did you forget to set some examples as `"{UNLABELED_TOKEN}"`?')
        print('Proceeding with purely supervised training.')
        print()
        
        unlabeled_counts = None
        unlabeled_genes  = None
        
        X = utils.get_adata_asarray(adata)
        y = pd.Categorical(
            np.array(adata.obs[config['groupby']]),
            categories=np.unique(adata.obs[config['groupby']]),
        ).codes
        class_names = np.unique(adata.obs[config['groupby']])
    else:
        print(f'{n_unlabeled} unlabeled observations found.')
        print('Using unlabeled data as a target set for semi-supervised, adversarial training.')
        print()
        
        target_bidx = adata.obs[config['groupby']] == UNLABELED_TOKEN
        
        train_adata = adata[~target_bidx, :]
        target_adata = adata[target_bidx, :]
        
        print('training examples: ', train_adata.shape)
        print('target   examples: ', target_adata.shape)
        
        X = utils.get_adata_asarray(train_adata)
        y = pd.Categorical(
            np.array(train_adata.obs[config['groupby']]),
            categories=np.unique(train_adata.obs[config['groupby']]),
        ).codes
        unlabeled_counts = utils.get_adata_asarray(target_adata)
        class_names = np.unique(train_adata.obs[config['groupby']])
        
    print('X: ', X.shape)
    print('y: ', y.shape)
    
    if 'scNym_split' not in adata.obs_keys():
        # perform a 90/10 train test split
        traintest_idx = np.random.choice(
            X.shape[0],
            size=int(np.floor(0.9*X.shape[0])),
            replace=False
        )
        val_idx = np.setdiff1d(np.arange(X.shape[0]), traintest_idx)
    else:
        train_idx = np.where(
            train_adata.obs['scNym_split'] == 'train'
        )[0]
        test_idx = np.where(
            train_adata.obs['scNym_split'] == 'test',
        )[0]
        val_idx = np.where(
            train_adata.obs['scNym_split'] == 'val'
        )[0]
        
        if len(train_idx) < 100 or len(test_idx) < 10 or len(val_idx) < 10:
            msg = 'Few samples in user provided data split.\n'
            msg += f'{len(train_idx)} training samples.\n'
            msg += f'{len(test_idx)} testing samples.\n'
            msg += f'{len(val_idx)} validation samples.\n'
            msg += 'Halting.'
            raise RuntimeError(msg)
        # `fit_model()` takes a tuple of `traintest_idx`
        # as a training index and testing index pair.
        traintest_idx = (
            train_idx,
            test_idx,
        )
        
    # check if domain labels were manually specified
    if config.get('domain_groupby', None) is not None:
        domain_groupby = config['domain_groupby']
        # check that the column actually exists
        if domain_groupby not in adata.obs.columns:
            msg = f'no column {domain_groupby} exists in `adata.obs`.\n'
            msg += 'if domain labels are specified, a matching column must exist.'
            raise ValueError(msg)
        # get the label indices as unique integers using pd.Categorical
        # to code each unique label with an int
        domains = np.array(
            pd.Categorical(
                adata.obs[domain_groupby],
                categories=np.unique(adata.obs[domain_groupby]),
            ).codes,
            dtype=np.int32,
        )
        # split domain labels into source and target sets for `fit_model`
        input_domain = domains[~target_bidx]
        unlabeled_domain = domains[target_bidx]
        print('Using user provided domain labels.')
        n_source_doms = len(np.unique(input_domain))
        n_target_doms = len(np.unique(unlabeled_domain))
        print(
            f'Found {n_source_doms} source domains and {n_target_doms} target domains.'
        )
    else:
        # no domains manually supplied, providing `None` to `fit_model`
        # will treat source data as one domain and target data as another
        input_domain = None
        unlabeled_domain = None
        
    # check if pre-trained weights should be used to initialize the model
    if config['trained_model'] is None:
        pretrained = None
    elif 'pretrained_' in config['trained_model']:
        msg = 'pretrained model fetching is not supported for training.'
        raise NotImplementedError(msg)
    else:
        # setup a prediction model
        pretrained = osp.join(
            config['trained_model'],
            '00_best_model_weights.pkl',
        )
        if not osp.exists(pretrained):
            msg = f'{pretrained} file not found.'
            raise FileNotFoundError(msg)
        
    acc, loss = main.fit_model(
        X=X,
        y=y,
        traintest_idx=traintest_idx,
        val_idx=val_idx,
        batch_size=config['batch_size'],
        n_epochs=config['n_epochs'],
        lr=config['lr'],
        optimizer_name=config['optimizer_name'],
        weight_decay=config['weight_decay'],
        ModelClass=model.CellTypeCLF,
        out_path=config['out_path'],
        mixup_alpha=config['mixup_alpha'],
        unlabeled_counts=unlabeled_counts,
        input_domain=input_domain,
        unlabeled_domain=unlabeled_domain,
        unsup_max_weight=config['unsup_max_weight'],
        unsup_mean_teacher=config['unsup_mean_teacher'],
        ssl_method=config['ssl_method'],
        ssl_kwargs=config['ssl_kwargs'],
        pretrained=pretrained,
        patience=config.get('patience', None),
        save_freq=config.get('save_freq', None),
        tensorboard=config.get('tensorboard', False),
        **config['model_kwargs'],
    )
    
    # add the final model results to `adata`
    results = {
        'model_path': osp.realpath(osp.join(config['out_path'], '00_best_model_weights.pkl')),
        'final_acc': acc,
        'final_loss': loss,
        'n_genes': adata.shape[1],
        'n_cell_types': len(np.unique(y)),
        'class_names': class_names,
        'gene_names': adata.var_names.tolist(),
        'model_kwargs': config['model_kwargs'],
        'traintest_idx': traintest_idx,
        'val_idx': val_idx,
    }
    assert osp.exists(results['model_path'])
    
    adata.uns['scNym_train_results'] = results
    
    # save the final model results to disk
    train_results_path = osp.join(
        config['out_path'], 
        'scnym_train_results.pkl',
    )

    with open(train_results_path, 'wb') as f:
        pickle.dump(
            results,
            f
        )
    return


@torch.no_grad()
def scnym_predict(
    adata: AnnData,
    config: dict,
) -> None:
    '''Predict cell identities using an scNym model.
    
    Parameters
    ----------
    adata : AnnData
        [Cells, Genes] experiment containing annotated
        cells to train on.
    config : dict
        configuration options.
        
    Returns
    -------
    None. Adds `adata.obs[config["key_added"]]` and `adata.obsm["X_scnym"]`.
    
    Notes
    -----
    This method should only be directly called by advanced users.
    Most users should use `scnym_api`.
    
    See Also
    --------
    scnym_api    
    '''
    # check if a pretrained model was requested
    if 'pretrained_' in config['trained_model']:
        msg = 'Pretrained Request Error\n'
        msg += 'Pretrained weights are no longer supported in scNym.\n'
        raise NotImplementedError(msg)
#         species = _get_pretrained_weights(
#             trained_model=config['trained_model'],
#             out_path=config['out_path'],
#         )
#         print(f'Successfully downloaded pretrained model for {species}.')
#         config['trained_model'] = config['out_path']
    
    # load training parameters
    with open(
        osp.join(config['trained_model'], 'scnym_train_results.pkl'), 
        'rb',
    ) as f:
        results = pickle.load(f)
        
    # setup a prediction model
    model_weights_path = osp.join(
        config['trained_model'],
        '00_best_model_weights.pkl',
    )
    
    P = predict.Predicter(
        model_weights=model_weights_path,
        n_genes=results['n_genes'],
        n_cell_types=results['n_cell_types'],
        labels=results['class_names'],
        **config['model_kwargs'],
    )
    n_cell_types = results["n_cell_types"]
    n_genes      = results["n_genes"]
    print(f'Loaded model predicting {n_cell_types} classes from {n_genes} features')
    print(results['class_names'])
    
    # Generate a classification matrix
    print('Building a classification matrix...')
    X_raw = utils.get_adata_asarray(adata)
    X = utils.build_classification_matrix(
        X=X_raw,
        model_genes=np.array(results['gene_names']),
        sample_genes=np.array(adata.var_names),
    )
    
    # Predict cell identities
    print('Predicting cell types...')
    pred, names, prob = P.predict(
        X,
        output='prob',
    )
    
    prob = pd.DataFrame(
        prob, 
        columns=results['class_names'], 
        index=adata.obs_names,
    )
    
    # Extract model embeddings
    print('Extracting model embeddings...')
    ds = dataprep.SingleCellDS(X=X, y=np.zeros(X.shape[0]))
    dl = torch.utils.data.DataLoader(
        ds, 
        batch_size=config['batch_size'], 
        shuffle=False,
    )

    model = P.models[0]
    lz_02 = torch.nn.Sequential(
        *list(list(model.modules())[0].children())[1][:-1]
    )
    
    embeddings = []
    for data in dl:
        input_ = data['input']
        input_ = input_.to(device=next(model.parameters()).device)
        z = lz_02(input_)
        embeddings.append(z.detach().cpu())
    Z = torch.cat(embeddings, 0)
    
    # Store results in the anndata object
    adata.obs[config['key_added']] = names
    adata.obs[config['key_added'] + '_confidence'] = np.max(prob, axis=1)
    adata.uns['scNym_probabilities'] = prob
    adata.obsm['X_scnym'] = Z.numpy()
    
    return


def _get_pretrained_weights(
    trained_model: str,
    out_path: str,
) -> str:
    '''Given the name of a set of pretrained model weights,
    fetch weights from GCS and return the model state dict.
    
    Parameters
    ----------
    trained_model : str
        the name of a pretrained model to use, formatted as
        "pretrained_{species}". 
        species should be one of {"human", "mouse", "rat"}.
    out_path : str
        path for saving model weights and outputs.
    
    Returns
    -------
    species : str
        species parsed from the trained model name.
    Saves "{out_path}/00_best_model_weights.pkl" and 
    "{out_path}/scnym_train_results.pkl".
    
    Notes
    -----
    Requires an internet connection to download pre-trained weights.
    '''
    # check that the trained_model argument is valid
    if 'pretrained_' not in trained_model:
        msg = 'pretrained model names must contain `"pretrained_"`'
        raise ValueError(msg)
    
    species = trained_model.split('pretrained_')[1]
    
    # download a table of available pretrained models
    try:
        pretrained_weights_dict = json.loads(
            requests.get(WEIGHTS_JSON).text
        )
    except requests.exceptions.ConnectionError:
        print('Could not download pretrained weighs listing from:')
        print(f'\t{WEIGHTS_JSON}')
        print('Loading pretrained model failed.')

    # check that the species specified has pretrained weights
    if species not in pretrained_weights_dict.keys():
        msg = f'pretrained weights not available for {species}.'
        raise ValueError(species)
    
    # get pretrained weights
    path_for_weights =  osp.join(out_path, f'00_best_model_weights.pkl')
    urllib.request.urlretrieve(
        pretrained_weights_dict[species],
        path_for_weights,
    )
    
    # load model parameters
    model_params = {}
    urllib.request.urlretrieve(
        pretrained_weights_dict['model_params'][species]['gene_names'],
        osp.join(out_path, 'pretrained_gene_names.csv'),
    )
    urllib.request.urlretrieve(
        pretrained_weights_dict['model_params'][species]['class_names'],
        osp.join(out_path, 'pretrained_class_names.csv'),
    )
    model_params['gene_names'] = np.loadtxt(
        osp.join(out_path, 'pretrained_gene_names.csv'),
        delimiter=',',
        dtype='str',
    )
    model_params['class_names'] = np.loadtxt(
        osp.join(out_path, 'pretrained_class_names.csv'),
        delimiter=',',
        dtype='str',
    )
    model_params['n_genes'] = len(model_params['gene_names'])
    model_params['n_cell_types'] = len(model_params['class_names'])
    
    # save model parameters to a results file in the output dir
    path_for_results = f'{out_path}/scnym_train_results.pkl'
    with open(path_for_results, 'wb') as f:
        pickle.dump(model_params, f)
        
    # check that files are present
    if not osp.exists(path_for_weights):
        raise FileNotFoundError(path_for_weights)
    if not osp.exists(path_for_results):
        raise FileNotFoundError(path_for_results)

    return species


def atlas2target(
    adata: AnnData,
    species: str,
    key_added: str='annotations',
) -> AnnData:
    '''Download a preprocessed cell atlas dataset and
    append your new dataset as a target to allow for 
    semi-supervised scNym training.
    
    Parameters
    ----------
    adata : anndata.AnnData
        [Cells, Features] experiment to use as a target
        dataset.
        `adata.var_names` must be formatted as Ensembl gene
        names for the relevant species to match the atlas.
        e.g. `"Gapdh`" for mouse or `"GAPDH"` for human, rather
        than Ensembl gene IDs or another gene annotation.
        
    Returns
    -------
    joint_adata : anndata.AnnData
        [Cells, Features] experiment concatenated with a
        preprocessed cell atlas reference dataset.
        Annotations from the atlas are copied to `.obs[key_added]`
        and all cells in the target dataset `adata` are labeled
        with the special "Unlabeled" token.
        
    Examples
    --------
    >>> adata = sc.datasets.pbmc3k()
    >>> joint_adata = scnym.api.atlas2target(
    ...     adata=adata,
    ...     species='human',
    ...     key_added='annotations',
    ... )
    
    Notes
    -----
    Requires an internet connection to download reference datasets.    
    '''
    # download a directory of cell atlases
    try:
        reference_dict = json.loads(
            requests.get(REFERENCE_JSON).text
        )
    except requests.exceptions.ConnectionError:
        print('Could not download pretrained weighs listing from:')
        print(f'\t{REFERENCE_JSON}')
        print('Loading pretrained model failed.')
    
    # check that the species presented is available
    if species not in reference_dict.keys():
        msg = f'pretrained weights not available for {species}.'
        raise ValueError(species)
        
    # check that there are no gene duplications
    n_uniq_genes = len(np.unique(adata.var_names))
    if n_uniq_genes < len(adata.var_names):
        msg = f'{n_uniq_genes} unique features found, but {adata.shape[1]} features are listed.\n'
        msg += 'Please de-duplicate features in `adata` before joining with an atlas dataset.\n'
        msg += 'Consider `adata.var_names_make_unique()` or aggregating values for features with the same identifier.'
        raise ValueError(msg)

    # download the atlas of interest
    atlas = sc.datasets._datasets.read(
        sc.settings.datasetdir / f'atlas_{species}.h5ad',
        backup_url=reference_dict[species],
    )
    del atlas.raw
    
    # get the key used by the cell atlas
    atlas_annot_key = ATLAS_ANNOT_KEYS[species]
    
    # copy atlas annotations to the specified column
    atlas.obs[key_added] = np.array(atlas.obs[atlas_annot_key])
    atlas.obs['scNym_dataset'] = 'atlas_reference'
    
    # label target data with "Unlabeled"
    adata.obs[key_added] = 'Unlabeled'
    adata.obs['scNym_dataset'] = 'target'
    
    # check that at least some genes overlap between the atlas
    # and the target data
    FEW_GENES = 100
    n_overlapping_genes = len(np.intersect1d(adata.var_names, atlas.var_names))
    if n_overlapping_genes == 0:
        msg = 'No genes overlap between the target data `adata` and the atlas.\n'
        msg += 'Genes in the atlas are named using Ensembl gene symbols (e.g. `"Gapdh"`).\n'
        msg += 'Ensure `adata.var_names` also uses gene symbols.'
        raise RuntimeError(msg)
    elif n_overlapping_genes < FEW_GENES:
        msg = f'Only {n_overlapping_genes} overlapping genes were found between the target and atlas.\n'
        msg += 'Ensure your target dataset `adata.var_names` are Ensembl gene names.\n'
        msg += 'Continuing with transer, but performance is likely to be poor.'
        warnings.warn(msg)
    else:
        msg = f'{n_overlapping_genes} overlapping genes found between the target and atlas data.'
        print(msg)
    
    # join the target and atlas data
    joint_adata = atlas.concatenate(
        adata, 
        join='inner',
    )
        
    return joint_adata


def list_configs():
    for k in CONFIGS.keys():
        print(f'name: {k}')
        print('\t'+CONFIGS[k]['description'])
    return


def _get_keys_and_list(d: dict) -> (List[list], List[list]):
    '''Get a set of keys mapping to a list in a
    nested dictionary structure and the list value.
    
    Parameters
    ----------
    d : dict
        a nested dictionary structure where all terminal
        values are lists.
    
    Returns
    -------
    keys : List[list]
        sequential keys required to access a set of 
        associated terminal values.
        mapped by index to `values`.
    values : List[list]
        lists of terminal values, each accessed by the
        set of `keys` with a matching index from `d`.
    '''
    accession_keys = []
    associated_values = []
    for k in d.keys():
        if type(d[k])==dict:
            # the value is nested, recurse
            keys, values = _get_keys_and_list(d[k])
            keys = [[k,]+x for x in keys]
        else:
            keys = [[k],]
            values = [d[k]]
        
        for i in range(len(values)):
            accession_keys.append(keys[i])
            associated_values.append(values[i])

    return accession_keys, associated_values


def _updated_nested(d: dict, keys: list, value: list) -> dict:
    '''Updated the values in a dictionary with multiple nested levels.
    
    Parameters
    ----------
    d : dict
        multilevel dictionary.
    keys : list
        sequential keys specifying a value to update
    value : list
        new value to use in the update.
    
    Returns
    -------
    d : dict
        updated dictionary.
    '''
    if type(d.get(keys[0], None)) == dict:
        # multilevel, recurse
        _updated_nested(d[keys[0]], keys[1:], value)
    else: 
        d[keys[0]] = value
    return


def split_data(
    adata: AnnData,
    groupby: str,
    n_splits: int,
) -> None:
    '''Split data using a stratified k-fold.
    
    Parameters
    ----------
    adata : anndata.AnnData
        [Cells, Genes] experiment.
    groupby : str
        annotation column in `.obs`.
        used for stratification.
    n_splits : int
        number of train/test/val splits to perform for tuning.
        performs at least 5-fold splitting and uses a subset of
        the folds if `n_splits < 5`.
        
    Returns
    -------
    None. Adds `f"scNym_split_{n}"` to `adata.obs` for all `n`
    in `[0, n_splits)`.
    '''
    # generate cross val splits
    cv = StratifiedKFold(
        n_splits=max(5, n_splits), 
        shuffle=True,
    )
    split_indices = list(
        cv.split(adata.X, adata.obs[groupby])
    )

    for split_number, train_test in enumerate(split_indices):

        train_idx = train_test[0]
        testval_idx = train_test[1]

        test_idx = np.random.choice(
            testval_idx,
            size=int(np.ceil(len(testval_idx)/2)),
            replace=False,
        )
        val_idx = np.setdiff1d(
            testval_idx,
            test_idx,
        )

        # these tokens are recognized by `api.scnym_train`
        adata.obs[f'scNym_split_{split_number}'] = 'ERROR'
        adata.obs.loc[
            adata.obs_names[train_idx], f'scNym_split_{split_number}'
        ] = 'train'
        adata.obs.loc[
            adata.obs_names[test_idx], f'scNym_split_{split_number}'
        ] = 'test'    
        adata.obs.loc[
            adata.obs_names[val_idx], f'scNym_split_{split_number}'
        ] = 'val'

    return


def _circular_train(
    search_config: dict,
    params: tuple,
    adata: AnnData,
    groupby: str,
    out_path: str,
    accession_keys: List[list],
    hold_out_only: bool,
    groupby_eval: str,
) -> pd.DataFrame:
    '''
    Perform a circular training loop for a parameter set.
    
    Parameters
    ----------
    search_config : tuple
        configuration for parameter search.
    params : tuple
        search parameter values
    adata : anndata.AnnData
        [Cells, Genes] experiment for optimization.
    groupby : str
        annotation column in `.obs`.
    accession_keys : List[list]
        sequential keys required to access a set of 
        associated terminal values.
        mapped by index to `values`.
    hold_out_only : bool
        evaluate the circular accuracy only on a held-out set of 
        training data, not used in the training of the first
        source -> target model.        

    Returns
    -------
    search_df : pd.DataFrame
        [1, (params,) + (acc,)]
    search_config : dict
        adjusted configuration file for this parameter search.
    '''
    search_number = search_config['search_number']
    split_number  = search_config['split_number']
    # fit the source2target
    s2t_out_path = osp.join(out_path, f'search_{search_number:04}_split_{split_number:04}_source2target')
    adata = adata.copy()
    
    print('\n>>>\nTraining source2target model\n>>>\n')
    scnym_api(
        adata=adata,
        groupby=groupby,
        task='train',
        out_path=s2t_out_path,
        config=search_config,
    )
    
    # load the hold out test acc
    with open(osp.join(s2t_out_path, 'scnym_train_results.pkl'), 'rb') as f:
        s2t_res = pickle.load(f)
        s2t_source_test_acc = s2t_res['final_acc']

    print('\n>>>\nPredicting with source2target model\n>>>\n')
    # predict on the target set
    scnym_api(
        adata=adata,
        task='predict',
        trained_model=s2t_out_path,
        config=search_config,
    )

    # invert the problem -- train on the new labels
    circ_adata = adata.copy()
    circ_adata.obs[groupby] = adata.obs['scNym']
    circ_adata.obs.drop(columns=['scNym'], inplace=True)
    # set the training data as unlabeled, leaving labels only on the target data
    circ_adata.obs.loc[adata.obs[groupby]!=UNLABELED_TOKEN, groupby] = UNLABELED_TOKEN

    # fit a new model
    t2s_out_path = osp.join(out_path, f'search_{search_number:04}_split_{split_number:04}_target2source')
    
    print('\n>>>\nTraining target2source model\n>>>\n')
    
    scnym_api(
        adata=circ_adata,
        groupby=groupby,
        task='train',
        out_path=t2s_out_path,
        config=search_config,
    )

    # predict with new model
    print('\n>>>\nPredicting with target2source model\n>>>\n')    
    scnym_api(
        adata=circ_adata,
        task='predict',
        trained_model=t2s_out_path,
        config=search_config,
    )

    # evaluate the model
    samples_bidx = adata.obs[groupby]!='Unlabeled'
    samples_bidx = (
        samples_bidx & (adata.obs['scNym_split']=='val') if hold_out_only else samples_bidx
    )
    y_true = np.array(adata.obs[groupby])[samples_bidx]
    y_pred = np.array(circ_adata.obs['scNym'])[samples_bidx]

    n_correct = np.sum(y_true==y_pred)
    n_total = len(y_true)
    acc = n_correct/n_total

    accession_keys_str = [
        '::'.join(x) for x in accession_keys
    ]
    search_df = pd.DataFrame(
        columns = accession_keys_str + ['acc'],
        index=[search_number],
    )
    search_df.loc[search_number] = params + (acc,)
    search_df['test_source_acc'] = s2t_source_test_acc
    
    if groupby_eval is not None:
        # compute the test accuracy in the target domain
        # here, we use the predictions made by the source2target
        # model stored in `adata.obs["scNym"]`.
        samples_bidx = adata.obs[groupby]=='Unlabeled'
        y_true = np.array(adata.obs[groupby_eval])[samples_bidx]
        y_pred = np.array(adata.obs['scNym'])[samples_bidx]
        n_correct = np.sum(y_true == y_pred)
        test_acc = n_correct/len(y_true)
        search_df['test_target_acc'] = 'None'
        search_df.loc[search_number, 'test_target_acc'] = test_acc
    
    search_df.to_csv(osp.join(t2s_out_path, 'result.csv'))

    return search_df


def scnym_tune(
    adata: AnnData,
    groupby: str,
    parameters: dict,
    search: str='grid',
    base_config: str='no_new_identity',
    n_points: int=100,
    out_path: str='./scnym_tune',
    hold_out_only: bool=True,
    groupby_eval: str=None,
    n_splits: int=1,
) -> (pd.DataFrame, dict):
    '''Perform hyperparameter tuning of an scNym model using 
    circular cross-validation.
    
    Parameters
    ----------
    adata : anndata.AnnData
        [Cells, Genes] experiment for optimization.
    groupby : str
        annotation column in `.obs`.
    parameters : dict
        key:List[value] pairs of parameters to use for 
        hyperparameter tuning.
    base_config : str
        one of {"no_new_identity", "new_identity_discovery"}.
        base configuration for model training that described
        default parameters, not explicitly provided in
        `parameters`.
    search : str
        {"grid", "random"} perform either a random or grid
        search over `parameters`.        
    n_points : int
        number of random points to search if `search == "random"`.
    out_path : str
        path for intermediary files during hyperparameter tuning.
    hold_out_only : bool
        evaluate the circular accuracy only on a held-out set of 
        training data, not used in the training of the first
        source -> target model.
    groupby_eval : str
        column in `adata.obs` containing ground truth labels
        for the "Unlabeled" dataset to use for evaluation.
    n_splits : int
        number of train/test/val splits to perform for tuning.
        performs at least 5-fold splitting and uses a subset of
        the folds if `n_splits < 5`.
        
    Returns
    -------
    tuning_results : pd.DataFrame
        [n_points, (parameters,) + (circ_acc, circ_loss)]
    best_parameter_set : dict
        a configuration describing the best parameter set tested.
        
    Examples
    --------
    >>> # `adata` contains labels in `.obs["annotations"]` where
    ... # the target dataset is labeled "Unlabeled"
    >>> tuning_results, best_parameters = scnym_tune(
    ...   adata=adata,
    ...   groupby="annotations",
    ...   parameters={
    ...     "weight_decay": [1e-6, 1e-5, 1e-4],
    ...     "unsup_max_weight": [0.1, 1., 10.],
    ...   },
    ...   base_config="no_new_identity",
    ...   search="grid",
    ...   out_path="./scnym_tuning",
    ...  n_splits=5,
    ... )
        
    Notes
    -----
    Circular/Reverse cross-validation evaluates the impact of hyperparameter
    selection in semi-supervised learning settings using the training data,
    training labels, and target data, but not the target labels.
    
    This is achieved by training a model :math:`f` on the training set, then 
    predicting "pseudolabels" for the target set.
    A second model :math:`g` is then trained on the target data and
    the associated pseudolabels.
    The model :math:`g` is used to predict labels for the *training* set.
    The accuracy of this "reverse" prediction is then used as an estimate
    of the effectiveness of a hyperparameter set.
    '''
    os.makedirs(out_path, exist_ok=True)
    
    # get the base configuration dict
    # configurations have one layer of nested dictionaries within
    config = CONFIGS.get(base_config, None)
    if config is None:
        msg = f'{base_config} is not a valid base configuration.'
        raise ValueError(msg)
        
    #################################################
    # get all possible combinations of parameters
    #################################################
    # `_get_keys_and_list` traverses a nested dictionary and
    # returns a List[list] of sequential keys to access each
    # item in `parameter_ranges`.
    # items in `parameter_ranges: List[list]` are lists of 
    # values for the parameter specified in `accession_keys`.
    accession_keys, parameter_ranges = _get_keys_and_list(
        parameters
    )
    # find all possible combinations of parameters
    # each item in `param_sets` is a tuple of parameter values
    # each element in the tuple matches the keys in `keys` with 
    # the same index.
    param_sets = list(
        itertools.product(
            *parameter_ranges,
        )
    )
    
    #################################################
    # select a set of parameters to search
    #################################################
    if search.lower() == 'random':
        # perform a random search by subsetting grid points
        param_idx = np.random.choice(
            len(param_sets),
            size=n_points,
            replace=False,
        )
    else:
        param_idx = range(len(param_sets))
        
    #################################################
    # set a common train/test/val split for all params
    #################################################
    
    splits_provided = (
        'scNym_split_0' in adata.obs.columns
    )
    splits_provided = (
        splits_provided or 'scNym_split' in adata.obs.columns
    )
    
    if not splits_provided:
        split_data(
            adata,
            groupby=groupby,
            n_splits=n_splits,
        )
    elif n_splits == 1 and 'scNym_split' in adata.obs.columns:
        adata.obs['scNym_split_0'] = adata.obs['scNym_split']
    elif n_splits > 1 and splits_provided:
        # check that we have the relevant split for each fold
        splits_correct = True
        for s in range(n_splits):
            splits_correct = (
                splits_correct & (f'scNym_split_{s}' in adata.obs.columns)
            )
        if not splits_correct:
            msg = '"scNym_split_" was provided with `n_splits>1.\n'
            msg += 'f"scNym_split_{n}"" must be present in `adata.obs` for all {n} in `range(n_splits)`\n'
            raise ValueError(msg)
    else:
        msg = 'invalid argument for n_splits'
        raise ValueError(msg)

    #################################################
    # circular training for each parameter set
    #################################################
    
    accession_keys_str = [
        '::'.join(x) for x in accession_keys
    ]
    
    search_results = []
    search_config_store = []
    for search_number, idx in enumerate(param_idx):
        # get the parameter set
        params = param_sets[idx]
        # update the base config with search parameters
        search_config = copy.deepcopy(config)
        for p_i in range(len(params)):
            keys2update = accession_keys[p_i]
            value2set   = params[p_i]
            # updates in place
            _updated_nested(
                search_config,
                keys2update,
                value2set,
            )
        
        # disable checkpoints, tensorboard to reduce I/O
        search_config['save_freq'] = 10000
        search_config['tensorboard'] = False
        # add search number to config
        search_config['search_number'] = search_number
        
        search_config_store.append(
            copy.deepcopy(search_config),
        )
        print('searching config:')
        pprint.pprint(search_config)
        
        for split_number in range(n_splits):
            # set the relevant split indices
            adata.obs['scNym_split'] = (
                adata.obs[f'scNym_split_{split_number}']
            )
            # set the split number
            split_config = copy.deepcopy(search_config)
            split_config['split_number'] = split_number
            search_df = _circular_train(
                search_config=split_config,
                params=params,
                adata=adata,
                groupby=groupby,
                out_path=out_path,
                accession_keys=accession_keys,
                hold_out_only=hold_out_only,
                groupby_eval=groupby_eval,
            )
            # add the split information
            search_df['split_number'] = split_number
            search_df['search_number'] = search_number
            # save results
            search_results.append(search_df)
        
            print('search results:')
            print(search_df)
    
    # concatenate
    search_results = pd.concat(search_results, 0)
    best_idx = np.argmax(search_results['acc'])
    best_search = int(
        search_results.iloc[best_idx]['search_number']
    )
    
    best_config = search_config_store[best_search]
    print('>>>>>>')
    print('Best config')
    print(best_config)
    print('>>>>>>')
    print()
    return search_results, best_config
