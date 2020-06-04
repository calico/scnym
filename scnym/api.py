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
from typing import Optional, Union

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
# for fetching pretrained weights, all in standard lib
import requests
import json
import urllib
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
        },
        'model_kwargs' : {
            'n_hidden': 256,
            'n_layers': 2,
            'init_dropout': 0.0,
            'residual': False,
        },
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
        Used when `task=="predict"'.
        Path to the output directory of an scNym training run
        or a string specifying a pretrained model.
        Pretrained model strings are f"pretrained_{species}" where
        species is one of `{"human", "mouse", "rat"}`.
        Providing a pretrained model string will download pre-trained weights
        and predict directly on the target data, without additional training.
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
    
    **Predict cell identities with a pretrained scNym model**
    
    >>> scnym_api(
    ...   adata=adata,
    ...   task='predict',
    ...   groupby='scNym',
    ...   trained_model='pretrained_human',
    ...   config='no_new_identity',
    ... )
    
    **Perform semi-supervised training with an atlas**
    
    >>> joint_adata = atlas2target(
    ...   adata=adata,
    ...   species='human',
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
        for k in config:
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
        
    traintest_idx = np.random.choice(
        X.shape[0],
        size=int(np.floor(0.9*X.shape[0])),
        replace=False
    )
    val_idx = np.setdiff1d(np.arange(X.shape[0]), traintest_idx)
        
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
        unsup_max_weight=config['unsup_max_weight'],
        unsup_mean_teacher=config['unsup_mean_teacher'],
        ssl_method=config['ssl_method'],
        ssl_kwargs=config['ssl_kwargs'],
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
        species = get_pretrained_weights(
            trained_model=config['trained_model'],
            out_path=config['out_path'],
        )
        print(f'Successfully downloaded pretrained model for {species}.')
        config['trained_model'] = config['out_path']
    
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


def get_pretrained_weights(
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