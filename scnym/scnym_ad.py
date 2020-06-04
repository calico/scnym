'''scNym model training from standard anndata objects'''
import anndata
import os
import os.path as osp
import uuid
import configargparse
import numpy as np
import pandas as pd

from .main import train_cv, train_all
from .model import CellTypeCLF
from .utils import build_classification_matrix
from sklearn.model_selection import StratifiedKFold


def make_parser():
    parser = configargparse.ArgParser(
        description='train an scNym cell type classification model.'
    )
    parser.add_argument(
        '--config',
        type=str,
        is_config_file=True,
        required=False,
        help='path to a configuration file.'
    )
    parser.add_argument(
        '--data',
        type=str,
        help='path to an h5ad [Cells, Features] object.'
    )
    parser.add_argument(
        '--groupby',
        type=str,
        help='categorical feature in `adata.obs` to use for classifier training.'
    )
    parser.add_argument(
        '--out_path',
        type=str,
        help='path for outputs.'
    )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=256,
        help='batch size for training',
    )
    parser.add_argument(
        '--n_epochs', 
        type=int, 
        default=200,
        help='number of epochs for training',
    )
    parser.add_argument(
        '--init_dropout', 
        type=float, 
        default=0.0,
        help='initial dropout to perform on gene inputs',
    )
    parser.add_argument(
        '--n_hidden', 
        type=int, 
        default=128,                
        help='number of hidden units in the classifier',
    )
    parser.add_argument(
        '--n_layers', 
        type=int, 
        default=2,
        help='number of hidden layers in the model',
    )
    parser.add_argument(
        '--residual', 
        action='store_true',
        help='use residual layers in the model',
    )
    parser.add_argument(
        '--weight_decay', 
        type=float, 
        default=1e-5,
        help='weight decay applied by the optimizer',
    )
    parser.add_argument(
        '--weight_classes', 
        type=bool, 
        default=True,
        help='weight loss based on relative class abundance.',
    )
    parser.add_argument(
        '--mixup_alpha', 
        type=float, 
        default=None,
        help='alpha parameter for MixUp training. if set performs MixUp, otherwise does not.',
    )
    parser.add_argument(
        '--unlabeled_counts',
        type=str, 
        default=None,
        help='path to h5ad [Cells, Features] object of unlabeled data.',
    )
    parser.add_argument(
        '--unsup_max_weight', 
        type=float,
        default=2.,
        help='maximum weight for the unsupervised component of IC training.',
    )
    parser.add_argument(
        '--unsup_mean_teacher', 
        type=bool,
        default=True,
        help='use a mean teacher for IC training.',
    )
    parser.add_argument(
        '--cross_val_train', 
        action='store_true',
    )
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()
    
    adata = anndata.read_h5ad(args.data)
    print(f'{adata.shape[0]} cells, {adata.shape[1]} genes in the training data.')
    
    if args.groupby not in adata.obs:
        msg = f'{args.groupby} not in `adata.obs`'
        raise ValueError(msg)
        
    os.makedirs(args.out_path, exist_ok=True)
    
    
    if args.unlabeled_counts is None:
        unlabeled_counts = None
    else:
        # load unlabeled counts and build a matrix that follows
        # gene dimension ordering of the training data
        unlabeled_adata = anndata.read_h5ad(args.unlabeled_counts)
        unlabeled_counts = build_classification_matrix(
            X=unlabeled_adata.X if type(unlabeled_adata.X)==np.ndarray else unlabeled_adata.X.toarray(),
            model_genes=np.array(adata.var_names),
            sample_genes=np.array(unlabeled_adata.var_names),
        )
    
    X = adata.X if type(adata.X)==np.ndarray else adata.X.toarray()
    y = pd.Categorical(adata.obs[args.groupby]).codes
    
    model_params = {
        'n_hidden' : args.n_hidden,
        'residual' : args.residual,
        'n_layers' : args.n_layers,
        'init_dropout' : args.init_dropout,        
    }
    
    if args.cross_val_train:
        kf = StratifiedKFold(n_splits=5, shuffle=True)
        fold_indices = list(kf.split(X, y))

        fold_eval_acc, fold_eval_losses = train_cv(
            X=X,
            y=y,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            weight_decay=args.weight_decay,
            ModelClass=CellTypeCLF,
            fold_indices=fold_indices,
            out_path=args.out_path,
            n_genes=adata.shape[1],
            mixup_alpha=args.mixup_alpha,
            unlabeled_counts=unlabeled_counts,
            unsup_max_weight=args.unsup_max_weight,
            unsup_mean_teacher=args.unsup_mean_teacher,
            weighted_classes=args.weight_classes,
            **model_params,
        )
        np.savetxt(
            osp.join(args.out_path, 'fold_eval_losses.csv',),
            fold_eval_losses,
        )
        np.savetxt(
            osp.join(args.out_path, 'fold_eval_acc.csv',),
            fold_eval_acc,
        )
    
    val_loss, val_acc = train_all(
        X=X,
        y=y,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        weight_decay=args.weight_decay,
        ModelClass=CellTypeCLF,
        out_path=args.out_path,
        n_genes=adata.shape[1],
        mixup_alpha=args.mixup_alpha,
        unlabeled_counts=unlabeled_counts,
        unsup_max_weight=args.unsup_max_weight,
        unsup_mean_teacher=args.unsup_mean_teacher,
        weighted_classes=args.weight_classes,
        **model_params,
    )
    print(f'Final validation loss: {val_loss:08}')
    print(f'Final validation acc : {val_acc:08}')
    
    # get exp id
    exp_id = uuid.uuid4()
    res = pd.DataFrame(
        {'val_acc': val_acc, 'val_loss': val_loss},
        index=[exp_id],
    ).to_csv(
        osp.join(
            args.out_path,
            'all_data_val_results.csv',
        )
    )
    return
