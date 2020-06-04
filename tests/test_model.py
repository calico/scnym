import pytest
import os
import os.path as osp
import sys
import numpy as np
import torch

sys.path.append('../')


def test_model_build():
    '''Test construction of a standard model'''
    import scnym
    torch.manual_seed(1)
    np.random.seed(1)    
    
    model = scnym.model.CellTypeCLF(
        n_genes=10000,
        n_cell_types=10,
        n_hidden=64,
        n_layers=2,
        residual=False,
        init_dropout=0.0,
    )
    
    assert hasattr(model, 'classif')
    assert ~hasattr(model, 'dan')
    
    
    model = scnym.model.CellTypeCLF(
        n_genes=10000,
        n_cell_types=10,
        n_hidden=64,
        n_layers=2,
        residual=False,
        init_dropout=0.0,
        track_running_stats=False,
    )
    
    return


def test_model_dan_build():
    '''Test construction of a DANN model'''
    import scnym
    torch.manual_seed(1)
    np.random.seed(1)    
    
    model = scnym.model.CellTypeCLF(
        n_genes=10000,
        n_cell_types=10,
        n_hidden=64,
        n_layers=2,
        residual=False,
        init_dropout=0.0,
    )
    
    assert hasattr(model, 'classif')
    
    dann = scnym.model.DANN(
        model=model,
        n_domains=2,
    )
        
    assert hasattr(dann, 'embed')
    assert hasattr(dann, 'domain_clf')
    
    return


def test_model_save_load():
    '''Test saving and loading of different model types'''
    import scnym
    torch.manual_seed(1)
    np.random.seed(1)    
    
    std_model = scnym.model.CellTypeCLF(
        n_genes=10000,
        n_cell_types=10,        
        n_hidden=64,
        n_layers=2,
        residual=False,
        init_dropout=0.0,
    )
    
    os.makedirs('./tmp', exist_ok=True)
    # save
    torch.save(
        std_model.state_dict(),
        './tmp/std_model.pkl',
    )
    
    # load
    std_model.load_state_dict(
        torch.load(
            './tmp/std_model.pkl',
        )
    )
    
    return


def main():
    test_model_dan_build()
    return


if __name__ == '__main__':
    main()