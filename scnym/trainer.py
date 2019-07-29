import numpy as np
import os
import os.path as osp
import torch
import torch.nn as nn
import json
from typing import Callable


class Trainer(object):
    '''
    Trains a PyTorch model.

    Attributes
    ----------
    model : nn.Module
        model with required `.forward(...)` method.
    criterion : Callable
        loss criterion to optimize.
    optimizer : torch.optim.Optimizer
        optimizer for the model parameters.
    dataloaders : dict
        keyed by ['train', 'val'] with values corresponding
        to `torch.utils.data.DataLoader` for training
        and validation sets.
    out_path : str
        output path for best model.
    n_epochs : int
        number of epochs for training.
    reg_criterion : Callable
        criterion to penalize layer weights.
    use_gpu : bool
        use CUDA acceleration.
    verbose : bool
        write all batch losses to stdout.
    save_freq : int
        Number of epochs between model checkpoints. Default = 10.
    scheduler : learning rate scheduler.
    '''

    def __init__(self,
                 model: nn.Module,
                 criterion: Callable,
                 optimizer: torch.optim.Optimizer,
                 dataloaders: dict,
                 out_path: str,
                 n_epochs: int = 50,
                 exp_name: str = '',
                 reg_criterion: Callable = None,
                 use_gpu: bool = torch.cuda.is_available(),
                 verbose: bool = False,
                 save_freq: int = 10,
                 scheduler=None,) -> None:
        '''
        Trains a PyTorch `nn.Module` object provided in `model`
        on training and testing sets provided in `dataloaders`
        using `criterion` and `optimizer`.

        Saves model weight snapshots every `save_freq` epochs and saves the
        weights with the best testing loss at the end of training.

        Parameters
        ----------
        model : nn.Module
            model with required `.forward(...)` method.
        criterion : Callable
            loss criterion to optimize.
        optimizer : torch.optim.Optimizer
            optimizer for the model parameters.
        dataloaders : dict
            keyed by ['train', 'val'] with values corresponding
            to `torch.utils.data.DataLoader` for training
            and validation sets.
        out_path : str
            output path for best model.
        n_epochs : int
            number of epochs for training.
        reg_criterion : callable
            criterion to penalize layer weights.
        use_gpu : bool
            use CUDA acceleration.
        verbose : bool
            write all batch losses to stdout.
        save_freq : int
            Number of epochs between model checkpoints. Default = 10.
        scheduler : learning rate scheduler.

        Returns
        -------
        None.
        '''
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.n_epochs = n_epochs
        self.dataloaders = dataloaders
        self.out_path = out_path
        self.use_gpu = use_gpu
        self.verbose = verbose
        self.save_freq = save_freq
        self.best_acc = 0.
        self.best_loss = 1.0e10
        self.scheduler = scheduler
        self.reg_criterion = reg_criterion

        if not os.path.exists(self.out_path):
            os.mkdir(self.out_path)
        # initialize log

        self.log_path = os.path.join(
            self.out_path, '_'.join([exp_name, 'log.csv']))
        with open(self.log_path, 'w') as f:
            header = 'Epoch,Running_Loss,Mode\n'
            f.write(header)

        self.parameters = {
            'out_path': out_path,
            'exp_name': exp_name,
            'n_epochs': n_epochs,
            'use_cuda': self.use_gpu,
            'train_batch_size': self.dataloaders['train'].batch_size,
            'val_batch_size': self.dataloaders['val'].batch_size,
            'train_batch_sampler': str(
                type(self.dataloaders['train'].sampler)),
            'val_batch_sampler': str(
                type(self.dataloaders['val'].sampler)),
            'optimizer_type': str(type(self.optimizer)),
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'model_hidden': self.model.n_hidden,
            'model_ngenes': self.model.n_genes,
            'model_ncelltypes': self.model.n_cell_types,
        }

        # save parameters to JSON
        try:
            with open(
                osp.join(self.out_path,
                         exp_name + '_parameters.json'),
                    'w') as f:
                json.dump(dict(self.parameters), f)
        except:
            print('JSON saving of parameters failed!')

        # write the log file header
        with open(self.log_path, 'w') as f:
            header = 'Epoch,Iter,Running_Loss,Mode\n'
            f.write(header)

    def train_epoch(self):
        '''Perform training across one full iteration through
        the data.
        '''
        self.model.train(True)
        i = 0
        running_loss = 0.0
        running_corrects = 0.
        running_total = 0.
        for data in self.dataloaders['train']:
            inputs, labels = data['input'], data['output']
            if self.use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()
            else:
                pass
            inputs.requires_grad_()
            labels.requires_grad = False

            # zero gradients
            self.optimizer.zero_grad()

            # forward pass
            outputs = self.model(inputs)
            # predictions are the output nodes with
            # the highest values
            _, predictions = torch.max(outputs, 1)

            if self.verbose:
                print('Preds:')
                print(predictions[:10])
                print('Labels:')
                print(labels[:10])

            correct = torch.sum(predictions.detach() == labels.detach())

            # compute loss
            if self.reg_criterion is not None:
                reg_loss = self.reg_criterion(self.model)
                loss = self.criterion(outputs, labels) + reg_loss
            else:
                loss = self.criterion(outputs, labels)

            if self.verbose:
                print('batch loss: ', loss.item())
            if np.isnan(loss.data.cpu().numpy()):
                raise RuntimeError('NaN loss encountered in training')

            # compute gradients in a backward pass, update parameters
            loss.backward()
            self.optimizer.step()

            # statistics update
            running_loss += loss.item() / inputs.size(0)
            running_corrects += float(correct.item())
            running_total += float(labels.size(0))

            if i % 100 == 0:
                print('Iter : ', i)
                print('running_loss : ', running_loss / (i + 1))
                print('running_acc  : ', running_corrects/running_total)
                print('corrects: %f | total: %f' %
                      (running_corrects, running_total))
                # append to log
                with open(self.log_path, 'a') as f:
                    f.write(str(self.epoch) + ',' + str(i) + ',' +
                            str(running_loss / (i + 1)) + ',train\n')
            i += 1

        epoch_loss = running_loss / len(self.dataloaders['train'])
        epoch_acc = running_corrects / running_total

        # append to log
        with open(self.log_path, 'a') as f:
            f.write(str(self.epoch) + ',' + str(i) + ',' +
                    str(running_loss / (i + 1)) + ',train_epoch\n')

        print('{} Loss : {:.4f}'.format('train', epoch_loss))
        print('{} Acc : {:.4f}'.format('train', epoch_acc))
        print('TRAIN EPOCH corrects: %f | total: %f' %
              (running_corrects, running_total))

    @torch.no_grad()
    def val_epoch(self):
        '''Perform a pass through the validation data.
        Do not record gradients to speed things up.
        '''
        self.model.train(False)
        i = 0
        running_loss = 0.0
        running_corrects = 0
        running_total = 0
        for data in self.dataloaders['val']:
            inputs, labels = data['input'], data['output']
            if self.use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()
            else:
                pass

            # zero gradients
            self.optimizer.zero_grad()
            # forward pass
            outputs = self.model(inputs)
            _, predictions = torch.max(outputs, 1)
            correct = torch.sum(predictions.detach() == labels.detach())
            if self.verbose:
                print('PRED\n', predictions[:10, ...])
                print('LABEL\n', labels[:10, ...])
                print('CORRECT: ', correct)

            if self.reg_criterion is not None:
                reg_loss = self.reg_criterion(self.model)
                loss = self.criterion(outputs, labels) + reg_loss
            else:
                loss = self.criterion(outputs, labels)

            # statistics update
            running_loss += loss.item() / inputs.size(0)
            running_corrects += int(correct.item())
            running_total += int(labels.size(0))

            if i % 1 == 10:
                print('Iter : ', i)
                print('running_loss : ', running_loss / (i + 1))
                print('running_acc  : ', running_corrects/running_total)
                print('corrects: %f | total: %f' %
                      (running_corrects, running_total))
                # append to log
                with open(self.log_path, 'a') as f:
                    f.write(str(self.epoch) + ',' + str(i) + ',' +
                            str(running_loss / (i + 1)) + ',val\n')
            i += 1

        epoch_loss = running_loss / len(self.dataloaders['val'])
        epoch_acc = running_corrects / running_total
        # append to log
        with open(self.log_path, 'a') as f:
            f.write(
                str(self.epoch)
                + ','
                + str(i)
                + ','
                + str(running_loss / (i + 1))
                + ',val_epoch\n')

        if epoch_loss < self.best_loss:
            self.best_loss = epoch_loss
            self.best_model_wts = self.model.state_dict()
            torch.save(
                self.model.state_dict(),
                os.path.join(self.out_path,
                             ('model_weights_'
                              + str(self.epoch).zfill(3)
                              + '.pkl')
                             )
            )
            print('Saving best model weights...')
            torch.save(
                self.model.state_dict(),
                os.path.join(self.out_path, '00_best_model_weights.pkl'))
            print('Saved best weights.')
        elif (self.epoch % self.save_freq == 0):
            torch.save(self.model.state_dict(),
                       os.path.join(self.out_path,
                                    'model_weights_'
                                    + str(self.epoch).zfill(3)
                                    + '.pkl'))

        print('{} Loss : {:.4f}'.format('val', epoch_loss))
        print('{} Acc : {:.4f}'.format('val', epoch_acc))
        print('VAL EPOCH corrects: %f | total: %f' %
              (running_corrects, running_total))

    def train(self):
        for epoch in range(self.n_epochs):
            self.epoch = epoch
            print('Epoch {}/{}'.format(epoch, self.n_epochs - 1))
            print('-' * 10)
            # run training epoch
            if self.scheduler is not None:
                self.scheduler.step()
            self.train_epoch()
            self.val_epoch()

        self.model.load_state_dict(self.best_model_wts)
        return self.model
