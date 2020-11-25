import numpy as np
import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from typing import Callable
from .dataprep import SampleMixUp
from .utils import compute_entropy_of_mixing
from .model import CellTypeCLF, DANN
import copy
from torch.utils.tensorboard import SummaryWriter


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
    min_epochs : int
        minimum number of epochs before saving weights.
    patience : int
        maximum number of epochs to wait before early stopping.
        if `None`, infinite patience is used (up to `n_epochs`).
    waiting_time : int
        number of epochs since the last best val loss.
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

    def __init__(
        self,
        model: nn.Module,
        criterion: Callable,
        optimizer: torch.optim.Optimizer,
        dataloaders: dict,
        out_path: str,
        batch_transformers: dict = {},
        n_epochs: int = 50,
        min_epochs: int = 0,
        patience: int = None,
        exp_name: str = '',
        reg_criterion: Callable = None,
        use_gpu: bool = torch.cuda.is_available(),
        verbose: bool = False,
        save_freq: int = 10,
        scheduler: torch.optim.lr_scheduler = None,
        tb_writer: str = None,
    ) -> None:
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
        batch_transformers : dict
            apply transforms to minibatch inputs and targets.
            keys are ['train', 'val'], values are Callable.
        n_epochs : int
            number of epochs for training.
        min_epochs : int
            minimum number of epochs before saving weights.
        patience : int
            maximum number of epochs to wait before early stopping.
            if `None`, infinite patience is used (up to `n_epochs`).
        reg_criterion : callable
            criterion to penalize layer weights.
        use_gpu : bool
            use CUDA acceleration.
        verbose : bool
            write all batch losses to stdout.
        save_freq : int
            Number of epochs between model checkpoints. Default = 10.
        scheduler : torch.optim.lr_scheduler
            learning rate schedule.
            
        Returns
        -------
        None.
        '''
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.n_epochs = n_epochs
        self.min_epochs = min_epochs
        self.patience = patience if patience is not None else n_epochs
        self.waiting_time = 0
        self.dataloaders = dataloaders
        self.batch_transformers = batch_transformers
        self.out_path = out_path
        self.use_gpu = use_gpu
        self.verbose = verbose
        self.save_freq = save_freq
        self.best_acc = 0.
        self.best_loss = 1.0e10
        self.scheduler = scheduler
        self.reg_criterion = reg_criterion
        if tb_writer is not None:
            self.tb_writer = SummaryWriter(log_dir=tb_writer)
            os.makedirs(tb_writer, exist_ok=True)
        else:
            self.tb_writer = None

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

        btrans = self.batch_transformers.get('train', None)
        for data in self.dataloaders['train']:
            # if a batch transformer is present,
            # transform the data before use
            if btrans is not None:
                data = btrans(data)

            inputs = data['input']
            labels = data['output']  # one-hot

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

            # remake an integer version of the labels for quick checking
            int_labels = torch.argmax(labels, 1)

            correct = torch.sum(predictions.detach() == int_labels.detach())

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

            if i % 100 == 0 and self.verbose:
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
        
        if self.tb_writer is not None:
            self.tb_writer.add_scalar('Loss/train', epoch_loss, self.epoch)
            self.tb_writer.add_scalar('Acc/train', epoch_acc, self.epoch)
            for i, p in enumerate(self.model.parameters()):
                self.tb_writer.add_histogram(
                    f'Grad/param{i:04}', p.grad, self.epoch,
                )

            self.tb_writer.add_scalar(
                'lr/lr',
                self.optimizer.state_dict()['param_groups'][0]['lr'],
                self.epoch,
            )
        
        if self.verbose:
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

        btrans = self.batch_transformers.get('val', None)
        for data in self.dataloaders['val']:
            # if a batch transformer is present,
            # transform the data before use
            if btrans is not None:
                data = btrans(data)

            inputs = data['input']
            labels = data['output']  # one-hot
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

            # remake an integer version of the labels for quick checking
            int_labels = torch.argmax(labels, 1)
            correct = torch.sum(predictions.detach() == int_labels.detach())
            if self.verbose > 1:
                print('PRED\n', predictions[:10, ...])
                print('LABEL\n', int_labels[:10, ...])
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

            if i % 1 == 10 and self.verbose > 1:
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

        # add one epoch to the waiting time for best loss
        # if we had a new best loss, the counter is reset below
        self.waiting_time += 1
        if (epoch_loss < self.best_loss) and (self.epoch >= self.min_epochs):
            self.best_loss = epoch_loss
            self.best_model_wts = self.model.state_dict()
            self.waiting_time = 0
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
                os.path.join(self.out_path, '00_best_model_weights.pkl'),
            )
            print('Saved best weights.')
            
            if hasattr(self, 'dan_criterion'):
                print('Trainer has a `dan_criterion`.')
                if self.dan_criterion is not None:
                    print('Saving DAN weights...')
                    torch.save(
                        self.dan_criterion.dann.state_dict(),
                        os.path.join(
                            self.out_path,
                            '02_best_dan_weights.pkl',
                        )
                    )
            
            with open(self.log_path, 'a') as f:
                f.write(
                    str(self.epoch)
                    + ','
                    + str(i)
                    + ','
                    + str(running_loss / (i + 1))
                    + ',best_model_weights\n',
                )
                
            if self.tb_writer is not None:
                self.tb_writer.add_text(
                    'BestWeights',
                    f'Saved best weights at {self.epoch}, loss {epoch_loss}',
                    self.epoch,
                )
                self.tb_writer.flush()
            
        elif (self.epoch % self.save_freq == 0):
            torch.save(
                self.model.state_dict(),
                os.path.join(
                    self.out_path,
                    'model_weights_'
                    + str(self.epoch).zfill(3)
                    + '.pkl',
                ),
            )
        
        elif self.epoch == (self.n_epochs-1):
            torch.save(
                self.model.state_dict(),
                os.path.join(
                    self.out_path,
                    '01_final_model_weights.pkl'
                ),
            )
        if self.verbose:
            print(f'{self.waiting_time} epochs since last best weights.\n')
        
        if self.tb_writer is not None:
            self.tb_writer.add_scalar('Loss/val', epoch_loss, self.epoch)
            self.tb_writer.add_scalar('Acc/val', epoch_acc, self.epoch)
            self.tb_writer.flush()
        
        if self.verbose:
            print('{} Loss : {:.4f}'.format('val', epoch_loss))
            print('{} Acc : {:.4f}'.format('val', epoch_acc))
            print('VAL EPOCH corrects: %f | total: %f' %
                  (running_corrects, running_total))

    def train(self):
        for epoch in range(self.n_epochs):
            self.epoch = epoch
            msg = f'Epoch {epoch}/{self.n_epochs-1}'
            p_complete = epoch/self.n_epochs
            n_bars = int(np.floor(30*p_complete))
            msg += '|' + '-'*n_bars + '_'*(30-n_bars) + '|'
            # print a new line so the progress bar isn't overwritten
            # on the final stdout
            end_char = '\n' if epoch == (self.n_epochs-1) else '\r'
            print(msg, end=end_char)
            
            # training epoch
            self.train_epoch()
            # evaluate model
            self.val_epoch()
            
            # update learning rate
            # NOTE: change in `torch>=1.1.0`, `scheduler.step()`
            # is now called AFTER `optimizer.step()`
            if self.scheduler is not None:
                self.scheduler.step()
                
            if self.waiting_time > self.patience:
                # we have waited a sufficient number of epochs
                # to perform early stopping
                print('>'*5)
                print(f'Early stopping at epoch {self.epoch}')
                print('>'*5)
                break

        self.model.load_state_dict(self.best_model_wts)
        
        if self.tb_writer is not None:
            # close tensorboard writer
            self.tb_writer.flush()
            self.tb_writer.close()
        
        return self.model


class SemiSupervisedTrainer(Trainer):

    def __init__(
        self,
        unsup_criterion: Callable,
        unsup_dataloader: torch.utils.data.DataLoader,
        unsup_weight: Callable,
        dan_criterion: Callable=None,
        dan_weight: Callable=None,
        **kwargs,
    ) -> None:
        '''Train a PyTorch model using both a supervised and
        unsupervised loss as described for Interpolation
        Consistency Training.

        Parameters
        ----------
        unsup_criterion : Callable
            loss function for unlabeled samples.
            takes both the current `nn.Module` model and a `torch.FloatTensor`
            of unlabeled samples as input.
        unsup_dataloader : torch.utils.data.DataLoader
            data loader supplying unlabeled samples.
        unsup_weight : Callable
            takes an int epoch as input and returns a weight coefficient
            to scale the importance of the unsupervised loss.
        dan_criterion : Callable, optional
            domain adaptation loss. takes in a model, labeled batch, and
            unlabeled batch, and returns a `torch.Tensor` loss value.
        dan_weight : Callable, optional
            domain adaptation loss weight schedule.
            takes an int epoch as input and returns a weight coefficient.

        Returns
        -------
        None.
        '''
        super(SemiSupervisedTrainer, self).__init__(**kwargs)
        self.unsup_criterion = unsup_criterion
        self.unsup_dataloader = unsup_dataloader
        self.unsup_weight = unsup_weight
        self.dan_criterion = dan_criterion
        if self.dan_criterion is not None:
            print('Using a Domain Adaptation Loss.')
        self.dan_weight = dan_weight
        return

    def train_epoch(self,) -> None:
        '''
        Perform training using both a supervised and semi-supervised loss.

        Notes
        -----
        (1) Sample labeled examples, compute the standard supervised loss.
        (2) Sample unlabeled examples, compute unsupervised loss.
        (3) Perform backward pass and update parameters.
        '''
        self.model.train(True)
        i = 0
        running_loss = 0.
        running_sup_loss = 0. # supervised loss
        running_uns_loss = 0. # unsupervised loss
        running_dom_loss = 0. # domain adaptation loss
        running_corrects = 0.
        running_total = 0.

        btrans = self.batch_transformers.get('train', None)

        iter_unsup_dl = iter(self.unsup_dataloader)
        for data in self.dataloaders['train']:

            ####################################
            # (1) Prepare data and graph
            ####################################
            
            # get unlabeled batch
            unsup_data = next(iter_unsup_dl)
            
            if btrans is not None:
                data = btrans(data)
            
            if self.use_gpu:
                # push all the data to the CUDA device
                data['input'] = data['input'].cuda()
                data['output'] = data['output'].cuda()
                
                unsup_data['input'] = unsup_data['input'].cuda()
        
            # capture gradients on labeled and unlabeled inputs
            # do not store gradients on labels
            data['input'].requires_grad = True
            data['output'].requires_grad = False
            
            unsup_data['input'].requires_grad = True
            
            # zero gradients across the graph
            self.optimizer.zero_grad()

            ####################################
            # (2) Compute loss terms
            ####################################

            sup_loss, unsup_loss, sup_outputs = self.unsup_criterion(
                model=self.model,
                unlabeled_sample=unsup_data,
                labeled_sample=data,
            )
            
            # check supervised classification accuracy
            _, predictions = torch.max(sup_outputs, 1)
            int_labels = torch.argmax(data['output'], 1)

            correct = torch.sum(predictions.detach() == int_labels.detach())

            # compute regularization loss
            if self.reg_criterion is not None:
                reg_loss = self.reg_criterion(self.model)
            else:
                reg_loss = 0.
                
            # compute the domain adaptation loss if desired
            if self.dan_criterion is not None:
                dan_weight = self.dan_weight(self.epoch)
                # NOTE: pseudolabel confidence is only used if `use_conf_pseudolabels`
                # was passed to the initiatilization of `DANLoss`
                pseudolabel_confidence = self.unsup_criterion.running_confidence_scores[-1][0]
                dan_loss = self.dan_criterion(
                    labeled_sample=data,
                    unlabeled_sample=unsup_data,
                    weight=dan_weight,
                    pseudolabel_confidence=pseudolabel_confidence,
                )
            else:
                dan_loss = torch.zeros(1,).float()
                dan_loss = dan_loss.to(device=sup_loss.device)
                dan_weight = 0.

            ####################################
            # (3) Perform backward pass
            ####################################

            loss = (
                sup_loss 
                + reg_loss 
                + (self.unsup_weight(self.epoch) * unsup_loss)
                + dan_loss
            )

            if self.verbose > 1:
                print('sup.  loss:   ', sup_loss.item())
                print('usup. loss:   ', unsup_loss.item())
                print('usup. weight: ', self.unsup_weight(self.epoch))
                if self.dan_criterion is not None:
                    print('Dom.  loss:   ', dan_loss.item())
                    print('Dom.  weight: ', dan_weight)
                print('total loss: ', loss.item())
            if np.isnan(loss.data.cpu().numpy()):
                raise RuntimeError('NaN loss encountered in training')

            # compute gradients in a backward pass, update parameters
            loss.backward()
            self.optimizer.step()

            # statistics update
            labeled_n = data['input'].size(0)
            unlabel_n = unsup_data['input'].size(0)
            
            running_loss += loss.item()
            running_sup_loss += sup_loss.item()
            running_uns_loss += unsup_loss.item()
            running_dom_loss += dan_loss.item()
            running_corrects += float(correct.item())
            running_total += float(data['input'].size(0))

            if i % 100 == 0 and self.verbose:
                print('Iter : ', i)
                print('running_sup_loss : ', running_sup_loss / (i + 1))
                print('running_uns_loss : ', running_uns_loss / (i + 1))
                print('running_dom_loss : ', running_dom_loss / (i + 1))                
                print('running_loss : ', running_loss / (i + 1))
                print('running_acc  : ', running_corrects/running_total)
                print('corrects: %f | total: %f' %
                      (running_corrects, running_total))
                # append to log
                with open(self.log_path, 'a') as f:
                    f.write(str(self.epoch) + ',' + str(i) + ',' +
                            str(running_loss / (i + 1)) + ',train\n')
            i += 1

        epoch_sup_loss = running_sup_loss / len(self.dataloaders['train'])
        epoch_uns_loss = running_uns_loss / len(self.dataloaders['train'])
        epoch_dom_loss = running_dom_loss / len(self.dataloaders['train'])
        epoch_loss = running_loss / len(self.dataloaders['train'])
        epoch_acc = running_corrects / running_total
        
        if self.tb_writer is not None:
            self.tb_writer.add_scalar(
                'Loss/train', epoch_loss, self.epoch,
            )
            self.tb_writer.add_scalar(
                'Acc/train', epoch_acc, self.epoch,
            )
            self.tb_writer.add_scalar(
                'Loss/super', epoch_sup_loss, self.epoch,
            )
            self.tb_writer.add_scalar(
                'Loss/unsup', epoch_uns_loss, self.epoch,
            )
            self.tb_writer.add_scalar(
                'SSL/UnsWeight', self.unsup_weight(self.epoch), self.epoch,
            )
            if self.dan_criterion is not None:
                self.tb_writer.add_scalar(
                    'Loss/domain', epoch_dom_loss, self.epoch,
                )            
                self.tb_writer.add_scalar(
                    'SSL/DomWeight', self.dan_weight(self.epoch), self.epoch,
                )
                
                # add embedding
                dlabel = self.dan_criterion.dlabel.numpy()
                self.tb_writer.add_embedding(
                    self.dan_criterion.x_embed,
                    metadata=dlabel.tolist(),
                    global_step=self.epoch,
                    tag='Embed/DAN',
                )
                
                # compute the entropy of mixing
                dan_embedding = self.dan_criterion.x_embed.numpy()
                
                eom = compute_entropy_of_mixing(
                    X=dan_embedding,
                    y=dlabel[:,0],
                    n_neighbors=100,
                    n_iters=512,
                    n_jobs=-1,
                )
                self.tb_writer.add_scalar(
                    'SSL/entropy_of_mixing', np.mean(eom), self.epoch,
                )
                self.tb_writer.add_histogram(
                    'SSL/dist_entropy_of_mixing', eom, self.epoch,
                )
                self.tb_writer.add_scalar(
                    'SSL/domain_acc', self.dan_criterion.dan_acc, self.epoch,
                )
                
                for i, param in enumerate(self.dan_criterion.dann.domain_clf.parameters()):
                    self.tb_writer.add_histogram(
                        f'Grad/domain_clf_{i:04}', param.grad, self.epoch,
                    )
                self.tb_writer.add_scalar(
                    'SSL/dan_n_conf_pseudolabels', 
                    self.dan_criterion.n_conf_pseudolabels,
                    self.epoch,
                )
                self.tb_writer.add_scalar(
                    'SSL/dan_p_conf_pseudolabels',
                    self.dan_criterion.n_conf_pseudolabels/self.dan_criterion.n_total_unlabeled,
                    self.epoch,                    
                )

            self.tb_writer.flush()
                   
            for i, named_mod in enumerate(self.model.classif.named_modules()):
                module_name = named_mod[0]
                module = named_mod[1]
                for j, param in enumerate(module.parameters()):
                    self.tb_writer.add_histogram(
                        f'Grad/{module_name}/{j:04}', param.grad, self.epoch,
                    )
            
            # add the running confidence scores of unlabeled examples
            # if we're using MixMatch
            if hasattr(self.unsup_criterion, 'running_confidence_scores'):
                # get the number of confident pseudolabels
                # and the total number of pseudolabels per batch
                n_conf = torch.Tensor([
                    torch.sum(s[0]).item() for s in self.unsup_criterion.running_confidence_scores
                ])
                n_total = torch.Tensor([
                    s[0].size(0) for s in self.unsup_criterion.running_confidence_scores
                ])
                conf_dist = torch.cat(
                    [
                        s[1] for s in self.unsup_criterion.running_confidence_scores
                    ],
                    dim=0,
                )
                self.tb_writer.add_scalar(
                    'SSL/p_conf_pseudolabels',
                    torch.sum(n_conf)/torch.sum(n_total),
                    self.epoch,
                )
                self.tb_writer.add_scalar(
                    'SSL/avg_pseudolabel_conf',
                    torch.mean(conf_dist),
                    self.epoch,
                )
                self.tb_writer.add_histogram(
                    'SSL/dist_p_conf_pseudolabels',
                    n_conf/n_total,
                    self.epoch,
                )
                self.tb_writer.add_histogram(
                    'SSL/pseudolabel_conf',
                    conf_dist,
                    self.epoch,
                )

        # append to log
        with open(self.log_path, 'a') as f:
            f.write(str(self.epoch) + ',' + str(i) + ',' +
                    str(epoch_loss) + ',train_epoch\n')
            # write out the supervised and unsupervised components
            # of loss separately
            f.write(str(self.epoch) + ',' + str(i) + ',' +
                    str(epoch_sup_loss) + ',train_epoch_sup\n')
            f.write(str(self.epoch) + ',' + str(i) + ',' +
                    str(epoch_uns_loss) + ',train_epoch_uns\n')
            f.write(str(self.epoch) + ',' + str(i) + ',' +
                    str(self.unsup_weight(self.epoch)) + ',train_epoch_uns_weight\n')
        if self.verbose:
            print('{} Sup. Loss : {:.6f}'.format('train', epoch_sup_loss))
            print('{} Unsup. Loss : {:.6f}'.format('train', epoch_uns_loss))
            print('{} Unsup. Weight : {:.6f}'.format(
                'train', self.unsup_weight(self.epoch)))
            if self.dan_criterion is not None:
                print('{} Dom.  Loss : {:.6f}'.format('train', epoch_dom_loss))
                print(f'train Dom.  Weight : {self.dan_weight(self.epoch)}')
            print('{} Loss : {:.4f}'.format('train', epoch_loss))
            print('{} Acc : {:.4f}'.format('train', epoch_acc))
            print('TRAIN EPOCH corrects: %f | total: %f' %
                  (running_corrects, running_total))
        return


'''Loss functions'''


def get_class_weight(
    y: np.ndarray,
) -> np.ndarray:
    '''Generate relative class weights based on the representation
    of classes in a label vector `y`

    Parameters
    ----------
    y : np.ndarray
        [N,] vector of class labels.

    Returns
    -------
    class_weight : np.ndarray
        [Classes,] vector of loss weight coefficients.
        if classes are `str`, returns weights in lexographically
        sorted order.

    '''
    # find all unique class in y and their counts
    u_classes, class_counts = np.unique(y, return_counts=True)
    # compute class proportions
    class_prop = class_counts/len(y)
    # invert proportions to get class weights
    class_weight = 1./class_prop
    # normalize so that the minimum value is 1.
    class_weight = class_weight / class_weight.min()
    return class_weight


def cross_entropy(
    pred_: torch.FloatTensor,
    label: torch.FloatTensor,
    class_weight: torch.FloatTensor = None,
    sample_weight: torch.FloatTensor = None,
    reduction: str='mean',
) -> torch.FloatTensor:
    '''Compute cross entropy loss for prediction outputs
    and potentially non-binary targets.

    Parameters
    ----------
    pred_ : torch.FloatTensor
        [Batch, C] model outputs.
    label : torch.FloatTensor
        [Batch, C] labels. may not necessarily be one-hot,
        but must satisfy simplex criterion.
    class_weight : torch.FloatTensor
        [C,] relative weights for each of the output classes.
        useful for increasing attention to underrepresented
        classes.
    reduction : str
        reduction method across the batch.

    Returns
    -------
    loss : torch.FloatTensor
        mean cross-entropy loss across the batch indices.

    Notes
    -----
    Crossentropy is defined as:

    .. math::

        H(P, Q) = -\Sum_{k \in K} P(k) log(Q(k))

    where P, Q are discrete probability distributions defined
    with a common support K.
    
    References
    ----------
    See for class weight computation:
    https://pytorch.org/docs/stable/nn.html#crossentropyloss
    '''
    if pred_.size() != label.size():
        msg = f'pred size {pred_.size()} not compatible with label size {label.size()}\n'
        raise ValueError(msg)
        
    if reduction.lower() not in ('mean', 'sum', 'none'):
        raise ValueError(f'{reduction} is not a valid reduction method.')
    
    # Apply softmax transform to predictions and log transform
    pred_log_sm = torch.nn.functional.log_softmax(pred_, dim=1)
    # Compute cross-entropy with the label vector
    samplewise_loss = -1 * torch.sum(label * pred_log_sm, dim=1)
    
    if sample_weight is not None:
        # weight individual samples using sample_weight
        # we squeeze into a single column in-case it had an
        # empty singleton dimension
        samplewise_loss *= sample_weight.squeeze()
        
    if class_weight is not None:
        class_weight = class_weight.to(label.device)
        # weight the losses
        # copy the weights across the batch to allow for elementwise
        # multiplication with the samplewise losses
        class_weight = class_weight.repeat(samplewise_loss.size(0), 1)
        # compute an [N,] vector of weights for each samples' loss
        weight_vec, _ = torch.max(
            class_weight * label, 
            dim=1,
        )
        
        samplewise_loss = samplewise_loss * weight_vec
    if reduction == 'mean':
        loss = torch.mean(samplewise_loss)
    elif reduction == 'sum':
        loss = torch.sum(samplewise_loss)
    else:
        loss = samplewise_loss
    return loss


class InterpolationConsistencyLoss(object):

    def __init__(
        self,
        unsup_criterion: Callable,
        sup_criterion: Callable,
        decay_coef: float = 0.9997,
        mean_teacher: bool = True,
        augment: Callable = None,
        teacher_eval: bool = True,
        teacher_bn_running_stats: bool = None,
        **kwargs,
    ) -> None:
        '''Computes an Interpolation Consistency Loss
        given a trained model and an unlabeled minibatch.

        Parameters
        ----------
        unsup_criterion : Callable
            loss criterion for similarity between "mixed-up" 
            "fake labels" and predicted labels for "mixed-up"
            samples.
        sup_criterion : Callable
            loss for samples with a primarily labeled component.       
        decay_coef : float
            decay coefficient for mean teacher parameter
            updates.
        mean_teacher : bool
            use a mean teacher model for interpolation consistency
            loss estimation.
        augment : Callable
            augments a batch of samples.
        teacher_eval : bool
            place teacher in evaluation mode, deactivating stochastic
            model components.
        teacher_bn_running_stats : bool
            use running statistics for batch normalization mean and
            variance. 
            if False, uses minibatch statistics.
            if None, uses setting of the student model batch norm layer.

        Returns
        -------
        None.

        Notes
        -----
        Instantiates a `SampleMixUp` class and passes any
        `**kwargs` to this class.

        Uses a "mean teacher" method by keeping a running 
        average of parameter sets used in the `__call__` 
        method.

        `decay_coef` taken from the Mean Teacher paper experiments
        on ImageNet.
        https://arxiv.org/abs/1703.01780

        Formalism:

        .. math::

            icl(u) = criterion( f(Mixup(u_i, u_j)), 
                                Mixup(f(u_i), f(u_j)) )

        References
        ----------
        1. Interpolation consistency training for semi-supervised learning
        2019, arXiv:1903.03825v3, stat.ML
        Vikas Verma, Alex Lamb, Juho Kannala, Yoshua Bengio

        2. Mean teachers are better role models: \
            Weight-averaged consistency targets improve \
            semi-supervised deep learning results
        2017, arXiv:1703.01780, cs.NE
        Antti Tarvainen, Harri Valpola
        '''
        self.unsup_criterion = unsup_criterion
        self.sup_criterion = sup_criterion
        self.decay_coef = decay_coef
        self.mean_teacher = mean_teacher
        if self.mean_teacher:
            print('IC Loss is using a mean teacher.')
        self.augment = augment
        self.teacher_eval = teacher_eval
        self.teacher_bn_running_stats = teacher_bn_running_stats

        # instantiate a callable MixUp operation
        self.mixup_op = SampleMixUp(**kwargs)

        self.teacher = None
        self.step = 0
        return

    def _update_teacher(self, model: nn.Module,) -> None:
        '''Update the teacher model based on settings'''
        if self.mean_teacher:
            if self.teacher is None:
                # instantiate the teacher with a copy
                # of the model
                self.teacher = copy.deepcopy(model,)
            else:
                self._update_teacher_params(model,)
        else:
            self.teacher = copy.deepcopy(model,)
        
        if self.teacher_eval:
            self.teacher = self.teacher.eval()
            
        if self.teacher_bn_running_stats is not None:
            # enforce our preference for teacher model batch 
            # normalization statistics
            for m in self.teacher.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.track_running_stats = self.teacher_bn_running_stats
        
        # check that our parameters are preserved
        if self.teacher_bn_running_stats is not None:
            # enforce our preference for teacher model batch 
            # normalization statistics
            for m in self.teacher.modules():
                if isinstance(m, nn.BatchNorm1d):
                    assert m.track_running_stats == self.teacher_bn_running_stats
            
        return

    def _update_teacher_params(self, model: nn.Module,) -> None:
        '''Update parameters in the teacher model using an
        exponential averaging method.

        Notes
        -----
        Logic derived from the Mean Teacher implementation
        https://github.com/CuriousAI/mean-teacher/
        '''
        # Per the mean-teacher paper, we use the global average
        # of parameter values until the exponential average is more effective
        # For a `decay_coef ~= 0.997`, this hand-off happens at ~step 333.
        alpha = min(1 - 1 / (self.step+1), self.decay_coef)
        # Perform in-place operations on the teacher parameters to average
        # with the new model parameters
        # Here, we're computing a simple weighted average where alpha is
        # the weight on past parameters, and (1 - alpha) is the weight on
        # new parameters
        zipped_params = zip(self.teacher.parameters(), model.parameters())
        for teacher_param, model_param in zipped_params:
            (teacher_param
             .data.mul_(alpha)
             .add_(1 - alpha, model_param.data))
        return

    def __call__(
        self,
        model: nn.Module,
        unlabeled_sample: dict,        
        labeled_sample: dict,
    ) -> torch.FloatTensor:
        '''Takes a model and set of unlabeled samples as input
        and computes the Interpolation Consistency Loss.

        Parameters
        ----------
        model : nn.Module
            model with parameters accessible via the `.parameters()`
            method.
        unlabeled_sample : dict
            input - torch.FloatTensor
                [Batch, Features] minibatch of unlabeled samples.
            output - torch.LongTensor
                zeros.
        labeled_sample : dict
            input - torch.FloatTensor
                [Batch, Features] minibatch of labeled examples.
            output - torch.LongTensor 
                one-hot labels.

        Returns
        -------
        supervised_loss : torch.FloatTensor
            supervised loss computed using `sup_criterion` between
            model predictions on mixed observations and true labels.
        unsupervised_loss : torch.FloatTensor
            unsupervised loss computed using `criterion` and the 
            interpolation consistency method.
        supervised_outputs : torch.FloatTensor
            [Batch, Classes] model outputs for augmented labeled examples.
            

        Notes
        -----
        Algorithm description:

        (0) Update the mean teacher.
        (1) Compute "fake labels" for unlabeled samples by performing
        a forward pass through the "mean teacher" and using the output
        as a representative label for the sample.
        (2) Perform a MixUp operation on unlabeled samples and their
        corresponding fake labels.
        (3) Compute the loss criterion between the mixed-up fake labels
        and the predicted fake labels for the mixed up samples.
        '''
        ###############################
        # (0) Update the mean teacher
        ###############################

        self._update_teacher(model,)

        ###############################
        # (1) Compute Fake Labels
        ###############################
        
        with torch.no_grad():
            fake_y = F.softmax(
                self.teacher(unlabeled_sample['input']),
                dim=1,
            )

        ###############################
        # (2) Perform MixUp and Forward
        ###############################

        unlabeled_sample['output'] = fake_y

        mixed_sample = self.mixup_op(unlabeled_sample)
        # move sample to model device if necessary
        mixed_sample['input'] = mixed_sample['input'].to(
            device=next(model.parameters()).device,
        )
        mixed_output = F.softmax(
            model(mixed_sample['input']),
        )
        assert mixed_output.requires_grad
        
        # set outputs as attributes for later access
        self.mixed_output = mixed_output
        self.mixed_sample = mixed_sample
        self.unlabeled_sample = unlabeled_sample

        ###############################
        # (3) Compute unsupervised loss
        ###############################

        icl = self.unsup_criterion(
            mixed_output,
            fake_y,
        )
        
        ###############################
        # (4) Compute supervised loss
        ###############################
        
        if self.augment is not None:
            labeled_sample = self.augment(labeled_sample)
        # move sample to the model device if necessary
        labeled_sample['input'] = labeled_sample['input'].to(
            device=next(model.parameters()).device,
        )
        labeled_sample['input'].requires_grad = True
        
        sup_outputs = model(labeled_sample['input'])
        sup_loss = self.sup_criterion(
            sup_outputs,
            labeled_sample['output'],
        )
        
        self.step += 1
        return sup_loss, icl, sup_outputs
    
    
def sharpen_labels(
    q: torch.FloatTensor,
    T: float=0.5,
) -> torch.FloatTensor:
    '''Reduce the entropy of a categorical label using a 
    temperature adjustment
    
    Parameters
    ----------
    q : torch.FloatTensor
        [N, C] pseudolabel.
    T : float
        temperature parameter.
    
    Returns
    -------
    q_s : torch.FloatTensor
        [C,] sharpened pseudolabel.
        
    Notes
    -----
    .. math::
    
        S(q, T) = q_i^{1/T} / \sum_j^L q_j^{1/T}

    '''
    if T == 0.:
        # equivalent to argmax
        _, idx = torch.max(q, dim=1)
        oh = torch.nn.functional.one_hot(
            idx,
            num_classes=q.size(1),
        )
        return oh
    
    if T == 1.0:
        # no-op
        return q
    
    q = torch.pow(q, 1. / T)
    q /= torch.sum(q, dim=1,).reshape(-1, 1)
    return q

    
class MixMatchLoss(InterpolationConsistencyLoss):
    '''Compute the MixMatch Loss given a batch of labeled
    and unlabeled examples.
    
    Attributes
    ----------
    n_augmentations : int
        number of augmentated samples to average across when
        computing pseudolabels.
        default = 2 from MixMatch paper.         
    T : float
        temperature parameter.
    augment_pseudolabels : bool
        perform augmentations during pseudolabel generation.
    pseudolabel_min_confidence : float
        minimum confidence to compute a loss for a given pseudolabeled
        example. examples below this confidence threshold will be given
        `0` loss. see the `FixMatch` paper for discussion.
    teacher : nn.Module
        teacher model for pseudolabeling.
    running_confidence_scores : list
        [n_batches_to_store,] (torch.Tensor, torch.Tensor,) of unlabeled
        example (Confident_Bool, BestConfidenceScore) tuples.
    n_batches_to_store : int
        determines how many batches to keep in `running_confidence_scores`.
    '''
    
    def __init__(
        self,
        n_augmentations: int=2,
        T: float=0.5,
        augment_pseudolabels: bool=True,
        pseudolabel_min_confidence: float=0.0,
        **kwargs,
    ) -> None:
        '''Compute the MixMatch Loss given a batch of labeled
        and unlabeled examples.
        
        Parameters
        ----------
        n_augmentations : int
            number of augmentated samples to average across when
            computing pseudolabels.
            default = 2 from MixMatch paper.         
        T : float
            temperature parameter.
        augment_pseudolabels : bool
            perform augmentations during pseudolabel generation.
        pseudolabel_min_confidence : float
            minimum confidence to compute a loss for a given pseudolabeled
            example. examples below this confidence threshold will be given
            `0` loss. see the `FixMatch` paper for discussion.
        
        Returns
        -------
        None.
        
        References
        ----------
        MixMatch: A Holistic Approach to Semi-Supervised Learning
        http://papers.nips.cc/paper/8749-mixmatch-a-holistic-approach-to-semi-supervised-learning
        
        FixMatch: https://arxiv.org/abs/2001.07685
        '''
        # inherit from IC loss, forcing the SampleMixUp to keep 
        # the identity of the dominant observation in each mixed sample
        super(MixMatchLoss, self).__init__(
            **kwargs, 
            keep_dominant_obs=True,
        )
        if not callable(self.augment):
            msg = 'MixMatch requires a Callable for augment'
            raise TypeError(msg)
        self.n_augmentations = n_augmentations
        self.augment_pseudolabels = augment_pseudolabels
        self.T = T
        
        self.pseudolabel_min_confidence = pseudolabel_min_confidence
        # keep a running score of the last 50 batches worth of pseudolabel
        # confidence outcomes
        self.n_batches_to_store = 50
        self.running_confidence_scores = []
        return
    
    
    @torch.no_grad()
    def _generate_labels(
        self,
        unlabeled_sample: dict,
    ) -> torch.FloatTensor:
        '''Generate labels by applying a set of augmentations
        to each unlabeled example and keeping the mean.
        
        Parameters
        ----------
        unlabeled_batch : dict
            "input" - [Batch, Features] minibatch of unlabeled samples.        
        '''
        # let the teacher model take guesses at the label for augmented
        # versions of the unlabeled observations
        raw_guesses = []
        for i in range(self.n_augmentations):
            to_augment = {
                'input': unlabeled_sample['input'].clone(),
                'output' : torch.zeros(1),
            }
            if self.augment_pseudolabels:
                # augment the batch before pseudolabeling
                augmented_batch = self.augment(to_augment)
            else:
                augmented_batch = to_augment
            # convert model guess to probability distribution `q`
            # with softmax, prior to considering it a label
            guess = F.softmax(
                self.teacher(augmented_batch['input']),
                dim=1,
            )
            raw_guesses.append(guess)

        # compute pseudolabels as the mean across all label guesses
        pseudolabels = torch.mean(
            torch.stack(
                raw_guesses, 
                dim=0,
            ),
            dim=0,
        )
        
        # before sharpening labels, determine if the labels are 
        # sufficiently confidence to use
        highest_conf, likeliest_class = torch.max(
            pseudolabels,
            dim=1,
        )
        # confident is a bool that we will use to decide if we should
        # keep loss from a given example or zero it out
        confident = highest_conf >= self.pseudolabel_min_confidence
        # store confidence outcomes in a running list so we can monitor
        # which fraction of pseudolabels are being used
        if len(self.running_confidence_scores) > self.n_batches_to_store:
            # remove the oldest batch
            self.running_confidence_scores.pop(0)

        # store tuples of (torch.Tensor, torch.Tensor)
        # (confident_bool, highest_conf_score)
        self.running_confidence_scores.append(
            (
                confident.detach().cpu(),
                highest_conf.detach().cpu(),
            ),
        )
        
        if self.T is not None:
            # sharpen labels
            pseudolabels = sharpen_labels(
                q=pseudolabels,
                T=self.T,
            )
        # ensure pseudolabels aren't attached to the
        # computation graph
        pseudolabels = pseudolabels.detach()
        
        return pseudolabels, confident
    
    
    def __call__(
        self,
        model: nn.Module,
        unlabeled_sample: dict,
        labeled_sample: dict,
    ) -> (torch.FloatTensor, torch.FloatTensor):
        '''
        Parameters
        ----------
        model : nn.Module
            model with parameters accessible via the `.parameters()`
            method.        
        unlabeled_sample : dict
            input - torch.FloatTensor
                [Batch, Features] minibatch of unlabeled samples.
            output - torch.LongTensor
                zeros.
        labeled_sample : dict
            input - torch.FloatTensor
                [Batch, Features] minibatch of labeled examples.
            output - torch.LongTensor 
                one-hot labels.
            
        Returns
        -------
        supervised_loss : torch.FloatTensor
            supervised loss computed using `sup_criterion` between
            model predictions on mixed observations and true labels.
        unsupervised_loss : torch.FloatTensor
            unsupervised loss computed using `criterion` between
            model predictions on mixed unlabeled observations 
            and pseudolabels generated as the mean
            across `n_augmentations` augmentation runs.
        supervised_outputs : torch.FloatTensor
            [Batch, Classes] model outputs for augmented labeled examples.
        '''
        
        ########################################
        # (0) Update the mean teacher
        ########################################

        self._update_teacher(model,)
        
        ########################################
        # (1) Generate labels for unlabeled data
        ########################################
        
        pseudolabels, pseudolabel_confidence = self._generate_labels(
            unlabeled_sample = unlabeled_sample,
        )
        # make sure pseudolabels match real label dtype
        # so that they can be concatenated
        pseudolabels = pseudolabels.to(
            dtype=labeled_sample['output'].dtype
        )
        
        ########################################
        # (2) Augment the labeled data
        ########################################
        
        labeled_sample = self.augment(
            labeled_sample,
        )
        
        ########################################
        # (3) Perform MixUp across both batches
        ########################################
        n_unlabeled_original = unlabeled_sample['input'].size(0)
        unlabeled_sample['output'] = pseudolabels
        
        # separate samples into confident and unconfident sample dicts
        # we only allow samples with confident pseudolabels to 
        # participate in the MixUp operation
        conf_unlabeled_sample = {}
        ucnf_unlabeled_sample = {}
        
        for k in unlabeled_sample.keys():
            conf_unlabeled_sample[k] = unlabeled_sample[k][pseudolabel_confidence]
            ucnf_unlabeled_sample[k] = unlabeled_sample[k][~pseudolabel_confidence]
        
        # unlabeled samples come BEFORE labeled samples
        # in the concatenated sample
        # NOTE: we only allow confident unlabeled samples
        # into the concatenated sample used for MixUp
        cat_sample = {
            k: torch.cat(
                [
                    conf_unlabeled_sample[k],
                    labeled_sample[k],
                ],
                dim=0,
            ) for k in ['input', 'output']
        }
        
        # mixup the concatenated sample
        # NOTE: dominant observations are maintained
        # by passing `keep_dominant_obs=True` in
        # `self.__init__`
        mixed_samples = self.mixup_op(
            cat_sample,
        )
        
        ########################################
        # (4) Forward pass for mixed samples
        ########################################
        
        # split the mixed samples based on the dominant
        # observation
        n_unlabeled = conf_unlabeled_sample['input'].size(0)
        unlabeled_m_ = mixed_samples['input'][:n_unlabeled]
        unlabeled_y_ = mixed_samples['output'][:n_unlabeled]
        
        labeled_m_ = mixed_samples['input'][n_unlabeled:]
        labeled_y_ = mixed_samples['output'][n_unlabeled:]
        
        # append low confidence samples to unlabeled_m_ and unlabeled_y_
        # this ensures that batch norm is still able to update it's
        # statistics based on batches from the train AND target domain
        unlabeled_m_ = torch.cat([
            unlabeled_m_,
            ucnf_unlabeled_sample['input'],
        ])
        unlabeled_y_ = torch.cat([
            unlabeled_y_,
            ucnf_unlabeled_sample['output'],
        ])          
        
        # perform a forward pass on mixed samples
        # NOTE: Our unsupervised criterion operates on post-softmax
        # probability vectors, so we transform the output here
        unlabeled_z_ = F.softmax(
            model(unlabeled_m_),
            dim=1,
        )
        # NOTE: Our supervised criterion operates directly on
        # logits and performs a `logsoftmax()` internally
        labeled_z_ = model(labeled_m_)
        
        ########################################
        # (5) Compute losses
        ########################################
        
        # compare mixed pseudolabels to the model guess
        # on the mixed input
        # NOTE: this returns an **unreduced** loss of size 
        # [Batch,] or [Batch, Classes] depending on the loss function
        unsupervised_loss = self.unsup_criterion(
            unlabeled_z_,
            unlabeled_y_,
        )
        # sum loss across classes if not reduced in the loss
        if unsupervised_loss.dim() > 1:
            unsupervised_loss = torch.sum(unsupervised_loss, dim=1)

        # scale the loss to 0 for all observations without confident pseudolabels
        # this allows the loss to slowly ramp up as labels become more confident
        scale_vec = torch.zeros_like(unsupervised_loss).float().to(
            device=unsupervised_loss.device
        )
        scale_vec[:n_unlabeled] += 1.
        unsupervised_loss = unsupervised_loss * scale_vec
        unsupervised_loss = torch.mean(unsupervised_loss)
        
        # compute model guess on the mixed supervised input
        # to the mixed labels
        # NOTE: we didn't allow non-confident pseudolabels 
        # into the MixUp, so this shouldn't propogate any
        # poor quality pseudolabel information
        supervised_loss = self.sup_criterion(
            labeled_z_,
            labeled_y_,
        )
        
        self.step += 1
        
        return supervised_loss, unsupervised_loss, labeled_z_
    
    
class DANLoss(object):
    '''Compute a domain adaptation network (DAN) loss.'''
    
    def __init__(
        self,
        dan_criterion: Callable,
        model: CellTypeCLF,
        use_conf_pseudolabels: bool=False,
        scale_loss_pseudoconf: bool=False,
        n_domains: int=2,
        **kwargs,
    ) -> None:
        '''Compute a domain adaptation network loss.
        
        Parameters
        ----------
        dan_criterion : Callable
            domain classification criterion `Callable(output, target)`.
        model : scnym.model.CellTypeCLF
            `CellTypeCLF` model to use for embedding.
        use_conf_pseudolabels : bool
            only use unlabeled observations with confident pseudolabels
            for discrimination. expects `pseudolabel_confidence` to be
            passed in the `__call__()` if so.
        scale_loss_pseudoconf : bool
            scale the weight of the gradients passed to both models based
            on the proportion of confident pseudolabels.
        n_domains : int
            number of domains of origin to predict using the adversary.
            
        Returns
        -------
        None.
        
        Notes
        -----
        **kwargs are passed to `scnym.model.DANN`
        
        See Also
        --------
        scnym.model.DANN
        '''
        self.dan_criterion = dan_criterion
        
        # build the DANN
        self.dann = DANN(
            model=model,
            n_domains=n_domains,
            **kwargs,
        )
        self.dann.domain_clf = self.dann.domain_clf.to(
            device=next(iter(model.parameters())).device,
        )
        # instantiate with small tensor to simplify downstream size
        # checking logic
        self.x_embed = torch.zeros((1,1))
        
        self.use_conf_pseudolabels = use_conf_pseudolabels
        self.scale_loss_pseudoconf = scale_loss_pseudoconf
        return
    
    def __call__(
        self,
        labeled_sample: dict,
        unlabeled_sample: dict,
        weight: float,
        pseudolabel_confidence: torch.Tensor=None,
    ) -> torch.FloatTensor:
        '''Compute the domain adaptation loss on a labeled source
        and unlabeled target domain batch.
        
        Parameters
        ----------
        unlabeled_sample : dict
            input - torch.FloatTensor
                [BatchU, Features] minibatch of unlabeled samples.
            output - torch.LongTensor
                zeros.
        labeled_sample : dict
            input - torch.FloatTensor
                [BatchL, Features] minibatch of labeled examples.
            output - torch.LongTensor 
                one-hot labels.
        weight : float
            weight for reversed gradients passed up to the embedding
            layer. gradients used for the domain classifier are normal
            gradients, but we weight and reverse the gradients flowing 
            upward to the embedding layer by this constant.
        pseudolabel_confidence : torch.Tensor
            [BatchU,] boolean identifying observations in `unlabeled_sample`
            with confident pseudolabels.
            if not None and `self.use_conf_pseudolabels`, only performs
            domain discrimination on unlabeled samples with confident
            pseudolabels.
            
        Returns
        -------
        dan_loss : torch.FloatTensor
            domain adversarial loss term.
        '''
        
        ########################################
        # (1) Create domain labels
        ########################################
        
        # check if domain labels are provided, if not assume
        # train and target are separate domains
        # domain labels of -1 indicate `None` was passed as a domain label
        # to `SingleCellDS`
        if torch.sum(labeled_sample.get('domain', torch.Tensor([-1])) == -1) > 0:
            source_label = torch.zeros(labeled_sample['input'].size(0)).long()
            source_label = torch.nn.functional.one_hot(
                source_label, 
                num_classes=2,
            )
        else:
            # domain labels should already by one-hot
            source_label = labeled_sample['domain']
        source_label = source_label.to(device=labeled_sample['input'].device)
        
        if torch.sum(unlabeled_sample.get('domain', torch.Tensor([-1])) == -1) > 0:
            target_label = torch.ones(unlabeled_sample['input'].size(0)).long()
            target_label = torch.nn.functional.one_hot(
                target_label, 
                num_classes=2,
            )
        else:
            target_label = unlabeled_sample['domain']
        target_label = target_label.to(device=unlabeled_sample['input'].device)
        
        lx = labeled_sample['input']
        ux = unlabeled_sample['input']
        
        ########################################
        # (2) Check confidence of unlabeled obs
        ########################################
        
        if self.use_conf_pseudolabels and pseudolabel_confidence is not None:
            # check confidence of unlabeled observations and remove
            # any unconfident observations from the minibatch
            ux = ux[pseudolabel_confidence]
            target_label = target_label[pseudolabel_confidence]
            # store the number of confident unlabeled obs
        self.n_conf_pseudolabels = ux.size(0)
        self.n_total_unlabeled = unlabeled_sample['input'].size(0)
        p_conf_pseudolabels = self.n_conf_pseudolabels / self.n_total_unlabeled
        ########################################
        # (3) Embed points and Classify domains
        ########################################
        
        x = torch.cat([lx,ux], 0)
        dlabel = torch.cat([source_label, target_label], 0)
        
        self.dann.set_rev_grad_weight(weight=weight)
        domain_pred, x_embed = self.dann(x)
        
        # store embeddings and labels
        if x_embed.size(0) >= self.x_embed.size(0):
            self.x_embed = copy.copy(
                x_embed.detach().cpu()
            )
            self.dlabel  = copy.copy(
                dlabel.detach().cpu()         
            )
        
        ########################################
        # (4) Compute DAN loss
        ########################################
        
        dan_loss = self.dan_criterion(
            domain_pred,
            dlabel,
        )
        
        ########################################
        # (5) Compute DAN accuracy for logs
        ########################################
        
        _, dan_pred = torch.max(domain_pred, dim=1)
        _, dlabel_int = torch.max(dlabel, dim=1)
        self.dan_acc = torch.sum(
            dan_pred == dlabel_int,
        ) / float(dan_pred.size(0))
        
        if self.scale_loss_pseudoconf:
            dan_loss *= p_conf_pseudolabels

        return dan_loss


'''Loss weight scheduling'''


class ICLWeight(object):

    def __init__(
        self,
        ramp_epochs: int,
        burn_in_epochs: int = 0,
        max_unsup_weight: float = 10.0,
        sigmoid: bool = False,
    ) -> None:
        '''Schedules the interpolation consistency loss 
        weights across a set of epochs.

        Parameters
        ----------
        ramp_epochs : int
            number of epochs to increase the unsupervised
            loss weight until reaching a maximum value.
        burn_in_epochs : int
            epochs to wait before increasing the unsupervised loss.
        max_unsup_weight : float
            maximum weight for the unsupervised loss component.
        sigmoid : bool
            scale weight using a sigmoid function.

        Returns
        -------
        None.
        '''
        self.ramp_epochs = ramp_epochs
        self.burn_in_epochs = burn_in_epochs
        self.max_unsup_weight = max_unsup_weight
        self.sigmoid = sigmoid
        # don't allow division by zero, set step size manually
        if self.ramp_epochs == 0.:
            self.step_size = self.max_unsup_weight
        else:
            self.step_size = self.max_unsup_weight / self.ramp_epochs
        print('Scaling ICL over %d epochs, %d epochs for burn in.'
              % (self.ramp_epochs, self.burn_in_epochs))
        return
    
    def _get_weight(
        self,
        epoch: int,
    ) -> float:
        '''Compute the current weight'''
        if epoch >= (self.ramp_epochs + self.burn_in_epochs):
            weight = self.max_unsup_weight
        elif self.sigmoid:
            x = (epoch - self.burn_in_epochs) / self.ramp_epochs
            coef = np.exp(-5 * (x-1)**2)
            weight = coef * self.max_unsup_weight
        else:
            weight = self.step_size*(epoch-self.burn_in_epochs)
            
        return weight

    def __call__(
        self, 
        epoch: int,
    ) -> float:
        '''Compute the weight for an unsupervised IC loss
        given the epoch.

        Parameters
        ----------
        epoch : int
            current training epoch.

        Returns
        -------
        weight : float
            weight for the unsupervised component of IC loss.
        '''
        if type(epoch) != int:
            raise TypeError(f'epoch must be int, you passed a {type(epoch)}')
        if epoch < self.burn_in_epochs:
            weight = 0.
        else:
            weight = self._get_weight(epoch)
        return weight
