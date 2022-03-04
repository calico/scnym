import numpy as np
import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import logging
from typing import Callable, Iterable, Union, List
from .dataprep import SampleMixUp
from .utils import compute_entropy_of_mixing
from .model import CellTypeCLF, DANN
import copy
from torch.utils.tensorboard import SummaryWriter

from .dataprep import SampleMixUp
from .utils import compute_entropy_of_mixing
from .model import CellTypeCLF, DANN, AE
from .losses import *


logger = logging.getLogger(__name__)


class Trainer(object):
    """
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
    """

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
        exp_name: str = "",
        reg_criterion: Callable = None,
        use_gpu: bool = torch.cuda.is_available(),
        verbose: bool = False,
        save_freq: int = 10,
        scheduler: torch.optim.lr_scheduler = None,
        tb_writer: str = None,
    ) -> None:
        """
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
        """
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
        self.best_acc = 0.0
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

        self.log_path = os.path.join(self.out_path, "_".join([exp_name, "log.csv"]))
        with open(self.log_path, "w") as f:
            header = "Epoch,Running_Loss,Mode\n"
            f.write(header)

        self.parameters = {
            "out_path": out_path,
            "exp_name": exp_name,
            "n_epochs": n_epochs,
            "use_cuda": self.use_gpu,
            "train_batch_size": self.dataloaders["train"].batch_size,
            "val_batch_size": self.dataloaders["val"].batch_size,
            "train_batch_sampler": str(type(self.dataloaders["train"].sampler)),
            "val_batch_sampler": str(type(self.dataloaders["val"].sampler)),
            "optimizer_type": str(type(self.optimizer)),
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "model_hidden": self.model.n_hidden,
            "model_ngenes": self.model.n_genes,
            "model_ncelltypes": self.model.n_cell_types,
        }

        # write the log file header
        with open(self.log_path, "w") as f:
            header = "Epoch,Iter,Running_Loss,Mode\n"
            f.write(header)

    def train_epoch(self):
        """Perform training across one full iteration through
        the data.
        """
        self.model.train(True)
        i = 0
        running_loss = 0.0
        running_corrects = 0.0
        running_total = 0.0

        btrans = self.batch_transformers.get("train", None)
        for data in self.dataloaders["train"]:
            # if a batch transformer is present,
            # transform the data before use
            if btrans is not None:
                data = btrans(data)

            inputs = data["input"]
            labels = data["output"]  # one-hot

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
                print("batch loss: ", loss.item())
            if np.isnan(loss.data.cpu().numpy()):
                raise RuntimeError("NaN loss encountered in training")

            # compute gradients in a backward pass, update parameters
            loss.backward()
            self.optimizer.step()

            # statistics update
            running_loss += loss.item() / inputs.size(0)
            running_corrects += float(correct.item())
            running_total += float(labels.size(0))

            if i % 100 == 0 and self.verbose:
                print("Iter : ", i)
                print("running_loss : ", running_loss / (i + 1))
                print("running_acc  : ", running_corrects / running_total)
                print("corrects: %f | total: %f" % (running_corrects, running_total))
                # append to log
                with open(self.log_path, "a") as f:
                    f.write(
                        str(self.epoch)
                        + ","
                        + str(i)
                        + ","
                        + str(running_loss / (i + 1))
                        + ",train\n"
                    )
            i += 1

        epoch_loss = running_loss / len(self.dataloaders["train"])
        epoch_acc = running_corrects / running_total

        # append to log
        with open(self.log_path, "a") as f:
            f.write(
                str(self.epoch)
                + ","
                + str(i)
                + ","
                + str(running_loss / (i + 1))
                + ",train_epoch\n"
            )

        if self.tb_writer is not None:
            self.tb_writer.add_scalar("Loss/train", epoch_loss, self.epoch)
            self.tb_writer.add_scalar("Acc/train", epoch_acc, self.epoch)
            for i, p in enumerate(self.model.parameters()):
                self.tb_writer.add_histogram(
                    f"Grad/param{i:04}",
                    p.grad,
                    self.epoch,
                )

            self.tb_writer.add_scalar(
                "lr/lr",
                self.optimizer.state_dict()["param_groups"][0]["lr"],
                self.epoch,
            )

        if self.verbose:
            print("{} Loss : {:.4f}".format("train", epoch_loss))
            print("{} Acc : {:.4f}".format("train", epoch_acc))
            print(
                "TRAIN EPOCH corrects: %f | total: %f"
                % (running_corrects, running_total)
            )

    @torch.no_grad()
    def val_epoch(self):
        """Perform a pass through the validation data.
        Do not record gradients to speed things up.
        """
        self.model.train(False)
        i = 0
        running_loss = 0.0
        running_corrects = 0
        running_total = 0

        btrans = self.batch_transformers.get("val", None)
        for data in self.dataloaders["val"]:
            # if a batch transformer is present,
            # transform the data before use
            if btrans is not None:
                data = btrans(data)

            inputs = data["input"]
            labels = data["output"]  # one-hot
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
                print("PRED\n", predictions[:10, ...])
                print("LABEL\n", int_labels[:10, ...])
                print("CORRECT: ", correct)

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
                print("Iter : ", i)
                print("running_loss : ", running_loss / (i + 1))
                print("running_acc  : ", running_corrects / running_total)
                print("corrects: %f | total: %f" % (running_corrects, running_total))
                # append to log
                with open(self.log_path, "a") as f:
                    f.write(
                        str(self.epoch)
                        + ","
                        + str(i)
                        + ","
                        + str(running_loss / (i + 1))
                        + ",val\n"
                    )
            i += 1

        epoch_loss = running_loss / len(self.dataloaders["val"])
        epoch_acc = running_corrects / running_total
        # append to log
        with open(self.log_path, "a") as f:
            f.write(
                str(self.epoch)
                + ","
                + str(i)
                + ","
                + str(running_loss / (i + 1))
                + ",val_epoch\n"
            )

        # add one epoch to the waiting time for best loss
        # if we had a new best loss, the counter is reset below
        self.waiting_time += 1
        if (epoch_loss < self.best_loss) and (self.epoch >= self.min_epochs):
            self.best_loss = epoch_loss
            self.best_model_wts = self.model.state_dict()
            self.waiting_time = 0
            torch.save(
                self.model.state_dict(),
                os.path.join(
                    self.out_path,
                    ("model_weights_" + str(self.epoch).zfill(3) + ".pkl"),
                ),
            )
            print("Saving best model weights...")
            torch.save(
                self.model.state_dict(),
                os.path.join(self.out_path, "00_best_model_weights.pkl"),
            )
            print("Saved best weights.")

            if hasattr(self, "dan_criterion"):
                print("Trainer has a `dan_criterion`.")
                if self.dan_criterion is not None:
                    print("Saving DAN weights...")
                    for i_adv, adv in enumerate(self.dan_criterion.dann):
                        weights = adv.state_dict()
                        torch.save(
                            weights,
                            os.path.join(
                                self.out_path,
                                f"02_best_dan_weights_{i_adv:03}.pkl",
                            ),
                        )

            with open(self.log_path, "a") as f:
                f.write(
                    str(self.epoch)
                    + ","
                    + str(i)
                    + ","
                    + str(running_loss / (i + 1))
                    + ",best_model_weights\n",
                )

            if self.tb_writer is not None:
                self.tb_writer.add_text(
                    "BestWeights",
                    f"Saved best weights at {self.epoch}, loss {epoch_loss}",
                    self.epoch,
                )
                self.tb_writer.flush()

        elif self.epoch % self.save_freq == 0:
            torch.save(
                self.model.state_dict(),
                os.path.join(
                    self.out_path,
                    "model_weights_" + str(self.epoch).zfill(3) + ".pkl",
                ),
            )

        elif self.epoch == (self.n_epochs - 1):
            torch.save(
                self.model.state_dict(),
                os.path.join(self.out_path, "01_final_model_weights.pkl"),
            )
        if self.verbose:
            print(f"{self.waiting_time} epochs since last best weights.\n")

        if self.tb_writer is not None:
            self.tb_writer.add_scalar("Loss/val", epoch_loss, self.epoch)
            self.tb_writer.add_scalar("Acc/val", epoch_acc, self.epoch)
            self.tb_writer.flush()

        if self.verbose:
            print("{} Loss : {:.4f}".format("val", epoch_loss))
            print("{} Acc : {:.4f}".format("val", epoch_acc))
            print(
                "VAL EPOCH corrects: %f | total: %f" % (running_corrects, running_total)
            )

    def train(self):
        for epoch in range(self.n_epochs):
            self.epoch = epoch
            msg = f"Epoch {epoch}/{self.n_epochs-1}"
            p_complete = epoch / self.n_epochs
            n_bars = int(np.floor(30 * p_complete))
            msg += "|" + "-" * n_bars + "_" * (30 - n_bars) + "|"
            # print a new line so the progress bar isn't overwritten
            # on the final stdout
            end_char = "\n" if epoch == (self.n_epochs - 1) else "\r"
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
                logger.info(">" * 5)
                logger.info(f"Early stopping at epoch {self.epoch}")
                logger.info(">" * 5)
                break

        # save final model weights
        torch.save(
            self.model.state_dict(),
            os.path.join(self.out_path, "01_final_model_weights.pkl"),
        )            

        self.model.load_state_dict(
            torch.load(
                os.path.join(
                    self.out_path,
                    "00_best_model_weights.pkl",
                )
            )
        )

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
        dan_criterion: Callable = None,
        dan_weight: Callable = None,
        **kwargs,
    ) -> None:
        """Train a PyTorch model using both a supervised and
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
        """
        super(SemiSupervisedTrainer, self).__init__(**kwargs)
        self.unsup_criterion = unsup_criterion
        self.unsup_dataloader = unsup_dataloader
        self.unsup_weight = unsup_weight
        self.dan_criterion = dan_criterion
        if self.dan_criterion is not None:
            print("Using a Domain Adaptation Loss.")
        self.dan_weight = dan_weight
        return

    def train_epoch(
        self,
    ) -> None:
        """
        Perform training using both a supervised and semi-supervised loss.

        Notes
        -----
        (1) Sample labeled examples, compute the standard supervised loss.
        (2) Sample unlabeled examples, compute unsupervised loss.
        (3) Perform backward pass and update parameters.
        """
        self.model.train(True)
        i = 0
        running_loss = 0.0
        running_sup_loss = 0.0  # supervised loss
        running_uns_loss = 0.0  # unsupervised loss
        running_dom_loss = 0.0  # domain adaptation loss
        running_corrects = 0.0
        running_total = 0.0

        btrans = self.batch_transformers.get("train", None)

        iter_unsup_dl = iter(self.unsup_dataloader)
        for data in self.dataloaders["train"]:

            ####################################
            # (1) Prepare data and graph
            ####################################

            # get unlabeled batch
            unsup_data = next(iter_unsup_dl)

            if btrans is not None:
                data = btrans(data)

            if self.use_gpu:
                # push all the data to the CUDA device
                data["input"] = data["input"].cuda()
                data["output"] = data["output"].cuda()

                unsup_data["input"] = unsup_data["input"].cuda()

            # capture gradients on labeled and unlabeled inputs
            # do not store gradients on labels
            data["input"].requires_grad = True
            data["output"].requires_grad = False

            unsup_data["input"].requires_grad = True

            # zero gradients across the graph
            self.optimizer.zero_grad()

            ####################################
            # (2) Compute loss terms
            ####################################

            sup_loss, unsup_loss, sup_outputs = self.unsup_criterion(
                model=self.model,
                labeled_sample=data,
                unlabeled_sample=unsup_data,
            )

            # check supervised classification accuracy
            _, predictions = torch.max(sup_outputs, 1)
            int_labels = torch.argmax(data["output"], 1)

            correct = torch.sum(predictions.detach() == int_labels.detach())

            # compute regularization loss
            if self.reg_criterion is not None:
                reg_loss = self.reg_criterion(self.model)
            else:
                reg_loss = 0.0

            # compute the domain adaptation loss if desired
            if self.dan_criterion is not None:
                dan_weight = self.dan_weight(self.epoch)
                # NOTE: pseudolabel confidence is only used if `use_conf_pseudolabels`
                # was passed to the initiatilization of `DANLoss`
                pseudolabel_confidence = self.unsup_criterion.running_confidence_scores[
                    -1
                ][0]
                dan_loss = self.dan_criterion(
                    labeled_sample=data,
                    unlabeled_sample=unsup_data,
                    weight=dan_weight,
                    pseudolabel_confidence=pseudolabel_confidence,
                )
            else:
                dan_loss = torch.zeros(
                    1,
                ).float()
                dan_loss = dan_loss.to(device=sup_loss.device)
                dan_weight = 0.0

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
                print("sup.  loss:   ", sup_loss.item())
                print("usup. loss:   ", unsup_loss.item())
                print("usup. weight: ", self.unsup_weight(self.epoch))
                if self.dan_criterion is not None:
                    print("Dom.  loss:   ", dan_loss.item())
                    print("Dom.  weight: ", dan_weight)
                print("total loss: ", loss.item())
            if np.isnan(loss.data.cpu().numpy()):
                raise RuntimeError("NaN loss encountered in training")

            # compute gradients in a backward pass, update parameters
            loss.backward()
            self.optimizer.step()

            # statistics update
            labeled_n = data["input"].size(0)
            unlabel_n = unsup_data["input"].size(0)

            running_loss += loss.item()
            running_sup_loss += sup_loss.item()
            running_uns_loss += unsup_loss.item()
            running_dom_loss += dan_loss.item()
            running_corrects += float(correct.item())
            running_total += float(data["input"].size(0))

            if i % 100 == 0 and self.verbose:
                print("Iter : ", i)
                print("running_sup_loss : ", running_sup_loss / (i + 1))
                print("running_uns_loss : ", running_uns_loss / (i + 1))
                print("running_dom_loss : ", running_dom_loss / (i + 1))
                print("running_loss : ", running_loss / (i + 1))
                print("running_acc  : ", running_corrects / running_total)
                print("corrects: %f | total: %f" % (running_corrects, running_total))
                # append to log
                with open(self.log_path, "a") as f:
                    f.write(
                        str(self.epoch)
                        + ","
                        + str(i)
                        + ","
                        + str(running_loss / (i + 1))
                        + ",train\n"
                    )
            i += 1

        epoch_sup_loss = running_sup_loss / len(self.dataloaders["train"])
        epoch_uns_loss = running_uns_loss / len(self.dataloaders["train"])
        epoch_dom_loss = running_dom_loss / len(self.dataloaders["train"])
        epoch_loss = running_loss / len(self.dataloaders["train"])
        epoch_acc = running_corrects / running_total

        if self.tb_writer is not None:
            self.tb_writer.add_scalar(
                "Loss/train",
                epoch_loss,
                self.epoch,
            )
            self.tb_writer.add_scalar(
                "Acc/train",
                epoch_acc,
                self.epoch,
            )
            self.tb_writer.add_scalar(
                "Loss/super",
                epoch_sup_loss,
                self.epoch,
            )
            self.tb_writer.add_scalar(
                "Loss/unsup",
                epoch_uns_loss,
                self.epoch,
            )
            self.tb_writer.add_scalar(
                "SSL/UnsWeight",
                self.unsup_weight(self.epoch),
                self.epoch,
            )
            if self.dan_criterion is not None:
                self.tb_writer.add_scalar(
                    "Loss/domain",
                    epoch_dom_loss,
                    self.epoch,
                )
                self.tb_writer.add_scalar(
                    "SSL/DomWeight",
                    self.dan_weight(self.epoch),
                    self.epoch,
                )

                # add embedding
                dlabel = self.dan_criterion.dlabel.numpy()
                self.tb_writer.add_embedding(
                    self.dan_criterion.x_embed,
                    metadata=dlabel.tolist(),
                    global_step=self.epoch,
                    tag="Embed/DAN",
                )

                # compute the entropy of mixing
                dan_embedding = self.dan_criterion.x_embed.numpy()

                eom = compute_entropy_of_mixing(
                    X=dan_embedding,
                    y=dlabel[:, 0],
                    n_neighbors=100,
                    n_iters=512,
                    n_jobs=-1,
                )
                self.tb_writer.add_scalar(
                    "SSL/entropy_of_mixing",
                    np.mean(eom),
                    self.epoch,
                )
                self.tb_writer.add_histogram(
                    "SSL/dist_entropy_of_mixing",
                    eom,
                    self.epoch,
                )
                self.tb_writer.add_scalar(
                    "SSL/domain_acc",
                    self.dan_criterion.dan_acc,
                    self.epoch,
                )

                for i_adv, adv in enumerate(self.dan_criterion.dann):
                    for i_param, param in enumerate(
                        adv.domain_clf.parameters()
                    ):
                        self.tb_writer.add_histogram(
                            f"Grad/domain_clf_adv{i_adv:03}_{i_param:04}",
                            param.grad,
                            self.epoch,
                        )
                self.tb_writer.add_scalar(
                    "SSL/dan_n_conf_pseudolabels",
                    self.dan_criterion.n_conf_pseudolabels,
                    self.epoch,
                )
                self.tb_writer.add_scalar(
                    "SSL/dan_p_conf_pseudolabels",
                    self.dan_criterion.n_conf_pseudolabels
                    / self.dan_criterion.n_total_unlabeled,
                    self.epoch,
                )

            self.tb_writer.flush()

            for i, named_mod in enumerate(self.model.classif.named_modules()):
                module_name = named_mod[0]
                module = named_mod[1]
                for j, param in enumerate(module.parameters()):
                    self.tb_writer.add_histogram(
                        f"Grad/{module_name}/{j:04}",
                        param.grad,
                        self.epoch,
                    )

            # add the running confidence scores of unlabeled examples
            # if we're using MixMatch
            if hasattr(self.unsup_criterion, "running_confidence_scores"):
                # get the number of confident pseudolabels
                # and the total number of pseudolabels per batch
                n_conf = torch.Tensor(
                    [
                        torch.sum(s[0]).item()
                        for s in self.unsup_criterion.running_confidence_scores
                    ]
                )
                n_total = torch.Tensor(
                    [
                        s[0].size(0)
                        for s in self.unsup_criterion.running_confidence_scores
                    ]
                )
                conf_dist = torch.cat(
                    [s[1] for s in self.unsup_criterion.running_confidence_scores],
                    dim=0,
                )
                self.tb_writer.add_scalar(
                    "SSL/p_conf_pseudolabels",
                    torch.sum(n_conf) / torch.sum(n_total),
                    self.epoch,
                )
                self.tb_writer.add_scalar(
                    "SSL/avg_pseudolabel_conf",
                    torch.mean(conf_dist),
                    self.epoch,
                )
                self.tb_writer.add_histogram(
                    "SSL/dist_p_conf_pseudolabels",
                    n_conf / n_total,
                    self.epoch,
                )
                self.tb_writer.add_histogram(
                    "SSL/pseudolabel_conf",
                    conf_dist,
                    self.epoch,
                )

        # append to log
        with open(self.log_path, "a") as f:
            f.write(
                str(self.epoch)
                + ","
                + str(i)
                + ","
                + str(epoch_loss)
                + ",train_epoch\n"
            )
            # write out the supervised and unsupervised components
            # of loss separately
            f.write(
                str(self.epoch)
                + ","
                + str(i)
                + ","
                + str(epoch_sup_loss)
                + ",train_epoch_sup\n"
            )
            f.write(
                str(self.epoch)
                + ","
                + str(i)
                + ","
                + str(epoch_uns_loss)
                + ",train_epoch_uns\n"
            )
            f.write(
                str(self.epoch)
                + ","
                + str(i)
                + ","
                + str(self.unsup_weight(self.epoch))
                + ",train_epoch_uns_weight\n"
            )
        if self.verbose:
            print("{} Sup. Loss : {:.6f}".format("train", epoch_sup_loss))
            print("{} Unsup. Loss : {:.6f}".format("train", epoch_uns_loss))
            print(
                "{} Unsup. Weight : {:.6f}".format(
                    "train", self.unsup_weight(self.epoch)
                )
            )
            if self.dan_criterion is not None:
                print("{} Dom.  Loss : {:.6f}".format("train", epoch_dom_loss))
                print(f"train Dom.  Weight : {self.dan_weight(self.epoch)}")
            print("{} Loss : {:.4f}".format("train", epoch_loss))
            print("{} Acc : {:.4f}".format("train", epoch_acc))
            print(
                "TRAIN EPOCH corrects: %f | total: %f"
                % (running_corrects, running_total)
            )
        return


class MultiTaskTrainer(Trainer):
    def __init__(
        self,
        criteria: List[dict],
        unsup_dataloader: torch.utils.data.DataLoader = None,
        **kwargs,
    ) -> None:
        """Train a multitask model with multiple criteria using
        labeled and unlabeled dataloaders.

        Parameters
        ----------
        criteria : List[dict]
            dictionary describing a single task criterion, containing keys.
                function - callable with `dict` kwargs `labeled_sample`
                    and `unlabeled_sample`, `nn.Module` kwarg `model`,
                    a `float` kwarg `weight`, and returns `torch.FloatTensor`.
                weight - Callable, maps `int` epoch to `float` weight.
                    can also pass float value for constant weight.
                validation - bool, use criterion for validation loss.
        unsup_dataloader : torch.utils.data.DataLoader
            data loader supplying unlabeled samples.
        **kwargs : dict
            passed to `Trainer` parent. Include:
                model - nn.Module
                criterion - Callable
                optimizer - torch.optim.Optimizer
                dataloaders - dict
                out_path - str
                n_epochs - int
                min_epochs - int
                patience - int
                use_gpu - bool
                scheduler - torch.optim.lr_scheduler

        Returns
        -------
        None.

        Notes
        -----
        criteria are applied sequentially, such that values extracted in one
        criterion can be added to the dictionary and used in another.
        if a criterion has a `no_weight=True` attribute, loss weights are not
        applied in the train loop (useful for DAN, weights applied to rev'd grads).
        all criteria should implement a `.train(bool)` method, even if they do not
        contain trainable parameters.
        """
        kwargs.update({"criterion": None})
        super(MultiTaskTrainer, self).__init__(**kwargs)
        self.keep_checkpoints = False # don't save every best model to disk
        self.criteria = criteria
        # check that criteria provided are actually callable
        for c in self.criteria:
            fxn = c.get("function", None)
            weight = c.get("weight", None)
            if not callable(fxn):
                msg = "One of the criteria provided is not callable.\n"
                msg += f"\t{fxn}"
                raise ValueError(fxn)

            if not callable(weight) and type(weight) != float:
                msg = 'One of the criteria did not include a `"weight"` property.\n'
                msg += f"\t{fxn}\n"
                msg += f"\tweight : {weight}"
                raise ValueError(msg)

        self.unsup_dataloader = unsup_dataloader
        self.best_weights = None
        return

    def train_epoch(
        self,
    ) -> float:
        """Perform a training loop by evaluating all the criteria
        in `self.criteria` sequentially, then computing the weighted
        loss and backproping."""

        self.model.train(True)

        i = 0
        # setup running values for all losses
        running_losses = np.zeros(len(self.criteria))

        btrans = self.batch_transformers.get("train", None)

        if self.unsup_dataloader is not None:
            iter_unsup_dl = iter(self.unsup_dataloader)

        for data in self.dataloaders["train"]:

            ####################################
            # (1) Prepare data and graph
            ####################################

            if btrans is not None:
                data = btrans(data)

            if self.use_gpu:
                # push all the data to the CUDA device
                data["input"] = data["input"].cuda()
                data["output"] = data["output"].cuda()

            # get unlabeled batch
            if self.unsup_dataloader is not None:
                unsup_data = next(iter_unsup_dl)
                unsup_data["input"] = unsup_data["input"].to(
                    device=data["input"].device,
                )
                # unsup_data["input"].requires_grad = True
            else:
                unsup_data = None

            # capture gradients on labeled and unlabeled inputs
            # do not store gradients on labels
            # data["input"].requires_grad = True
            # data["output"].requires_grad = False

            # zero gradients across the graph
            self.optimizer.zero_grad()

            ####################################
            # (2) Compute loss terms
            ####################################

            loss = torch.zeros(
                1,
            ).to(device=data["input"].device)
            for crit_idx, crit_dict in enumerate(self.criteria):

                crit_fxn = crit_dict["function"]
                weight_fxn = crit_dict["weight"]

                crit_name = crit_fxn.__class__.__name__
                crit_name = crit_dict.get("name", crit_name)
                logger.debug(f"Computing criterion: {crit_name}")

                # get the current weight from the weight function,
                # or use the constant weight value
                weight = weight_fxn(self.epoch) if callable(weight_fxn) else weight_fxn
                # prepare crit_fxn for loss computation
                crit_fxn.train(True)
                if hasattr(crit_fxn, "epoch"):
                    # update the epoch attribute for use by any internal functions
                    crit_fxn.epoch = self.epoch

                crit_loss = crit_fxn(
                    labeled_sample=data,
                    unlabeled_sample=unsup_data,
                    model=self.model,
                    weight=weight,
                )

                if hasattr(crit_fxn, "no_weight"):
                    # don't reweight the loss, already performed
                    # internally in the criterion
                    weight = 1.0

                logger.debug(f"crit_loss: {crit_loss}")
                logger.debug(f"weight: {weight}")

                # weight losses and accumulate
                weighted_crit_loss = crit_loss * weight
                logger.debug(f"weighted_crit_loss: {weighted_crit_loss}")
                logger.debug(f"loss: {loss}, type {type(loss)}")

                loss += weighted_crit_loss

                running_losses[crit_idx] += crit_loss.item()
                if self.verbose:
                    logger.debug(f"weight {crit_name} : {weight}")
                    logger.debug(f"batch {crit_name} : {weighted_crit_loss}")

            # backprop
            loss.backward()
            # update parameters
            self.optimizer.step()

        # perform logging
        n_batches = len(self.dataloaders["train"])

        epoch_losses = running_losses / n_batches

        if self.verbose:
            for crit_idx, crit_dict in enumerate(self.criteria):
                crit_name = crit_dict["function"].__class__.__name__
                # get a stored name if it exists
                crit_name = crit_dict.get("name", crit_name)
                logger.info(f"{crit_name}: {epoch_losses[crit_idx]}")

        if self.tb_writer is not None:
            for crit_idx in range(len(self.criteria)):
                crit_dict = self.criteria[crit_idx]
                crit_name = crit_dict["function"].__class__.__name__
                crit_name = crit_dict.get("name", crit_name)
                self.tb_writer.add_scalar(
                    "loss/" + crit_name,
                    float(epoch_losses[crit_idx]),
                    self.epoch,
                )
                weight_fxn = crit_dict["weight"]
                weight = weight_fxn(self.epoch) if callable(weight_fxn) else weight_fxn
                self.tb_writer.add_scalar(
                    "weight/" + crit_name,
                    float(weight),
                    self.epoch,
                )
            # TODO save embeddings
            # self.tb_writer.add_embedding(
            #     data["embed"].detach().cpu(),
            #     metadata=data["domain"].detach().cpu().numpy().tolist(),
            #     global_step=self.epoch,
            #     tag="Embed/train",
            # )
            for param_name, values in dict(self.model.named_parameters()).items():
                self.tb_writer.add_histogram(
                    f"weight/{param_name}",
                    values=values.data.detach().cpu().view(-1),
                    global_step=self.epoch,
                )
                # grads may be `None` if parameters are frozen
                if values.grad is not None:
                    self.tb_writer.add_histogram(
                        f"grad/{param_name}",
                        values=values.grad.data.detach().cpu().view(-1),
                        global_step=self.epoch,
                    )

        return np.sum(epoch_losses)

    @torch.no_grad()
    def val_epoch(self):
        """Perform a pass through the validation data."""
        self.model.train(False)
        i = 0
        running_losses = np.zeros(len(self.criteria))
        running_corrects = 0
        running_total = 0

        if self.unsup_dataloader is not None:
            iter_unsup_dl = iter(self.unsup_dataloader)

        btrans = self.batch_transformers.get("val", None)
        for data in self.dataloaders["val"]:

            # if a batch transformer is present,
            # transform the data before use
            if btrans is not None:
                data = btrans(data)

            if self.use_gpu:
                data["input"] = data["input"].cuda()
                data["output"] = data["output"].cuda()

            if self.unsup_dataloader is not None:
                unsup_data = next(iter_unsup_dl)
                unsup_data["input"] = unsup_data["input"].to(
                    device=data["input"].device
                )
            else:
                unsup_data = None

            inputs = data["input"]
            labels = data["output"]  # one-hot

            # zero gradients
            self.optimizer.zero_grad()

            # perform a forward pass to get prediction accuracies, regardless
            # of what other tasks our model is performing
            outputs = self.model(inputs)
            _, predictions = torch.max(outputs, 1)

            # remake an integer version of the labels for quick checking
            int_labels = torch.argmax(labels, 1)
            correct = torch.sum(predictions.detach() == int_labels.detach()).item()

            running_corrects += float(correct)
            running_total += int(int_labels.size(0))

            logger.debug(f"PRED\n{predictions[:10, ...]}")
            logger.debug(f"LABEL\n{int_labels[:10, ...]}")
            logger.debug(f"CORRECT: {correct}")

            # compute losses
            losses = []
            for crit_idx, crit_dict in enumerate(self.criteria):

                if not crit_dict.get("validation", False):
                    continue

                crit_fxn = crit_dict["function"]
                weight_fxn = crit_dict["weight"]
                # get the current weight from the weight function,
                # or use the constant weight value
                weight = weight_fxn(self.epoch) if callable(weight_fxn) else weight_fxn

                crit_fxn.train(False)
                crit_loss = crit_fxn(
                    labeled_sample=data,
                    unlabeled_sample=unsup_data,
                    model=self.model,
                    weight=weight,
                )

                crit_name = crit_fxn.__class__.__name__

                if hasattr(crit_fxn, "no_weight"):
                    # don't reweight the loss, already performed
                    # internally in the criterion
                    weight = 1.0
                # weight losses and accumulate
                weighted_crit_loss = crit_loss * weight
                losses.append(weighted_crit_loss)
                running_losses[crit_idx] += weighted_crit_loss.item()

                logger.debug(f"{crit_name}: {crit_loss}")
                logger.debug(f"\tweight : {weight}")
                logger.debug(f"weighted {crit_name}: {weighted_crit_loss}")

        epoch_losses = running_losses / len(self.dataloaders["val"])
        epoch_acc = running_corrects / running_total

        epoch_loss = np.sum(epoch_losses)

        # append to log
        with open(self.log_path, "a") as f:
            f.write(
                str(self.epoch)
                + ","
                + str(i)
                + ","
                + str(epoch_loss / (i + 1))
                + ",val_epoch\n"
            )

        # add one epoch to the waiting time for best loss
        # if we had a new best loss, the counter is reset below
        self.waiting_time += 1
        if (epoch_loss < self.best_loss) and (self.epoch >= self.min_epochs):
            self.best_loss = epoch_loss
            self.waiting_time = 0
            if self.keep_checkpoints:
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.out_path, f"model_weights_{self.epoch:03d}.pkl"),
                )
            logger.info(f"Saving best model weights, epoch {self.epoch}...")
            torch.save(
                self.model.state_dict(),
                os.path.join(self.out_path, "00_best_model_weights.pkl"),
            )
            self.best_weights = copy.deepcopy(self.model.state_dict())
            logger.info("Saved best weights.")

            # also save the best weights of additional model components
            for crit_fxn in self.criteria:
                if crit_fxn["function"].__class__.__name__ == "DANLoss":
                    # save DAN weights
                    logger.info("Saving DAN weights...")
                    for i_adv, adv in enumerate(crit_fxn["function"].dann):
                        weights = adv.state_dict()
                        torch.save(
                            weights,
                            os.path.join(
                                self.out_path,
                                f"02_best_dan_weights_{i_adv:03}.pkl",
                            ),
                        )
                elif crit_fxn["function"].__class__.__name__ == "ReconstructionLoss":
                    # save AE weights
                    logger.info("Saving Reconstruction weights...")
                    weights = crit_fxn["function"].rec_model.state_dict()
                    torch.save(
                        weights,
                        os.path.join(
                            self.out_path,
                            f"03_best_reconstruction_weights.pkl",
                        ),
                    )
                else:
                    pass

            with open(self.log_path, "a") as f:
                f.write(
                    str(self.epoch)
                    + ","
                    + str(i)
                    + ","
                    + str(epoch_loss)
                    + ",best_model_weights\n",
                )

            if self.tb_writer is not None:
                self.tb_writer.add_text(
                    "BestWeights",
                    f"Saved best weights at {self.epoch}, loss {epoch_loss}",
                    self.epoch,
                )
                self.tb_writer.flush()

        elif self.epoch % self.save_freq == 0:
            torch.save(
                self.model.state_dict(),
                os.path.join(
                    self.out_path,
                    "model_weights_" + str(self.epoch).zfill(3) + ".pkl",
                ),
            )

        elif self.epoch == (self.n_epochs - 1):
            torch.save(
                self.model.state_dict(),
                os.path.join(self.out_path, "01_final_model_weights.pkl"),
            )
        if self.verbose:
            logger.info(f"{self.waiting_time} epochs since last best weights.\n")

        if self.tb_writer is not None:
            self.tb_writer.add_scalar("Loss/val", epoch_loss, self.epoch)
            self.tb_writer.add_scalar("Acc/val", epoch_acc, self.epoch)
            self.tb_writer.flush()

        if self.verbose:
            logger.info("{} Loss : {:.4f}".format("val", epoch_loss))
            logger.info("{} Acc : {:.4f}".format("val", epoch_acc))
            logger.info(
                "VAL EPOCH corrects: %f | total: %f" % (running_corrects, running_total)
            )

        return epoch_loss


"""Loss weight scheduling"""


class ICLWeight(object):
    def __init__(
        self,
        ramp_epochs: int,
        burn_in_epochs: int = 0,
        max_unsup_weight: float = 10.0,
        sigmoid: bool = False,
    ) -> None:
        """Schedules the interpolation consistency loss
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
        """
        self.ramp_epochs = ramp_epochs
        self.burn_in_epochs = burn_in_epochs
        self.max_unsup_weight = max_unsup_weight
        self.sigmoid = sigmoid
        # don't allow division by zero, set step size manually
        if self.ramp_epochs == 0.0:
            self.step_size = self.max_unsup_weight
        else:
            self.step_size = self.max_unsup_weight / self.ramp_epochs
        print(
            "Scaling ICL over %d epochs, %d epochs for burn in."
            % (self.ramp_epochs, self.burn_in_epochs)
        )
        return

    def _get_weight(
        self,
        epoch: int,
    ) -> float:
        """Compute the current weight"""
        if epoch >= (self.ramp_epochs + self.burn_in_epochs):
            weight = self.max_unsup_weight
        elif self.sigmoid:
            x = (epoch - self.burn_in_epochs) / self.ramp_epochs
            coef = np.exp(-5 * (x - 1) ** 2)
            weight = coef * self.max_unsup_weight
        else:
            weight = self.step_size * (epoch - self.burn_in_epochs)

        return weight

    def __call__(
        self,
        epoch: int,
    ) -> float:
        """Compute the weight for an unsupervised IC loss
        given the epoch.

        Parameters
        ----------
        epoch : int
            current training epoch.

        Returns
        -------
        weight : float
            weight for the unsupervised component of IC loss.
        """
        if type(epoch) != int:
            raise TypeError(f"epoch must be int, you passed a {type(epoch)}")
        if epoch < self.burn_in_epochs:
            weight = 0.0
        else:
            weight = self._get_weight(epoch)
        return weight
