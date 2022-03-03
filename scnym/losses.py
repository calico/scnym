import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Callable, Union, Iterable, Tuple
from .dataprep import SampleMixUp
from .model import CellTypeCLF, DANN, AE
from . import attributionpriors as attrprior
from .distributions import NegativeBinomial
import copy

logger = logging.getLogger(__name__)


class MultiTaskCriterion(object):
    def __init__(
        self,
    ) -> None:
        """Abstraction for MultiTask losses

        Note: Depreceated, inheriting from `torch.nn.Module` now.
        """
        return

    def train(self, on: bool) -> None:
        """Toggle the training mode of learned parameters"""
        return

    def eval(
        self,
    ) -> None:
        """Disable training of learned parameters"""
        self.train(on=False)
        return


def get_class_weight(
    y: np.ndarray,
) -> np.ndarray:
    """Generate relative class weights based on the representation
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

    """
    # find all unique class in y and their counts
    u_classes, class_counts = np.unique(y, return_counts=True)
    # compute class proportions
    class_prop = class_counts / len(y)
    # invert proportions to get class weights
    class_weight = 1.0 / class_prop
    # normalize so that the minimum value is 1.
    class_weight = class_weight / class_weight.min()
    return class_weight


def cross_entropy(
    pred_: torch.FloatTensor,
    label: torch.FloatTensor,
    class_weight: torch.FloatTensor = None,
    sample_weight: torch.FloatTensor = None,
    reduction: str = "mean",
) -> torch.FloatTensor:
    """Compute cross entropy loss for prediction outputs
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
    """
    if pred_.size() != label.size():
        msg = (
            f"pred size {pred_.size()} not compatible with label size {label.size()}\n"
        )
        raise ValueError(msg)

    if reduction.lower() not in ("mean", "sum", "none"):
        raise ValueError(f"{reduction} is not a valid reduction method.")

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
    if reduction == "mean":
        loss = torch.mean(samplewise_loss)
    elif reduction == "sum":
        loss = torch.sum(samplewise_loss)
    else:
        loss = samplewise_loss
    return loss


class scNymCrossEntropy(nn.Module):
    def __init__(
        self,
        class_weight: torch.FloatTensor = None,
        sample_weight: torch.FloatTensor = None,
        reduction: str = "mean",
    ) -> None:
        """Class wrapper for scNym cross-entropy loss to be used
        in conjuction with `MultiTaskTrainer`

        Parameters
        ----------
        class_weight : torch.FloatTensor
            [C,] relative weights for each of the output classes.
            useful for increasing attention to underrepresented
            classes.
        reduction : str
            reduction method across the batch.

        See Also
        --------
        cross_entropy
        .trainer.MultiTaskTrainer
        """
        super(scNymCrossEntropy, self).__init__()

        self.class_weight = class_weight
        self.sample_weight = sample_weight
        self.reduction = reduction
        return

    def __call__(
        self,
        labeled_sample: dict,
        unlabeled_sample: dict,
        model: nn.Module,
        weight: float = None,
    ) -> torch.FloatTensor:
        """Perform class prediction and compute the supervised loss

        Parameters
        ----------
        labeled_sample : dict
            input - torch.FloatTensor
                [Batch, Features] minibatch of labeled examples.
            output - torch.LongTensor
                one-hot labels.
        unlabeled_sample : dict
            input - torch.FloatTensor
                [Batch, Features] minibatch of unlabeled samples.
            output - torch.LongTensor
                zeros.
            pass `None` if there are no unlabeled samples.
        model : nn.Module
            model with parameters accessible via the `.parameters()`
            method.
        weight : float
            default None. no-op, included for API compatibility.


        Returns
        -------
        loss : torch.FloatTensor
        """
        data = labeled_sample["input"]
        # forward pass
        outputs, x_embed = model(data, return_embed=True)
        probs = torch.nn.functional.softmax(outputs, dim=-1)
        _, predictions = torch.max(probs, dim=-1)

        # compute loss
        loss = cross_entropy(
            pred_=probs,
            label=labeled_sample["output"],
            sample_weight=self.sample_weight,
            class_weight=self.class_weight,
            reduction=self.reduction,
        )

        labeled_sample["embed"] = x_embed

        if unlabeled_sample is not None:
            outputs, u_embed = model(unlabeled_sample["input"], return_embed=True)
            unlabeled_sample["embed"] = u_embed

        return loss


class InterpolationConsistencyLoss(nn.Module):
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
        """Computes an Interpolation Consistency Loss
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
        """
        super(InterpolationConsistencyLoss, self).__init__()

        self.unsup_criterion = unsup_criterion
        self.sup_criterion = sup_criterion
        self.decay_coef = decay_coef
        self.mean_teacher = mean_teacher
        if self.mean_teacher:
            print("IC Loss is using a mean teacher.")
        self.augment = augment
        self.teacher_eval = teacher_eval
        self.teacher_bn_running_stats = teacher_bn_running_stats

        # instantiate a callable MixUp operation
        self.mixup_op = SampleMixUp(**kwargs)

        self.teacher = None
        self.step = 0
        return

    def _update_teacher(
        self,
        model: nn.Module,
    ) -> None:
        """Update the teacher model based on settings"""
        if self.mean_teacher:
            if self.teacher is None:
                # instantiate the teacher with a copy
                # of the model
                self.teacher = copy.deepcopy(
                    model,
                )
            else:
                self._update_teacher_params(
                    model,
                )
        else:
            self.teacher = copy.deepcopy(
                model,
            )

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

    def _update_teacher_params(
        self,
        model: nn.Module,
    ) -> None:
        """Update parameters in the teacher model using an
        exponential averaging method.

        Notes
        -----
        Logic derived from the Mean Teacher implementation
        https://github.com/CuriousAI/mean-teacher/
        """
        # Per the mean-teacher paper, we use the global average
        # of parameter values until the exponential average is more effective
        # For a `decay_coef ~= 0.997`, this hand-off happens at ~step 333.
        alpha = min(1 - 1 / (self.step + 1), self.decay_coef)
        # Perform in-place operations on the teacher parameters to average
        # with the new model parameters
        # Here, we're computing a simple weighted average where alpha is
        # the weight on past parameters, and (1 - alpha) is the weight on
        # new parameters
        zipped_params = zip(self.teacher.parameters(), model.parameters())
        for teacher_param, model_param in zipped_params:
            (teacher_param.data.mul_(alpha).add_(1 - alpha, model_param.data))
        return

    def __call__(
        self,
        model: nn.Module,
        unlabeled_sample: dict,
        labeled_sample: dict,
    ) -> torch.FloatTensor:
        """Takes a model and set of unlabeled samples as input
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
        """
        ###############################
        # (0) Update the mean teacher
        ###############################

        self._update_teacher(
            model,
        )

        ###############################
        # (1) Compute Fake Labels
        ###############################

        with torch.no_grad():
            fake_y = F.softmax(
                self.teacher(unlabeled_sample["input"]),
                dim=1,
            )

        ###############################
        # (2) Perform MixUp and Forward
        ###############################

        unlabeled_sample["output"] = fake_y

        mixed_sample = self.mixup_op(unlabeled_sample)
        # move sample to model device if necessary
        mixed_sample["input"] = mixed_sample["input"].to(
            device=next(model.parameters()).device,
        )
        mixed_output = F.softmax(
            model(mixed_sample["input"]),
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
        labeled_sample["input"] = labeled_sample["input"].to(
            device=next(model.parameters()).device,
        )
        labeled_sample["input"].requires_grad = True

        sup_outputs = model(labeled_sample["input"])
        sup_loss = self.sup_criterion(
            sup_outputs,
            labeled_sample["output"],
        )

        self.step += 1
        return sup_loss, icl, sup_outputs


def sharpen_labels(
    q: torch.FloatTensor,
    T: float = 0.5,
) -> torch.FloatTensor:
    """Reduce the entropy of a categorical label using a
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

    """
    if T == 0.0:
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

    q = torch.pow(q, 1.0 / T)
    q /= torch.sum(
        q,
        dim=1,
    ).reshape(-1, 1)
    return q


class MixMatchLoss(InterpolationConsistencyLoss):
    """Compute the MixMatch Loss given a batch of labeled
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
    """

    def __init__(
        self,
        n_augmentations: int = 2,
        T: float = 0.5,
        augment_pseudolabels: bool = True,
        pseudolabel_min_confidence: float = 0.0,
        **kwargs,
    ) -> None:
        """Compute the MixMatch Loss given a batch of labeled
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
        """
        # inherit from IC loss, forcing the SampleMixUp to keep
        # the identity of the dominant observation in each mixed sample
        super(MixMatchLoss, self).__init__(
            **kwargs,
            keep_dominant_obs=True,
        )
        if not callable(self.augment):
            msg = "MixMatch requires a Callable for augment"
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
        """Generate labels by applying a set of augmentations
        to each unlabeled example and keeping the mean.

        Parameters
        ----------
        unlabeled_batch : dict
            "input" - [Batch, Features] minibatch of unlabeled samples.
        """
        # let the teacher model take guesses at the label for augmented
        # versions of the unlabeled observations
        raw_guesses = []
        for i in range(self.n_augmentations):
            to_augment = {
                "input": unlabeled_sample["input"].clone(),
                "output": torch.zeros(1),
            }
            if self.augment_pseudolabels:
                # augment the batch before pseudolabeling
                augmented_batch = self.augment(to_augment)
            else:
                augmented_batch = to_augment
            # convert model guess to probability distribution `q`
            # with softmax, prior to considering it a label
            guess = F.softmax(
                self.teacher(augmented_batch["input"]),
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
        labeled_sample: dict,
        unlabeled_sample: dict,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Parameters
        ----------
        model : nn.Module
            model with parameters accessible via the `.parameters()`
            method.
        labeled_sample : dict
            input - torch.FloatTensor
                [Batch, Features] minibatch of labeled examples.
            output - torch.LongTensor
                one-hot labels.
        unlabeled_sample : dict
            input - torch.FloatTensor
                [Batch, Features] minibatch of unlabeled samples.
            output - torch.LongTensor
                zeros.


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
        """

        ########################################
        # (0) Update the mean teacher
        ########################################

        self._update_teacher(
            model,
        )

        ########################################
        # (1) Generate labels for unlabeled data
        ########################################

        pseudolabels, pseudolabel_confidence = self._generate_labels(
            unlabeled_sample=unlabeled_sample,
        )
        # make sure pseudolabels match real label dtype
        # so that they can be concatenated
        pseudolabels = pseudolabels.to(dtype=labeled_sample["output"].dtype)

        ########################################
        # (2) Augment the labeled data
        ########################################

        labeled_sample = self.augment(
            labeled_sample,
        )

        ########################################
        # (3) Perform MixUp across both batches
        ########################################
        n_unlabeled_original = unlabeled_sample["input"].size(0)
        unlabeled_sample["output"] = pseudolabels

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
            )
            for k in ["input", "output"]
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
        n_unlabeled = conf_unlabeled_sample["input"].size(0)
        unlabeled_m_ = mixed_samples["input"][:n_unlabeled]
        unlabeled_y_ = mixed_samples["output"][:n_unlabeled]

        labeled_m_ = mixed_samples["input"][n_unlabeled:]
        labeled_y_ = mixed_samples["output"][n_unlabeled:]

        # append low confidence samples to unlabeled_m_ and unlabeled_y_
        # this ensures that batch norm is still able to update it's
        # statistics based on batches from the train AND target domain
        unlabeled_m_ = torch.cat(
            [
                unlabeled_m_,
                ucnf_unlabeled_sample["input"],
            ]
        )
        unlabeled_y_ = torch.cat(
            [
                unlabeled_y_,
                ucnf_unlabeled_sample["output"],
            ]
        )

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
        scale_vec = (
            torch.zeros_like(unsupervised_loss)
            .float()
            .to(device=unsupervised_loss.device)
        )
        scale_vec[:n_unlabeled] += 1.0
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


class MultiTaskMixMatchWrapper(nn.Module):
    def __init__(
        self,
        mixmatch_loss: MixMatchLoss,
        sup_weight: Union[float, Callable] = 1.0,
        unsup_weight: Union[float, Callable] = 1.0,
        use_sup_eval: bool = True,
    ) -> None:
        """Wrapper around the `MixMatchLoss` class for use with `MultiTaskTrainer`.
        The wrapper performs weighting of the supervised and unsupervised loss
        internally, then returns a single `torch.FloatTensor` to `MultiTaskTrainer`
        to maintain a consistent "one criterion, one loss" API.

        Parameters
        ----------
        mixmatch_loss : MixMatchLoss
            an instance of the `MixMatchLoss` class.
        sup_weight : float, Callable
            constant weight or callable weight schedule function for the
            supervised MixMatch loss.
        unsup_weight : float, Callable
            constant weight or callable weight schedule function for the
            unsupervised MixMatch loss.
        use_sup_eval : bool
            use only the supervised loss when in eval mode.

        Returns
        -------
        None.

        Notes
        -----
        Relies upon updating the `.epoch` attribute during the training
        loop to properly enforce weight scheduling.
        """
        super(MultiTaskMixMatchWrapper, self).__init__()
        self.mixmatch_loss = mixmatch_loss
        self.sup_weight = sup_weight
        self.unsup_weight = unsup_weight
        self.use_sup_eval = use_sup_eval
        # initialize the epoch attribute so `MultiTaskTrainer` can find it
        # `.epoch` will be updated in the training loop
        self.epoch = 0
        return

    def __call__(
        self,
        *,
        labeled_sample: dict,
        unlabeled_sample: dict,
        model: nn.Module,
        weight: float = None,
    ) -> torch.FloatTensor:
        """Compute MixMatch losses, weight them internally, then return
        the weighted sum.

        Parameters
        ----------
        labeled_sample : dict
            input - torch.FloatTensor
                [Batch, Features] minibatch of labeled examples.
            output - torch.LongTensor
                one-hot labels.
        unlabeled_sample : dict
            input - torch.FloatTensor
                [Batch, Features] minibatch of unlabeled samples.
            output - torch.LongTensor
                zeros.
        model : nn.Module
            model with parameters accessible via the `.parameters()`
            method.
        weight : float
            unused weight parameter for compatability with the `MultiTaskTrainer`
            API.

        Returns
        -------
        loss : torch.FloatTensor
            weighted sum of MixMatch supervised and unsupervised loss.
        """
        sup_loss, unsup_loss, labeled_z_ = self.mixmatch_loss(
            labeled_sample=labeled_sample,
            unlabeled_sample=unlabeled_sample,
            model=model,
        )
        # get weights for each loss by either calling the function or keeping
        # the constant value provided
        sup_weight = (
            self.sup_weight(self.epoch)
            if callable(self.sup_weight)
            else self.sup_weight
        )
        unsup_weight = (
            self.unsup_weight(self.epoch)
            if callable(self.unsup_weight)
            else self.unsup_weight
        )

        # don't use the unsupervised loss if we're in eval mode
        # `use_sup_eval` is set
        if self.use_sup_eval and not self.training:
            unsup_weight = 0.0

        loss = (sup_weight * sup_loss) + (unsup_weight * unsup_loss)
        return loss


"""Domain adaptation losses"""


class DANLoss(nn.Module):
    """Compute a domain adaptation network (DAN) loss."""

    def __init__(
        self,
        dan_criterion: Callable,
        model: CellTypeCLF,
        use_conf_pseudolabels: bool = False,
        scale_loss_pseudoconf: bool = False,
        n_domains: Union[int, tuple] = 2,
        **kwargs,
    ) -> None:
        """Compute a domain adaptation network loss.

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
        n_domains : int, Tuple[int]
            number of domains of origin to predict using the adversary.
            each int in a tuple is considered a unique domain label for 
            multi-adversarial training. unique domain labels must be supplied
            in the order as provided to `SingleCellDS`.

        Returns
        -------
        None.

        Notes
        -----
        **kwargs are passed to `scnym.model.DANN`

        See Also
        --------
        scnym.model.DANN
        scnym.trainer.MultiTaskTrainer
        """
        super(DANLoss, self).__init__()

        self.dan_criterion = dan_criterion
        self.n_domains = (n_domains,) if type(n_domains)==int else n_domains
        self.n_dans = 1 if type(n_domains)==int else len(n_domains)
        # build the DANN (build multiple if needed)
        dann = []
        for nd in self.n_domains:
            adv = DANN(
                model=model,
                n_domains=nd,
                **kwargs,
            )
            adv.domain_clf = adv.domain_clf.to(
                device=next(iter(model.parameters())).device,
            )
            dann.append(adv)
        self.dann = nn.ModuleList(dann)
        # instantiate with small tensor to simplify downstream size
        # checking logic
        self.x_embed = torch.zeros((1, 1))

        self.use_conf_pseudolabels = use_conf_pseudolabels
        self.scale_loss_pseudoconf = scale_loss_pseudoconf
        # note that weighting is performed on gradients internally;
        # accessed by `trainer.MultiTaskTrainer`
        self.no_weight = True
        return

    def __call__(
        self,
        labeled_sample: dict,
        unlabeled_sample: dict = None,
        weight: float = 1.0,
        pseudolabel_confidence: torch.Tensor = None,
        **kwargs,
    ) -> torch.FloatTensor:
        """Compute the domain adaptation loss on a labeled source
        and unlabeled target domain batch.

        Parameters
        ----------
        labeled_sample : dict
            input - torch.FloatTensor
                [BatchL, Features] minibatch of labeled examples.
            output - torch.LongTensor
                one-hot labels.
        unlabeled_sample : dict
            input - torch.FloatTensor
                [BatchU, Features] minibatch of unlabeled samples.
            output - torch.LongTensor
                zeros.
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
        **kwargs : dict
            kwargs are a no-op, included to allow for `model` kwarg per
            `MultiTaskTrainer` API.

        Returns
        -------
        dan_loss : torch.FloatTensor
            domain adversarial loss term.
        """
        # if no unlabeled data is provided, we create a dict of empty
        # tensors. these tensors lead to no-ops for all the `.cat` ops
        # below.
        if unlabeled_sample is None:
            t = torch.FloatTensor().to(device=labeled_sample["input"].device)
            unlabeled_sample = {k: t for k in ["input", "domain"]}
        # we also ignore the unlabeled data if it doesn't have a domain label
        # associated with it. null domain labels are `-1`, so this checks absence/null.
        if torch.sum(unlabeled_sample.get("domain", torch.tensor([-1,])) < 0 ) > 0:
            t = torch.FloatTensor().to(device=labeled_sample["input"].device)
            unlabeled_sample = {k: t for k in ["input", "domain"]}       

        ########################################
        # (1) Create domain labels
        ########################################

        # check if domain labels are provided, if not assume
        # train and target are separate domains
        # domain labels of -1 indicate `None` was passed as a domain label
        # to `SingleCellDS`
        if torch.sum(labeled_sample.get("domain", torch.Tensor([-1])) == -1) > 0:
            source_label = torch.zeros(labeled_sample["input"].size(0)).long()
            source_label = torch.nn.functional.one_hot(
                source_label,
                num_classes=2,
            )
            logger.debug("DAN source domain labels inferred.")
        else:
            # domain labels should already by one-hot
            # if multiple domains are present, one-hot labels are concatenated
            # along the dim=1 dimension, so we will need to split them using info
            # in `self.n_domains` later
            source_label = labeled_sample["domain"]
        source_label = source_label.to(device=labeled_sample["input"].device)

        if torch.sum(unlabeled_sample.get("domain", torch.Tensor([-1])) == -1) > 0:
            target_label = torch.ones(unlabeled_sample["input"].size(0)).long()
            target_label = torch.nn.functional.one_hot(
                target_label,
                num_classes=2,
            )
            logger.debug("DAN target domain labels inferred.")
        else:
            target_label = unlabeled_sample["domain"]
        target_label = target_label.to(device=unlabeled_sample["input"].device)

        lx = labeled_sample["input"]
        ux = unlabeled_sample["input"]

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
        self.n_total_unlabeled = unlabeled_sample["input"].size(0)
        p_conf_pseudolabels = self.n_conf_pseudolabels / max(self.n_total_unlabeled, 1)

        ########################################
        # (3) Embed points and Classify domains
        ########################################

        x = torch.cat([lx, ux], 0)
        dlabel = torch.cat([source_label, target_label], 0)

        # predict each domain variable with a separate adversary
        dan_loss = torch.zeros(1,).to(device=lx.device)
        for i, adv in enumerate(self.dann):
            adv.set_rev_grad_weight(weight=weight)
            domain_pred, x_embed = adv(x)

            # store embeddings and labels only fir the first adversary
            if (x_embed.size(0) >= self.x_embed.size(0)) and (i==0):
                self.x_embed = copy.copy(x_embed.detach().cpu())
                self.dlabel = copy.copy(dlabel.detach().cpu())

            ########################################
            # (4) Compute DAN loss
            ########################################

            # we need to extract the labels for *this* domain from the concatenated
            # one-hot label matrices
            idx_start = sum(self.n_domains[:i])
            idx_end = idx_start + self.n_domains[i]
            curr_dlabel = dlabel[:, idx_start:idx_end]

            adv_loss = self.dan_criterion(
                domain_pred,
                curr_dlabel,
            )
            dan_loss += adv_loss

            ########################################
            # (5) Compute DAN accuracy for logs
            ########################################

            # only record accuracy for first domain label
            if i != 0:
                continue
            _, dan_pred = torch.max(domain_pred, dim=1)
            _, dlabel_int = torch.max(dlabel, dim=1)
            self.dan_acc = (
                torch.sum(
                    dan_pred == dlabel_int,
                )
                / float(dan_pred.size(0))
            )

        if self.scale_loss_pseudoconf:
            dan_loss *= p_conf_pseudolabels

        return dan_loss


"""Reconstruction losses"""


def poisson_loss(
    input_: torch.FloatTensor,
    target: torch.FloatTensor,
    dispersion: torch.FloatTensor = None,
) -> torch.FloatTensor:
    """Compute a Poisson loss for count data.

    Parameters
    ----------
    input_ : torch.FloatTensor
        [Batch, Feature] Poisson rate parameters.
    target : torch.FloatTensor
        [Batch, Features] count based target.
    dispersion : torch.FloatTensor
        Ignored for Poisson loss.

    Returns
    -------
    nll : torch.FloatTensor
        Poisson negative log-likelihood.
    """
    # input_ are Poisson rates, compute likelihood of target data
    # and sum likelihood across genes
    nll = -1 * torch.sum(
        torch.distributions.Poisson(input_).log_prob(target),
        dim=-1,
    )
    return nll


def negative_binomial_loss(
    input_: torch.FloatTensor,
    target: torch.FloatTensor,
    dispersion: torch.FloatTensor,
    eps: float = 1e-8,
) -> torch.FloatTensor:
    """Compute a Negative Binomial loss for count data.

    Parameters
    ----------
    input_ : torch.FloatTensor
        [Batch, Feature] Negative Binomial mean parameters.
    target : torch.FloatTensor
        [Batch, Features] count based target.
    dispersion : torch.FloatTensor
        [Features,] Negative Binomial dispersion parameters.
    eps : float
        small constant to avoid numerical issues.

    Returns
    -------
    nll : torch.FloatTensor
        Negative Binomial negative log-likelihood.

    References
    ----------
    Credit to `scvi-tools`:
    https://github.com/YosefLab/scvi-tools/blob/42315756ba879b9421630696ea7afcd74e012a07/scvi/distributions/_negative_binomial.py#L67
    """
    res = -1 * (NegativeBinomial(mu=input_, theta=dispersion).log_prob(target).sum(-1))
    return res


def mse_loss(
    input_: torch.FloatTensor,
    target: torch.FloatTensor,
    dispersion: torch.FloatTensor,
) -> torch.FloatTensor:
    """MSELoss wrapped for scNym compatibility"""
    return torch.nn.functional.mse_loss(input_, target)


class ReconstructionLoss(nn.Module):
    """Computes a reconstruction of the input data from the
    embedding"""

    def __init__(
        self,
        *,
        model: nn.Module,
        rec_criterion: Callable,
        reduction: str = "mean",
        norm_before_loss: float = None,
        **kwargs,
    ) -> None:
        """Computes a reconstruction loss of the input data
        from the embedding.

        Parameters
        ----------
        model : nn.Module
            cell type classification model to use for cellular
            embedding.
        rec_criterion : Callable
            reconstruction loss that takes two arguments `(input_, target)`.
        reduction : str
            {"none", "mean", "sum"} reduction operation for [Batch,] loss values.
        norm_before_loss : float
            normalize profiles to the following depth before computing loss.
            this helps balance loss contribution from cells with dramatically
            different depths (e.g. Drop-seq and Smart-seq2).
            if `None`, does not normalize before loss.
        **kwargs : dict
            passed to recontruction model `.model.AE`.

        Returns
        -------
        None.
        """
        super(ReconstructionLoss, self).__init__()

        self.rec_criterion = rec_criterion
        self.model = model
        self.reduction = reduction
        if reduction not in (None, "none", "sum", "mean"):
            msg = f"reduction argument {self.reduction} is invalid."
            raise ValueError(msg)
        self.norm_before_loss = norm_before_loss

        # build the reconstruction autoencoder
        self.rec_model = AE(
            model=model,
            **kwargs,
        )
        # move rec_model to the appropriate computing device
        self.rec_model = self.rec_model.to(
            device=list(self.model.parameters())[1].device,
        )

        return

    def __call__(
        self,
        labeled_sample: dict,
        unlabeled_sample: dict = None,
        weight: float = 1.0,
        **kwargs,
    ) -> torch.FloatTensor:
        """Compute the domain adaptation loss on a labeled source
        and unlabeled target domain batch.

        Parameters
        ----------
        labeled_sample : dict
            input - torch.FloatTensor
                [BatchL, Features] minibatch of labeled examples.
            output - torch.LongTensor
                [BatchL,] one-hot labels.
            embed - torch.FloatTensor, optional
                [BatchL, n_hidden] minibatch embedding.
        unlabeled_sample : dict, optional.
            input - torch.FloatTensor
                [BatchU, Features] minibatch of unlabeled samples.
            output - torch.LongTensor
                [BatchU,] zeros.
            embed - torch.FloatTensor, optional
                [BatchU, n_hidden] minibatch embedding.
        weight : float
            reconstruction loss weight. Not used, present for compatability with the
            `MultiTaskTrainer` API.
        kwargs : dict
            currently not used, allows for compatibility with `Trainer` subclasses
            that pass `model` to call by default (e.g. as used for the old `MixMatchLoss`).

        Returns
        -------
        reconstruction_loss : torch.FloatTensor
            reconstruction loss, reduced across the batch.
        """
        if unlabeled_sample is None:
            # if no unlabeled data is passed, we create empty FloatTensors
            # to concat onto the labeled tensors below.
            # cat of an empty tensor is a no-op.
            t = torch.FloatTensor().to(device=labeled_sample["input"].device)
            unlabeled_sample = {
                "input": t,
                "embed": t,
                "domain": t,
            }

        # join data into a single batch
        x = torch.cat(
            [
                labeled_sample["input"],
                unlabeled_sample["input"],
            ],
            dim=0,
        )

        # use pre-computed embeddings if they're available from e.g.
        # a previous loss function.
        if "embed" in labeled_sample.keys() and "embed" in unlabeled_sample.keys():
            x_embed = torch.cat(
                [
                    labeled_sample["embed"],
                    unlabeled_sample["embed"],
                ],
                dim=0,
            )
        else:
            x_embed = None

        # pass domain arguments to the reconstruction model if specified
        # domains are already [Batch, Domains] one-hot encoded.
        if self.rec_model.n_domains > 0:
            x_domain = torch.cat(
                [
                    labeled_sample["domain"],
                    unlabeled_sample["domain"],
                ],
                dim=0,
            ).to(device=x.device)
        else:
            x_domain = None

        # perform embedding and reconstruction
        # if `x_embed is None`, computes the embedding using the
        # trunk of the classification model
        x_rec, x_scaled, dispersion, x_embed = self.rec_model(
            x,
            x_embed=x_embed,
            x_domain=x_domain,
        )

        if self.norm_before_loss is not None:
            # normalize to a common depth (CP-TenThousand) before computing loss
            x_scaled2use = x_scaled / x_scaled.sum(1).view(-1, 1) * 1e6
            x2use = x / x.sum(1).view(-1, 1) * self.norm_before_loss4
        else:
            x_scaled2use = x_scaled
            x2use = x

        # score reconstruction
        reconstruction_loss = self.rec_criterion(
            input_=x_scaled2use,
            target=x2use,
            dispersion=dispersion,
        )
        if self.reduction == "mean":
            reconstruction_loss = torch.mean(reconstruction_loss)
        elif (self.reduction == "none") or (self.reduction is None):
            reconstruction_loss = reconstruction_loss
        elif self.reduction == "sum":
            reconstruction_loss = torch.sum(reconstruction_loss)
        else:
            msg = f"reduction argument {self.reduction} is invalid."
            raise ValueError(msg)

        return reconstruction_loss


class LatentL2(nn.Module):
    def __init__(
        self,
    ) -> None:
        """Compute an l2-norm penalty on the latent embedding.
        This serves as a sufficient regularization in deterministic
        regularized autoencoders (RAE), akin to the KL term in VAEs.

        References
        ----------
        https://openreview.net/pdf?id=S1g7tpEYDS
        """
        super(LatentL2, self).__init__()

        return

    def __call__(
        self,
        labeled_sample: dict,
        unlabeled_sample: dict,
        model: nn.Module = None,
        weight: float = None,
    ) -> torch.FloatTensor:
        """Compute an l2 penalty on the latent space of a model"""
        # is the embedding pre-computed for both samples?
        embed_computed = "embed" in labeled_sample.keys()
        if unlabeled_sample is not None:
            embed_computed = embed_computed and ("embed" in unlabeled_sample.keys())
        keys = ["input"]
        if embed_computed:
            keys += ["embed"]

        if unlabeled_sample is not None:
            # join tensors across samples
            sample = {
                k: torch.cat([labeled_sample[k], unlabeled_sample[k]], 0) for k in keys
            }
        else:
            sample = labeled_sample

        if embed_computed:
            x_embed = sample["embed"]
        else:
            data = sample["input"]
            logits, x_embed = model(data, return_embed=True)

        l2 = 0.5 * torch.norm(x_embed, p=2)
        return l2


# TODO: Consider adding in one of the TC-VAE mutual information
# penalties for latent vars to substitute for the covariance penalty
# inherent in the mean field VAE KL term


class UnsupervisedLosses(object):
    """Compute multiple unsupervised loss functions"""

    def __init__(
        self,
        losses: list,
        weights: list = None,
    ) -> None:
        """Compute multiple unsupervised loss functions.

        Parameters
        ----------
        losses : List[Callable]
            each element in list is a Callable that takes arguments
            `labeled_sample, unlabeled_sample` and returns a `torch.FloatTensor`
            differentiable loss suitable for backprop.
            methods can also take or ignore a `weight` argument.
        weights : List[Callable]
            matching weight functions for each loss that take an input int epoch
            and return a float loss weight.

        Returns
        -------
        None.

        Notes
        -----
        Computes each loss in serial.

        """
        self.losses = losses
        # if no weights are provided, use a uniform schedule with
        # weight `1.` for each loss function.
        self.weights = weights if weights is not None else [lambda x: 1.0] * len(losses)
        return

    def __call__(
        self,
        labeled_sample: dict,
        unlabeled_sample: dict,
    ) -> torch.FloatTensor:
        loss = torch.zeros(
            1,
        )
        for i, fxn in enumerate(self.losses):
            fxn_loss = fxn(
                labeled_sample=labeled_sample,
                unlabeled_sample=unlabeled_sample,
                weight=self.weights[i],
            )
            loss += fxn_loss
        return loss


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


"""Structured latent variable learning"""


def set_prior_matrix_from_gene_sets(
    weight_class,
) -> None:
    """Generate a prior matrix from a set of gene programs
    and gene names for the input variables.
    """
    weight_class.gene_set_names = sorted(list(weight_class.gene_sets.keys()))

    # [n_programs, n_genes]
    P = torch.zeros(
        (
            weight_class.n_hidden,
            weight_class.n_genes,
        )
    ).bool()

    gene_names = weight_class.gene_names
    for i, k in enumerate(weight_class.gene_set_names):
        genes = weight_class.gene_sets[k]
        bidx = torch.tensor(
            [x in genes for x in gene_names],
            dtype=torch.bool,
        )
        P[i, :] = bidx

    return P


class StructuredSparsity(object):
    def __init__(
        self,
        n_genes: int,
        n_hidden: int,
        gene_sets: dict = None,
        gene_names: Iterable = None,
        prior_matrix: Union[np.ndarray, torch.Tensor] = None,
        n_dense_latent: int = 0,
        group_lasso: float = 0.0,
        p_norm: int = 1,
        nonnegative: bool = False,
    ) -> None:
        """Add structured sparsity penalties to regularize
        weights of an encoding layer.

        Parameters
        ----------
        n_genes : int
            number of genes in the input layer.
        n_hidden : int
            number of hidden units in the input layer.
        gene_sets : dict, optional.
            keys are program names, values are lists of gene names.
            must have fewer keys than `n_hidden`.
        gene_names : Iterable, optional.
            names for genes in `n_genes`. required for use of `gene_sets`.
        prior_matrix : np.ndarray, torch.FloatTensor
            [n_hidden, n_genes] binary matrix of prior constraints.
            if provided with `gene_sets`, this matrix is used instead.
        n_dense_latent : int
            number of latent variables with no l1 loss applied.
            applies to the final `n_dense_latent` variables.
        group_lasso : float, optional.
            weight for a group LASSO penalty on the second hidden
            layer. [Default = 0].
        p_norm : int
            p-norm to use for the prior penalty. [Default = 1] for lasso.
        nonnegative : bool
            apply an L1 penalty to *all* negative values. this implicitly enforces
            a roughly non-negative projection matrix.

        Returns
        -------
        None.
        """
        self.n_genes = n_genes
        self.n_hidden = n_hidden
        self.gene_sets = gene_sets
        self.gene_names = gene_names.tolist() if type(gene_names)!= list else gene_names
        self.prior_matrix = None
        self.n_dense_latent = n_dense_latent
        self.group_lasso = group_lasso
        self.p_norm = p_norm
        self.nonnegative = nonnegative

        if prior_matrix is None and gene_sets is None:
            msg = "Must provide either a prior_matrix or gene_sets to use."
            raise ValueError(msg)

        if gene_sets is not None and gene_names is None:
            msg = "Must provide `gene_names` to use `gene_sets`."
            raise ValueError(msg)

        if gene_sets is not None and gene_names is not None:

            if len(gene_sets.keys()) > self.n_hidden:
                # check that we didn't provide too many gene sets
                # given the size of our encoder
                msg = f"{len(gene_sets.keys())} gene sets provided,\n"
                msg += f"but there are only {n_hidden} hidden units.\n"
                msg += "Must specify fewer programs than hidden units."
                raise ValueError(msg)

            # set `self.prior_matrix` based on the gene sets
            # also sets `self.gene_set_names`
            self.prior_matrix = set_prior_matrix_from_gene_sets(weight_class=self)

        if prior_matrix is not None:
            # if the prior_matrix was provided, always prefer it.
            self.prior_matrix = prior_matrix

        assert self.prior_matrix is not None
        return

    def _set_prior_matrix_from_gene_sets(
        self,
    ) -> None:
        """Generate a prior matrix from a set of gene programs
        and gene names for the input variables.
        """
        self.gene_set_names = sorted(list(self.gene_sets.keys()))

        # [n_programs, n_genes]
        P = torch.zeros(
            (
                self.n_hidden,
                self.n_genes,
            )
        ).bool()

        # cast to set for list comprehension speed
        gene_names = set(self.gene_names)
        for i, k in enumerate(self.gene_set_names):
            genes = self.gene_sets[k]

            bidx = torch.tensor(
                [x in genes for x in gene_names],
                dtype=torch.bool,
            )
            P[i, :] = bidx

        self.prior_matrix = P
        return

    @torch.no_grad()
    def init_model_params(self, model: nn.Module, scale: float=0.01):
        """Scale parameters of the input layer to initialize close to a sparse regime"""
        W = dict(model.named_parameters())["embed.0.weight"]
        P = torch.logical_not(self.prior_matrix.to(device=W.data.device))
        prior_flat = P.view(-1)
        W_flat = W.view(-1)
        W_flat[prior_flat] *= scale
        return

    def __call__(
        self,
        model: nn.Module,
        **kwargs,
    ) -> torch.FloatTensor:
        """Compute the l1 sparsity loss."""
        # get first layer weights
        W = dict(model.named_parameters())["embed.0.weight"]
        logger.debug(f"Weights {W}, sum: {W.sum()}")
        # generate a "penalty" matrix `P` that we'll modify
        # before computing the l1
        # this elem-mult zeros out the loss on any annotated
        # genes in each gene program
        P = W * torch.logical_not(self.prior_matrix).float().to(device=W.device)
        logger.debug(f"Penalty {P}, sum {P.sum()}")
        # omit the dense latent factors (if any) from the l1
        # computation
        n_latent = P.size(0) - self.n_dense_latent
        prior_norm = torch.norm(P[:n_latent], p=self.p_norm)
        logger.debug(f"l1 {prior_norm}")

        if self.nonnegative:
            # place an optional non-negativity penalty on genes within the gene set
            # nonneg_inset = W * self.prior_matrix.float().to(device=W.device)
            # nonneg_norm = torch.norm(nonneg_inset[nonneg_inset < 0], p=self.p_norm)
            W_flat = W.view(-1)
            bidx = (W_flat < 0.0).bool()
            W_flat[bidx] = 0.0
        else:
            nonneg_norm = 0.0

        r = prior_norm + nonneg_norm
        return r

    def train(self, *args, **kwargs) -> None:
        """Dummy function to match `nn.Module` methods"""
        return

    def eval(self, *args, **kwargs) -> None:
        """Dummy function to match `nn.Module` methods"""
        return


class WithinGeneSetNorm(object):

    def __init__(
        self,
        gene_sets: dict = None,
        gene_names: Iterable = None,
    ) -> None:
        """Penalize a norm for latent variable weights that are *within* a prior
        gene set specification"""
        self.n_genes = len(gene_names)
        self.n_hidden = len(gene_sets)
        self.gene_sets = gene_sets
        self.gene_names = gene_names

        self._set_prior_matrix_from_gene_sets()
        return

    def _set_prior_matrix_from_gene_sets(
        self,
    ) -> None:
        """Generate a prior matrix from a set of gene programs
        and gene names for the input variables.
        """
        self.gene_set_names = sorted(list(self.gene_sets.keys()))

        # [n_programs, n_genes]
        P = torch.zeros(
            (
                self.n_hidden,
                self.n_genes,
            )
        ).bool()

        # cast to set for list comprehension speed
        gene_names = set(self.gene_names)
        for i, k in enumerate(self.gene_set_names):
            genes = self.gene_sets[k]
            bidx = torch.tensor(
                [x in genes for x in gene_names],
                dtype=torch.bool,
            )
            P[i, :] = bidx

        self.prior_matrix = P
        return

    def __call__(self, model, *args, **kwargs) -> torch.FloatTensor:
        """Compute an L2 penalty only on the latent variable embedding weights"""
        W = dict(model.named_parameters())["embed.0.weight"]
        # zero out weights outside the gene set itself
        P = W * self.prior_matrix.float().to(device=W.device)
        norm = torch.norm(P, p=2)
        return norm

    def train(self, *args, **kwargs) -> None:
        """Dummy function to match `nn.Module` methods"""
        return

    def eval(self, *args, **kwargs) -> None:
        """Dummy function to match `nn.Module` methods"""
        return        


class WeightMask(object):


    def __init__(
        self,
        model: nn.Module,
        gene_sets: dict,
        gene_names: Iterable,
        prior_matrix: np.ndarray=None,
        mask_value: float=0.0,
        nonnegative: bool=True,
    ) -> None:
        """Apply a simple mask to the weights of a model.

        Parameters
        ----------
        n_genes : int
            number of genes in the input layer.
        n_hidden : int
            number of hidden units in the input layer.
        gene_sets : dict, optional.
            keys are program names, values are lists of gene names.
            must have fewer keys than `n_hidden`.
        gene_names : Iterable, optional.
            names for genes in `n_genes`. required for use of `gene_sets`.
        prior_matrix : np.ndarray, torch.FloatTensor
            [n_hidden, n_genes] binary matrix of prior constraints.
            if provided with `gene_sets`, this matrix is used instead.
        nonnegative : bool
            force weight values to be non-negative by ReLU-ing the weight matrix.

        Returns
        -------
        None.
        """
        self.model = model
        self.n_genes = self.model.n_genes
        self.n_hidden = self.model.n_hidden_init
        self.gene_sets = gene_sets
        self.gene_names = gene_names
        self.prior_matrix = prior_matrix
        self.mask_value = mask_value
        self.nonnegative = nonnegative

        if prior_matrix is None and gene_sets is None:
            msg = "Must provide either a prior_matrix or gene_sets to use."
            raise ValueError(msg)

        if gene_sets is not None and gene_names is None:
            msg = "Must provide `gene_names` to use `gene_sets`."
            raise ValueError(msg)

        assert len(gene_names) == self.model.n_genes

        if gene_sets is not None and gene_names is not None:

            if len(gene_sets.keys()) > self.n_hidden:
                # check that we didn't provide too many gene sets
                # given the size of our encoder
                msg = f"{len(gene_sets.keys())} gene sets provided,\n"
                msg += f"but there are only {self.n_hidden} hidden units.\n"
                msg += "Must specify fewer programs than hidden units."
                raise ValueError(msg)

            # set `self.prior_matrix` based on the gene sets
            # also sets `self.gene_set_names`
            self.prior_matrix = set_prior_matrix_from_gene_sets(weight_class=self)

        if prior_matrix is not None:
            # if the prior_matrix was provided, always prefer it.
            self.prior_matrix = prior_matrix

        assert self.prior_matrix is not None
        ident_op = lambda x: x
        # inverse relu for grad descent, we step in the opposite direction of grads
        inv_lu = lambda x: torch.clamp_max(x, 0.) 
        self.grad_op = (
            inv_lu if self.nonnegative else ident_op
        )
        # get the weights matrix 
        W = dict(model.named_parameters())["embed.0.weight"]
        # set the weights matrix values to zero outside the prior matrix, 
        # pos values elsewhere
        P = torch.logical_not(self.prior_matrix)
        with torch.no_grad():
            # we can only manually change values when grad is off
            # https://discuss.pytorch.org/t/how-to-manually-set-the-weights-in-a-two-layer-linear-model/45902/2
            W[P] = 0.0
            W[W < 0.] *= -1
            # register the backward hook
        W.register_hook(self.mask_grad)
        device = W.device

        self.prior_matrix = self.prior_matrix.to(device=device)

        return

    def mask_grad(
        self,
        grad,
    ) -> None:
        """Mask gradients to follow the sparsity matrix."""
        P = self.prior_matrix.to(device=grad.device)
        new_grad = grad * P
        return new_grad

    @torch.no_grad()
    def __call__(
        self,
        model: nn.Module,
        **kwargs,
    ) -> float:
        """Adjust weights to fit gene set priors as needed.
        
        Returns
        -------
        zero : float
            placeholder for a loss term so we can use this in `MultiTaskTrainer`.
        """
        EPS=1e-9
        W = dict(model.named_parameters())["embed.0.weight"]
        if self.nonnegative:
            # get zero indices within the prior matrix, fill with rand vals near eps
            P = self.prior_matrix.to(device=W.data.device)
            prior_flat = P.view(-1)
            W_flat = W.view(-1)
            W_neg_bidx = (W_flat < 0.0).bool()
            bidx2flip = prior_flat & W_neg_bidx
            W_flat[bidx2flip] = torch.rand(
                size=(bidx2flip.sum().item(),)
            ).to(device=W.data.device)*EPS
        return torch.zeros(size=(1,)).to(W.data.device)
    
    def train(self, *args, **kwargs) -> None:
        """Dummy function to match `nn.Module` methods"""
        return

    def eval(self, *args, **kwargs) -> None:
        """Dummy function to match `nn.Module` methods"""
        return


class LatentGeneCorrGuide(nn.Module):

    def __init__(
        self,
        gene_names: Iterable,
        latent_var_genes: Iterable,
        criterion: str="pearson",
        noise_p: float=0.3,
        lv_dropout: bool=True,
        mean_corr_weight: float=0.0,
        **kwargs,
    ) -> None:
        """Encourage a correlation between a set of latent variables and a set of
        specific input features.
        
        Parameters
        ----------
        gene_names : Iterable
            list of gene names for input features.
        latent_var_genes : Iterable
            [n_hidden_init,] length iterable of gene names to match to each latent
            var. if element is `None`, does not encourage correlation for that latent
            variable.
        noise_p : float
            [0, 1] fraction of input genes to mask before extracting the latent 
            embedding.
        lv_dropout : bool
            perform dropout on latent variables. it can be useful to deactivate lv
            dropout during lv-guide pre-training.
        mean_corr_weight : float
            peanlize the correlation of activation means and mRNA means, weighted by
            `mean_corr_weight`. if `0.0` does not penalize.
            this may be useful to prevent LVs without expressed genes as a guide from
            taking on arbitrarily high values.
        
        Returns
        -------
        None.

        Notes
        -----
        For now, treat each latent var as a binary classifier for the positive cells
        and use BCE loss to optimize.
        """
        super(LatentGeneCorrGuide, self).__init__()
        self.gene_names = (
            gene_names.tolist() 
            if type(gene_names)==np.ndarray else gene_names
        )
        self.noise_p = noise_p
        self.mean_corr_weight = mean_corr_weight
        # NOTE: Assumes that latent variable to gene set assignments are lexographically
        # sorted, per `set_prior_matrix_from_gene_sets`
        self.latent_var_genes = sorted(
            latent_var_genes.tolist()
            if type(latent_var_genes)==np.ndarray else latent_var_genes
        )
        self.bce = nn.BCELoss()
        if criterion == "bce":
            self.criterion = self.get_bce_loss
        elif criterion == "pearson":
            self.criterion = self.get_pearson_loss
        else:
            raise ValueError(f"criterion must be in [bce, pearson], not {criterion}.")
        self.lv_dropout = lv_dropout

        # find the feature indices for each latent variable
        self.mask_idx = -1
        self.latent_gene_idx = np.array(
            [
                self.gene_names.index(x) 
                if x in self.gene_names else self.mask_idx for x in self.latent_var_genes
            ],
            dtype=np.int32,
        )
        n_missing = np.sum(self.latent_gene_idx==self.mask_idx)
        if n_missing > 0:
            logger.warn(f"{n_missing} latent variables have no matching mRNA. "
            "Masking from loss computations.")
        self.latents_idx2keep = np.where(self.latent_gene_idx!=self.mask_idx)[0]
        self.genes_idx2keep = self.latent_gene_idx[self.latents_idx2keep]
        return

    def get_bce_loss(self, z, target) -> torch.FloatTensor:
        """Compute the mean BCE loss for each feature:LV pair"""
        # standardize z and target [0, 1]
        z_s = z - z.min(0)[0].view(1, -1)
        z_s = z_s / z_s.max(0)[0].view(1, -1)
        # target_s = target - target.min(0)[0].view(1, -1)
        # target_s = target_s / target_s.max(0)[0].view(1, -1)
        target_s = (target > 0.).float()
        z_s[torch.isnan(z_s)] = 0.
        target_s[torch.isnan(target_s)] = 0.
        loss = self.bce(z_s, target_s)
        return loss

    def get_pearson_loss(self, z, target) -> torch.FloatTensor:
        """Compute the pearson loss for each feature:LV pair and return the mean.

        Notes
        -----
        .. math::
            r = E[(x - \bar x) (y - \bar y)] / (\sigma_x \sigma_y)

        References
        ----------
        We use this trick to get the diagonal of a large matmul in the numerator
        https://bit.ly/3euhsJN
        """
        z_s = z - z.mean(0)
        target_s = target - target.mean(0)
        # [lv, batch] @ [batch, lv] numerator E[(x-\barx)(y-\bary)] -> [lv, lv]
        # torch.diag(m1 @ m2) == torch.sum(m1 * m2.T).sum(dim=1)
        num = (target_s.T * z_s.T).sum(dim=1) / z_s.size(0) # [lv,]
        denom = z_s.std(0).view(-1) * target_s.std(0).view(-1)
        denom[torch.isnan(denom)] = torch.min(denom[~torch.isnan(denom)])
        denom = torch.clamp_min(denom, torch.min(denom[denom>0]))
        return -torch.mean(num / denom)

    def __call__(
        self,
        model: nn.Module,
        labeled_sample: dict,
        unlabeled_sample: dict=None,
        **kwargs,
    ) -> torch.FloatTensor:
        """
        Parameters
        ----------
        model : nn.Module
            model for training.
        labeled_sample, unlabeled_sample : dict
            keys {input, output, domain, embed} with torch.Tensor values.
        
        Returns
        -------
        loss : torch.FloatTensor
        """
        # join labeled and unlabeled samples into a single tensor
        x0 = labeled_sample["input"]
        emptyT = torch.FloatTensor([]).to(device=x0.device) # no-op in `.cat`
        x1 = unlabeled_sample["input"] if unlabeled_sample is not None else emptyT
        # get LV activations for all datapoints
        if labeled_sample.get("lv", False):
            z0 = labeled_sample["lv"]
            z1 = unlabeled_sample["lv"] if unlabeled_sample is not None else emptyT
        else:
            # `list` of modules to form the lv embedder
            # [Lin, BN, DO (optional), ReLU]
            mods = model.input_stack
            if not self.lv_dropout and len(mods) > 3:
                # if lv_dropout is False AND dropout in the mods, cut it out
                mods = [x for i, x in enumerate(mods) if i!=3]
            if len(mods)!=3 and len(mods)!=4:
                # this is unexpected, throw an error
                msg = f"num modules in lv embedder is {len(mods)}, should be 3 or 4"
                raise ValueError(msg)

            # make embedder
            lv_embedder = nn.Sequential(
                nn.Dropout(p=self.noise_p),
                *mods
            )
            # embed to latent variables
            z0 = lv_embedder(x0)
            z1 = lv_embedder(x1) if unlabeled_sample is not None else emptyT

        x = torch.cat([x0, x1], dim=0)
        z = torch.cat([z0, z1], dim=0)
        # get the matrix of target expression for each latent var [Batch, LVs]
        target = x[:, self.genes_idx2keep]
        # remove any masked latent vars that don't have a corresponding mRNA
        z = z[:, self.latents_idx2keep]
        # compute the loss for z and targets
        loss = self.criterion(z, target)

        # compute the corr loss across means of latent and target values
        if self.mean_corr_weight > 0:
            mean_corr = self.criterion(
                torch.mean(z, dim=0).reshape(-1, 1),
                torch.mean(target, dim=0).reshape(-1, 1),
            )
            loss += (self.mean_corr_weight * mean_corr)

        return loss


"""Attribution priors"""


class AttrPrior(nn.Module):
    def __init__(
        self,
        reference_dataset: torch.utils.data.Dataset,
        batch_size: int,
        attr_prior: str="gini",
        grad_activation: str="first_layer",
    ) -> None:
        """Implement an attribution prior by penalizing expected gradient estimation
        of Shapley values during training.
        
        Parameters
        ----------
        reference_dataset : torch.utils.data.Dataset
            dataset to use for sampling expected gradient references.
        batch_size : int
            batch size to use for expected gradient reference sampling. 
            must match training batch size.
        attr_prior : str
            attribution prior type, one of {'gini', 'gini_classwise', }
        grad_activation : str
            activations to use for gradient computation.
            one of {"input", "first_layer"}.

        Returns
        -------
        None.
        """
        super(AttrPrior, self).__init__()
        self.ref_ds = reference_dataset
        self.batch_size = batch_size

        self.attr_prior = attr_prior
        self.grad_activation = grad_activation

        if attr_prior == "gini":

            def attr_criterion(model, labeled_sample, unlabeled_sample, weight):
                input_ = labeled_sample["input"]
                target = labeled_sample["output_int"]
                exp_grad = self._get_exp_grad(model, input_, target)
                # l_ap = attrprior.gini_eg(exp_grad)
                l_ap = attrprior.gini_eg(
                    exp_grad,
                )
                return l_ap

        elif attr_prior == "gini_classwise":
            
            def attr_criterion(model, labeled_sample, unlabeled_sample, weight):
                input_ = labeled_sample["input"]
                target = labeled_sample["output_int"]
                exp_grad = self._get_exp_grad(model, input_, target)
                # l_ap = attrprior.gini_eg(exp_grad)
                l_ap = attrprior.gini_classwise_eg(exp_grad, target)
                return l_ap

        elif attr_prior == "gini_cellwise":

            def attr_criterion(model, labeled_sample, unlabeled_sample, weight):
                input_ = labeled_sample["input"]
                target = labeled_sample["output_int"]
                exp_grad = self._get_exp_grad(model, input_, target)
                l_ap = attrprior.gini_cellwise_eg(exp_grad,)
                return l_ap

        else:
            msg = f"{attr_prior} is not a valid attribution prior type."
            raise ValueError(msg)

        self.attr_criterion = attr_criterion

        self.APExp = attrprior.AttributionPriorExplainer(
            self.ref_ds,
            batch_size=self.batch_size,
            k=1,
            input_batch_index="input",
        )

        self._get_exp_grad = {
            "first_layer": self._first_layer_exp_grad,
            "input": self._input_exp_grad,
        }[self.grad_activation]
        return

    def _input_exp_grad(
        self,
        model,
        input_: torch.FloatTensor,
        target: torch.LongTensor,
    ) -> torch.FloatTensor:
        """Get expected gradients from the input layer"""
        exp_grad = self.APExp.shap_values(
            model,
            input_,
            sparse_labels=target,
        )
        return exp_grad

    def _first_layer_exp_grad(
        self,
        model,
        input_: torch.FloatTensor,
        target: torch.LongTensor,
    ) -> torch.FloatTensor:
        """Get expected gradients for the first layer activations"""
        embedder_model = copy.deepcopy(model.get_initial_embedder())
        embedder_model.train(False)

        score_model = nn.Sequential(
            *model.mid_stack,
            *model.hidden_layers,
            model.classif,
        )
        # display the score and embedder models for debugging
        m_desc = list(embedder_model.modules())[0]
        logger.debug(f"embedder_model extracted\n{m_desc}")
        m_desc = list(score_model.modules())[0]
        logger.debug(f"score_model extracted\n{m_desc}")

        # debugger model visualization
        emdl = list(embedder_model.modules())[0]
        smdl = list(score_model.modules())[0]
        logger.debug(f"embedder_model: {emdl}")
        logger.debug(f"score_model: {smdl}")

        def batch_transformation(x):
            y = embedder_model(x)
            return y.detach()

        # embed the input
        embedded_input_ = batch_transformation(
            input_,
        )
        logger.debug(f"input_ size: {input_.size()}")
        logger.debug(f"embedded_input_ size: {embedded_input_.size()}")

        exp_grad = self.APExp.shap_values(
            model=score_model,
            input_tensor=embedded_input_,
            sparse_labels=target,
            batch_transformation=batch_transformation,
        )
        return exp_grad

    def __call__(
        self, 
        model: nn.Module, 
        labeled_sample: dict, 
        unlabeled_sample: dict=None, 
        weight: float=1.0, 
        **kwargs
    ) -> torch.FloatTensor:
        """Compute attribution prior on a batch of data"""
        if labeled_sample["output"].dim() > 1:
            # cast targets to integers for attribution prior calculation
            _, target_int = torch.max(labeled_sample["output"], dim=-1)
            labeled_sample["output_int"] = target_int

        c = self.attr_criterion(
            model=model,
            labeled_sample=labeled_sample,
            unlabeled_sample=unlabeled_sample,
            weight=weight,
        )
        logger.debug(f"AttrPrior criterion: {c}")
        return c
