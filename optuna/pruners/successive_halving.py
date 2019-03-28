import math

from optuna.pruners.base import BasePruner
from optuna.storages import BaseStorage  # NOQA
from optuna.structs import FrozenTrial  # NOQA
from optuna.structs import StudyDirection
from optuna import types

if types.TYPE_CHECKING:
    from typing import List  # NOQA


class SuccessiveHalvingPruner(BasePruner):
    """Pruner using Asynchronous Successive Halving Algorithm.

    `Successive Halving <https://arxiv.org/abs/1502.07943>`_ is a bandit-based algorithm to
    identify the best one among multiple configurations. This class implements an asynchronous
    version of Successive Halving. Please refer to the paper of
    `Asynchronous Successive Halving <http://arxiv.org/abs/1810.05934>`_ for detailed descriptions.

    Note that, this class does not take care of the parameter for the maximum
    resource, referred to as :math:`R` in the paper. The maximum resource allocated to a trial is
    typically limited inside the objective function (e.g., ``step`` number in `simple.py
    <https://github.com/pfnet/optuna/tree/c5777b3e/examples/pruning/simple.py#L31>`_,
    ``EPOCH`` number in `chainer_integration.py
    <https://github.com/pfnet/optuna/tree/c5777b3e/examples/pruning/chainer_integration.py#L65>`_).

    Example:

        We minimize an objective function with ``SuccessiveHalvingPruner``.

        .. code::

            >>> from optuna import create_study
            >>> from optuna.pruners import SuccessiveHalvingPruner
            >>>
            >>> def objective(trial):
            >>>     ...
            >>>
            >>> study = create_study(pruner=SuccessiveHalvingPruner())
            >>> study.optimize(objective)

    Args:
        min_resource:
            A parameter for specifying the minimum resource allocated to a trial
            (in the `paper <http://arxiv.org/abs/1810.05934>`_ this parameter is
            referred to as :math:`r`).

            A trial is never pruned until it executes
            :math:`\\mathsf{min}\\_\\mathsf{resource} \\times
            \\mathsf{reduction}\\_\\mathsf{factor}^{
            \\mathsf{min}\\_\\mathsf{early}\\_\\mathsf{stopping}\\_\\mathsf{rate}}`
            steps (i.e., the completion point of the first rung). When the trial completes
            the first rung, it will be promoted to the next rung only
            if the value of the trial is placed in the top
            :math:`{1 \\over \\mathsf{reduction}\\_\\mathsf{factor}}` fraction of
            the all trials that already have reached the point (otherwise it will be pruned there).
            If the trial won the competition, it runs until the next completion point (i.e.,
            :math:`\\mathsf{min}\\_\\mathsf{resource} \\times
            \\mathsf{reduction}\\_\\mathsf{factor}^{
            (\\mathsf{min}\\_\\mathsf{early}\\_\\mathsf{stopping}\\_\\mathsf{rate}
            + \\mathsf{rung})}` steps)
            and repeats the same procedure.
        reduction_factor:
            A parameter for specifying reduction factor of promotable trials
            (in the `paper <http://arxiv.org/abs/1810.05934>`_ this parameter is
            referred to as :math:`\\eta`).  At the completion point of each rung,
            about :math:`{1 \\over \\mathsf{reduction}\\_\\mathsf{factor}}`
            trials will be promoted.
        min_early_stopping_rate:
            A parameter for specifying the minimum early-stopping rate
            (in the `paper <http://arxiv.org/abs/1810.05934>`_ this parameter is
            referred to as :math:`s`).
    """

    def __init__(self, min_resource=1, reduction_factor=4, min_early_stopping_rate=0):
        # type: (int, int, int) -> None

        if min_resource < 1:
            raise ValueError('The value of `min_resource` is {}, '
                             'but must be `min_resource >= 1`'.format(min_resource))

        if reduction_factor < 2:
            raise ValueError('The value of `reduction_factor` is {}, '
                             'but must be `reduction_factor >= 2`'.format(reduction_factor))

        if min_early_stopping_rate < 0:
            raise ValueError(
                'The value of `min_early_stopping_rate` is {}, '
                'but must be `min_early_stopping_rate >= 0`'.format(min_early_stopping_rate))

        self.min_resource = min_resource
        self.reduction_factor = reduction_factor
        self.min_early_stopping_rate = min_early_stopping_rate

    def prune(self, storage, study_id, trial_id, step):
        # type: (BaseStorage, int, int, int) -> bool
        """Please consult the documentation for :func:`BasePruner.prune`."""

        trial = storage.get_trial(trial_id)
        if len(trial.intermediate_values) == 0:
            return False

        rung = _get_current_rung(trial)
        value = trial.intermediate_values[step]
        all_trials = None
        while True:
            promotion_step = self.min_resource * \
                (self.reduction_factor ** (self.min_early_stopping_rate + rung))
            if step < promotion_step:
                return False

            if math.isnan(value):
                return True

            if all_trials is None:
                all_trials = storage.get_all_trials(study_id)

            storage.set_trial_system_attr(trial_id, _completed_rung_key(rung), value)
            direction = storage.get_study_direction(study_id)
            if not self._is_promotable(rung, value, all_trials, direction):
                return True

            rung += 1

    def _is_promotable(self, rung, value, all_trials, study_direction):
        # type: (int, float, List[FrozenTrial], StudyDirection) -> bool

        key = _completed_rung_key(rung)
        competing_values = [t.system_attrs[key] for t in all_trials if key in t.system_attrs]
        competing_values.append(value)
        competing_values.sort()

        promotable_idx = (len(competing_values) // self.reduction_factor) - 1
        if promotable_idx == -1:
            # Optuna does not support to suspend/resume ongoing trials.
            #
            # For the first `eta - 1` trials, this implementation promotes a trial if its
            # intermediate value is the smallest one among the trials that have completed the rung.
            promotable_idx = 0

        if study_direction == StudyDirection.MAXIMIZE:
            competing_values.reverse()
            return value >= competing_values[promotable_idx]

        return value <= competing_values[promotable_idx]


def _get_current_rung(trial):
    # type: (FrozenTrial) -> int

    # The following loop takes `O(log step)` iterations.
    rung = 0
    while _completed_rung_key(rung) in trial.system_attrs:
        rung += 1
    return rung


def _completed_rung_key(rung):
    # type: (int) -> str

    return 'completed_rung_{}'.format(rung)
