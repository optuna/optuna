import math

from optuna.pruners.base import BasePruner
from optuna.storages import BaseStorage  # NOQA
from optuna.structs import FrozenTrial  # NOQA
from typing import List  # NOQA


class SuccessiveHalvingPruner(BasePruner):

    """Pruner using Asynchronous Successive Halving Algorithm.

    `Successive Halving <https://arxiv.org/abs/1502.07943>`_ is a bandit-based algorithm to
    identify the best one among multiple configurations. This class implements an asynchronous
    version of Successive Halving. Please refer to the paper of
    `Asynchronous Successive Halving <http://arxiv.org/abs/1810.05934>`_ for detailed descriptions.

    Note that, this class does not take care of the parameter for the maximum
    resource, referred to as ``R`` in the paper. The maximum resource allocated to a trial is
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
            referred to as "r").

            More precisely, a trial is never pruned until it executes
            ``min_resource * (reduction_factor ** min_early_stopping_rate)`` steps
            (i.e., the completion point of the first rung). When the trial completes the first
            rung, it will be promoted to the next rung only if the value of the trial is placed in
            the top ``1/reduction_factor`` fraction of the whole trials that already have reached
            the point (otherwise it will be pruned there). If the trial won
            the competition, it continues to execute its work until the next rung completion point
            (i.e., ``min_resource * (reduction_factor ** (min_early_stopping_rate + rung))`` steps)
            is reached and then repeats the same process with a new ``rung``.
        reduction_factor:
            A parameter for specifying reduction factor of promotable trials
            (in the `paper <http://arxiv.org/abs/1810.05934>`_ this parameter is
            referred to as "eta").
            At the completion point of each rung, about ``1/reduction_factor`` trials
            will be promoted.
        min_early_stopping_rate:
            A parameter for specifying the minimum early-stopping rate
            (in the `paper <http://arxiv.org/abs/1810.05934>`_ this parameter is
            referred to as "s").
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

        rung = self._get_current_rung(trial)
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

            storage.set_trial_system_attr(trial_id, completed_rung_key(rung), value)
            if not self._is_promotable(storage, rung, value, all_trials):
                return True

            rung += 1

    def _is_promotable(self, storage, rung, value, all_trials):
        # type: (BaseStorage, int, float, List[FrozenTrial]) -> bool

        key = completed_rung_key(rung)
        competing_values = [t.system_attrs[key] for t in all_trials if key in t.system_attrs]
        competing_values.append(value)
        competing_values.sort()

        promotable_idx = (len(competing_values) // self.reduction_factor) - 1
        if promotable_idx == -1:
            # Optuna does not support to suspend/resume ongoing trials.
            #
            # As a result, a trial that has reached early to a rung cannot wait for
            # the following trials even if the rung contains only trials less than `eta`.
            #
            # So, we allow the first `eta - 1` trials to become promotable
            # if its value is the smallest among the trials that already have reached the rung.
            promotable_idx = 0

        # TODO(ohta): Deal with maximize direction.
        return value <= competing_values[promotable_idx]

    def _get_current_rung(self, trial):
        # type: (FrozenTrial) -> int

        # Below loop takes `O(log step)` iterations.
        rung = 0
        while completed_rung_key(rung) in trial.system_attrs:
            rung += 1
        return rung


def completed_rung_key(rung):
    # type: (int) -> str

    return 'completed_rung_{}'.format(rung)
