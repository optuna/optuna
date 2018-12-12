import math

from optuna.pruners.base import BasePruner
from optuna.storages import BaseStorage  # NOQA
from optuna.structs import FrozenTrial  # NOQA
from typing import List  # NOQA


class SuccessiveHalvingPruner(BasePruner):

    """Pruner using the Asynchronous Successive Halving Algorithm (ASHA).

    ASHA defines rounds (named "rung") started from ``0``.
    When a trial completed the current rung, it competes with other trials that have completed
    the same rung. And if it wins the competition, the trial will be promoted to the next rung
    for continuing the work. Conversely, the losers will be pruned there.
    This process is repeated by recursively until the trial finishes.

    Please refer to `the original paper <http://arxiv.org/abs/1810.05934>`_
    for a detailed description of ASHA.

    Note that, unlike the paper, ``SuccessiveHalvingPruner`` recognizes only "number of steps" as
    the resource consumed by a trial (in the paper, for example, input data size can be treated as
    a resource). Besides, it does not have a parameter to restrict the maximum resource usage
    (called ``R`` in the paper). The maximum number of steps executed by a trial is implicitly
    limited by users via implementation specific parameters (e.g., `step number in simple.py
    <https://github.com/pfnet/optuna/tree/c5777b3e/examples/pruning/simple.py#L31>`_,
    `epoch number in chainer_integration.py
    <https://github.com/pfnet/optuna/tree/c5777b3e/examples/pruning/chainer_integration.py#L65>`_).

    Example:

        We minimize an objective function with ASHA.

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
        r:
            A parameter for specifying the minimum resource allocated to a trial.
            More precisely, a trial is never pruned until it executes ``r * (eta ** s)`` steps
            (i.e., the completion point of the first rung).
            When it completes the first rung, it will be promoted to the next rung only if
            the value of the trial is placed in the top ``1/eta`` fraction of the whole trials that
            already have reached the point (otherwise it will be pruned there). If the trial won
            the competition, it continues to execute its work until the next rung
            completion point (i.e., ``r * (eta ** (s + rung))`` steps) is reached and then repeats
            the same process with a new ``rung``.
        eta:
            A parameter for specifying reduction factor of promotable trials.
            At the completion point of the each rung, about ``1/eta`` trials will be pruned.
        s:
            A parameter for specifying the completion point of the first rung.
            The point is determined by the expression ``r * (eta ** s)``.
    """

    def __init__(self, r=1, eta=4, s=0):
        # type: (int, int, int) -> None

        if r < 1:
            raise ValueError('`r` must be greater than `0`')

        if eta < 2:
            raise ValueError('`eta` must be greater than `1`')

        if s < 0:
            raise ValueError('`s` must be positive')

        self.r = r
        self.eta = eta
        self.s = s

    def prune(self, storage, study_id, trial_id, step):
        # type: (BaseStorage, int, int, int) -> bool
        """Please consult the documentation for :func:`BasePruner.prune`."""

        trial = storage.get_trial(trial_id)
        if len(trial.intermediate_values) == 0:
            return False

        rung = get_current_rung(trial)
        value = trial.intermediate_values[step]
        all_trials = None
        while True:
            promotion_step = self.r * (self.eta ** (self.s + rung))
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

        promotable_idx = (len(competing_values) // self.eta) - 1
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


def completed_rung_key(rung):
    # type: (int) -> str

    return 'completed_rung_{}'.format(rung)


def get_current_rung(trial):
    # type: (FrozenTrial) -> int

    # Below loop takes `O(log step)` iterations.
    rung = 0
    while completed_rung_key(rung) in trial.system_attrs:
        rung += 1
    return rung
