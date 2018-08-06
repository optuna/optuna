from __future__ import absolute_import

import chainer
from chainer.training import triggers
import math
from typing import Any  # NOQA
from typing import TYPE_CHECKING

import pfnopt

if TYPE_CHECKING:
    from typing import Tuple
    from typing import Union

    TriggerType = Union[Tuple[(int, str)], chainer.training.IntervalTrigger]


class ChainerPruningExtension(chainer.training.extension.Extension):

    def __init__(self, trial, observation_key, pruner_trigger):
        # type: (pfnopt.trial.Trial, str, TriggerType) -> None

        self.trial = trial
        self.observation_key = observation_key
        self.pruner_trigger = chainer.training.get_trigger(pruner_trigger)
        if not (isinstance(self.pruner_trigger, triggers.IntervalTrigger) or
                isinstance(self.pruner_trigger, triggers.ManualScheduleTrigger)):
            pruner_type = type(self.pruner_trigger)
            raise TypeError(
                "Invalid trigger class: " + str(pruner_type) + "\n"
                "Pruner trigger is supposed to be an instance of "
                "IntervalTrigger or ManualScheduleTrigger.")

    def _get_score(self, observation_value):
        # type: (Any) -> float

        score = observation_value
        if isinstance(score, chainer.Variable):
            score = score.data
        score = float(score)

        return score

    def __call__(self, trainer):
        # type: (chainer.training.Trainer) -> None

        if not self.pruner_trigger(trainer):
            return
        if self.observation_key not in trainer.observation:
            return

        current_step = getattr(trainer.updater, self.pruner_trigger.unit)
        current_score = self._get_score(trainer.observation[self.observation_key])
        if math.isnan(current_score):
            return

        self.trial.report(current_score, step=current_step)
        if self.trial.should_prune(current_step):
            msg = "Trial was pruned at {} {}.".format(self.pruner_trigger.unit, current_step)
            raise pfnopt.pruners.TrialPruned(msg)
