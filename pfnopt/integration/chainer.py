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

    @staticmethod
    def _get_float_value(observation_value):
        # type: (Union[float, chainer.Variable]) -> float

        value = observation_value
        if isinstance(value, chainer.Variable):
            value = value.data

        try:
            value = float(value)
        except TypeError:
            raise TypeError(
                'Type of observation value is not supported by ChainerPruningExtension.\n'
                '{} cannot be casted to float.'.format(type(value))
            )

        return value

    def _has_new_observation(self, trainer):
        # type: (chainer.training.Trainer) -> bool

        return self.pruner_trigger(trainer) and self.observation_key in trainer.observation

    def __call__(self, trainer):
        # type: (chainer.training.Trainer) -> None

        if not self._has_new_observation(trainer):
            return

        current_score = self._get_float_value(trainer.observation[self.observation_key])
        if math.isnan(current_score):
            return

        current_step = getattr(trainer.updater, self.pruner_trigger.unit)
        self.trial.report(current_score, step=current_step)
        if self.trial.should_prune(current_step):
            message = "Trial was pruned at {} {}.".format(self.pruner_trigger.unit, current_step)
            raise pfnopt.structs.TrialPruned(message)
