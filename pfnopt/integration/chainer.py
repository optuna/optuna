from __future__ import absolute_import

import chainer
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

    def __call__(self, trainer):
        # type: (chainer.training.Trainer) -> None

        if self.pruner_trigger(trainer) and self.observation_key in trainer.observation:
            current_step = getattr(trainer.updater, self.pruner_trigger.unit)
            current_score = float(trainer.observation[self.observation_key])
            self.trial.report(current_score, step=current_step)
            if self.trial.should_prune(current_step):
                msg = "Trial was pruned at {} {}.".format(self.pruner_trigger.unit, current_step)
                raise pfnopt.pruners.TrialPruned(msg)
