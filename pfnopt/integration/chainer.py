from __future__ import absolute_import

import chainer
from typing import TYPE_CHECKING

import pfnopt

if TYPE_CHECKING:
    from pfnopt.trial import Trial  # NOQA
    from typing import Tuple
    from typing import Union

    TriggerType = Union[Tuple[(int, str)], chainer.training.IntervalTrigger]


class ChainerPruningExtension(chainer.training.extension.Extension):

    def __init__(self, trial_, observation_key_, trigger_):
        # type: (Trial, str, TriggerType) -> None

        self.trial = trial_
        self.key = observation_key_
        self._trigger = chainer.training.get_trigger(trigger_)

    def __call__(self, trainer):
        # type: (chainer.training.Trainer) -> None

        if self._trigger(trainer):
            observation = trainer.observation
            if self.key in observation:
                current_step = getattr(trainer.updater, self._trigger.unit)
                current_score = float(observation[self.key])
                self.trial.report(current_score, step=current_step)
                if self.trial.should_prune(current_step):
                    msg = "Trial was pruned at {} {}.".format(self._trigger.unit, current_step)
                    raise pfnopt.pruners.TrialPruned(msg)
