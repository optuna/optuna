from __future__ import absolute_import

import math
from typing import TYPE_CHECKING

import pfnopt

try:
    import chainer
    from chainer.training.extension import Extension
    from chainer.training import triggers
    _available = True
except ImportError as e:
    _import_error = e
    Extension = object
    _available = False


if TYPE_CHECKING:
    from typing import Tuple
    from typing import Union

    TriggerType = Union[Tuple[(int, str)], triggers.IntervalTrigger,
                        triggers.ManualScheduleTrigger]


class ChainerPruningExtension(Extension):

    def __init__(self, trial, observation_key, pruner_trigger):
        # type: (pfnopt.trial.Trial, str, TriggerType) -> None

        _check_chainer_availability()

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

        _check_chainer_availability()

        if isinstance(observation_value, chainer.Variable):
            observation_value = observation_value.data

        try:
            observation_value = float(observation_value)
        except TypeError:
            raise TypeError(
                'Type of observation value is not supported by ChainerPruningExtension.\n'
                '{} cannot be casted to float.'.format(type(observation_value))
            )

        return observation_value

    def _observation_exists(self, trainer):
        # type: (chainer.training.Trainer) -> bool

        return self.pruner_trigger(trainer) and self.observation_key in trainer.observation

    def __call__(self, trainer):
        # type: (chainer.training.Trainer) -> None

        if not self._observation_exists(trainer):
            return

        current_score = self._get_float_value(trainer.observation[self.observation_key])
        if math.isnan(current_score):
            return

        current_step = getattr(trainer.updater, self.pruner_trigger.unit)
        self.trial.report(current_score, step=current_step)
        if self.trial.should_prune(current_step):
            message = "Trial was pruned at {} {}.".format(self.pruner_trigger.unit, current_step)
            raise pfnopt.structs.TrialPruned(message)


def _check_chainer_availability():
    # type: () -> None

    if not _available:
        raise ImportError(
            'Chainer is not available. Please install Chainer to use this feature. '
            'Chainer can be installed by executing `$ pip install chainer`. '
            'For further information, please refer to the installation guide of Chainer. '
            '(The actual import error is as follows: ' + str(_import_error) + ')')
