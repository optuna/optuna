from typing import Tuple
from typing import Union

import optuna


with optuna._imports.try_import() as _imports:
    import chainer
    from chainer.training.extension import Extension
    from chainer.training.triggers import IntervalTrigger
    from chainer.training.triggers import ManualScheduleTrigger

if not _imports.is_successful():
    Extension = object  # type: ignore # NOQA


class ChainerPruningExtension(Extension):
    """Chainer extension to prune unpromising trials.

    See `the example <https://github.com/optuna/optuna/blob/master/
    examples/pruning/chainer_integration.py>`__
    if you want to add a pruning extension which observes validation
    accuracy of a `Chainer Trainer <https://docs.chainer.org/en/stable/
    reference/generated/chainer.training.Trainer.html>`_.

    Args:
        trial:
            A :class:`~optuna.trial.Trial` corresponding to the current evaluation of the
            objective function.
        observation_key:
            An evaluation metric for pruning, e.g., ``main/loss`` and
            ``validation/main/accuracy``. Please refer to
            `chainer.Reporter reference <https://docs.chainer.org/en/stable/reference/
            util/generated/chainer.Reporter.html>`_ for further details.
        pruner_trigger:
            A trigger to execute pruning. ``pruner_trigger`` is an instance of
            `IntervalTrigger <https://docs.chainer.org/en/stable/reference/generated/
            chainer.training.triggers.IntervalTrigger.html>`_ or
            `ManualScheduleTrigger <https://docs.chainer.org/en/stable/reference/generated/
            chainer.training.triggers.ManualScheduleTrigger.html>`_. `IntervalTrigger <https://
            docs.chainer.org/en/stable/reference/generated/chainer.training.triggers.
            IntervalTrigger.html>`_ can be specified by a tuple of the interval length and its
            unit like ``(1, 'epoch')``.
    """

    def __init__(
        self,
        trial: optuna.trial.Trial,
        observation_key: str,
        pruner_trigger: Union[Tuple[(int, str)], "IntervalTrigger", "ManualScheduleTrigger"],
    ) -> None:

        _imports.check()

        self._trial = trial
        self._observation_key = observation_key
        self._pruner_trigger = chainer.training.get_trigger(pruner_trigger)
        if not (
            isinstance(self._pruner_trigger, IntervalTrigger)
            or isinstance(self._pruner_trigger, ManualScheduleTrigger)
        ):
            pruner_type = type(self._pruner_trigger)
            raise TypeError(
                "Invalid trigger class: " + str(pruner_type) + "\n"
                "Pruner trigger is supposed to be an instance of "
                "IntervalTrigger or ManualScheduleTrigger."
            )

    @staticmethod
    def _get_float_value(observation_value: Union[float, "chainer.Variable"]) -> float:

        _imports.check()

        if isinstance(observation_value, chainer.Variable):
            observation_value = observation_value.data  # type: ignore

        try:
            observation_value = float(observation_value)  # type: ignore
        except TypeError:
            raise TypeError(
                "Type of observation value is not supported by ChainerPruningExtension.\n"
                "{} cannot be cast to float.".format(type(observation_value))
            ) from None

        return observation_value

    def _observation_exists(self, trainer: "chainer.training.Trainer") -> bool:

        return self._pruner_trigger(trainer) and self._observation_key in trainer.observation

    def __call__(self, trainer: "chainer.training.Trainer") -> None:

        if not self._observation_exists(trainer):
            return

        current_score = self._get_float_value(trainer.observation[self._observation_key])
        current_step = getattr(trainer.updater, self._pruner_trigger.unit)
        self._trial.report(current_score, step=current_step)
        if self._trial.should_prune():
            message = "Trial was pruned at {} {}.".format(self._pruner_trigger.unit, current_step)
            raise optuna.TrialPruned(message)
