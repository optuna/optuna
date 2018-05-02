from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import chainer
    from pfnopt.trial import Trial  # NOQA
    from typing import Tuple
    from typing import Union

    TriggerType = Union[Tuple[(int, str)], chainer.training.IntervalTrigger]


def create_chainer_pruning_trigger(
        trial, observation_key, stop_trigger, test_trigger=(1, 'epoch')):
    # type: (Trial, str, TriggerType, TriggerType) -> TriggerType

    import chainer.training

    class _ChainerTrigger(chainer.training.IntervalTrigger):

        """The trigger class for Chainer to prune with intermediate results.

        """

        # This class inherits IntervalTrigger to properly work with Chainer's ProgressBar

        def __init__(self, trial_, observation_key_, stop_trigger_, test_trigger_):
            # type: (Trial, str, TriggerType, TriggerType) -> None

            stop_trigger_ = chainer.training.get_trigger(stop_trigger_)
            test_trigger_ = chainer.training.get_trigger(test_trigger_)
            # TODO(Akiba): raise ValueError
            assert isinstance(test_trigger_, chainer.training.IntervalTrigger)
            super(_ChainerTrigger, self).__init__(stop_trigger_.period, stop_trigger_.unit)

            self.trial = trial_
            self.stop_trigger = stop_trigger_
            self.test_trigger = test_trigger_
            self.key = observation_key_

        def __call__(self, trainer):
            # type: (chainer.training.Trainer) -> bool

            if self.stop_trigger(trainer):
                return True

            if not self.test_trigger(trainer):
                return False

            observation = trainer.observation
            if self.key not in observation:
                return False

            current_step = getattr(trainer.updater, self.test_trigger.unit)
            current_score = float(observation[self.key])
            self.trial.report(current_score, step=current_step)
            return self.trial.should_prune(current_step)

    return _ChainerTrigger(trial, observation_key, stop_trigger, test_trigger)
