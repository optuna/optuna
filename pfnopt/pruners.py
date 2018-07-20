import abc
import six

from pfnopt.storages import BaseStorage  # NOQA


@six.add_metaclass(abc.ABCMeta)
class BasePruner(object):

    @abc.abstractmethod
    def prune(self, storage, study_id, trial_id, step):
        # type: (BaseStorage, int, int, int) -> bool
        raise NotImplementedError


class MedianPruner(BasePruner):

    """Pruner using median

     Prune if the trial's best intermediate result is worse than
     median of intermediate results of previous trials at the same step.
    """

    n_startup_trials = 5  # TODO(Akiba): parameterize

    def prune(self, storage, study_id, trial_id, step):
        # type: (BaseStorage, int, int, int) -> bool

        if trial_id < self.n_startup_trials:
            return False

        best_intermediate_result = storage.get_best_intermediate_result_over_steps(trial_id)
        median = storage.get_median_intermediate_result_over_trials(
            study_id, step)

        print(step, best_intermediate_result, median)

        return best_intermediate_result > median


class WarmupMedianPruner(MedianPruner):

    """Pruner using median with Warm-up period

     If number of steps is smaller than threshold n_warmup_steps,
     this pruner never prunes trials. Otherwise, it works like MedianPruner.

    """

    def __init__(self, n_warmup_steps=5):
        # type: (int) -> None

        self.n_warmup_steps = n_warmup_steps

    def prune(self, storage, study_id, trial_id, step):
        # type: (BaseStorage, int, int, int) -> bool

        if step < self.n_warmup_steps:
            return False
        return super(WarmupMedianPruner, self).prune(storage, study_id, trial_id, step)


class TrialPruned(Exception):
    pass
