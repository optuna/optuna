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

    def __init__(self, n_startup_trials=5, n_warmup_steps=0):
        # type: (int, int) -> None

        self.n_startup_trials = n_startup_trials
        self.n_warmup_steps = n_warmup_steps

    def prune(self, storage, study_id, trial_id, step):
        # type: (BaseStorage, int, int, int) -> bool

        if len(storage.get_all_trials(study_id)) <= self.n_startup_trials:
            return False

        # step starts from 1.
        if step <= self.n_warmup_steps:
            return False

        if len(storage.get_trial(trial_id).intermediate_values) == 0:
            return False

        best_intermediate_result = storage.get_best_intermediate_result_over_steps(trial_id)
        median = storage.get_median_intermediate_result_over_trials(
            study_id, step)

        return best_intermediate_result > median
