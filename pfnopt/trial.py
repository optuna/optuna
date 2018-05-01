from typing import TYPE_CHECKING  # NOQA

from pfnopt import distributions
from pfnopt import frozen_trial

if TYPE_CHECKING:
    from pfnopt.study import Study  # NOQA
    from typing import Any  # NOQA
    from typing import Dict  # NOQA
    from typing import Sequence  # NOQA
    from typing import TypeVar  # NOQA

    T = TypeVar('T')


class Trial(object):

    """An active trial class that is passed to and communicates with users' objective functions."""

    def __init__(self, study, trial_id):
        # type: (Study, int) -> None
        self.study = study
        self.trial_id = trial_id

        self.study_id = self.study.study_id
        self.storage = self.study.storage

    def sample_uniform(self, name, low, high):
        # type: (str, float, float) -> float

        return self._sample(name, distributions.UniformDistribution(low=low, high=high))

    def sample_loguniform(self, name, low, high):
        # type: (str, float, float) -> float

        return self._sample(name, distributions.LogUniformDistribution(low=low, high=high))

    def sample_categorical(self, name, choices):
        # type: (str, Sequence[T]) -> T

        choices = tuple(choices)
        return self._sample(name, distributions.CategoricalDistribution(choices=choices))

    def complete(self, result):
        # type: (float) -> None

        self.storage.set_trial_value(self.trial_id, result)
        self.storage.set_trial_state(self.trial_id, frozen_trial.State.COMPLETE)

    def prune(self, step, current_result):
        # type: (int, float) -> bool

        self.storage.set_trial_intermediate_value(self.trial_id, step, current_result)
        ret = self.study.pruner.prune(self.storage, self.study_id, self.trial_id, step)
        return ret

    def set_user_attr(self, key, value):
        # type: (str, Any) -> None

        self.storage.set_trial_user_attr(self.trial_id, key, value)

    def _sample(self, name, distribution):
        # type: (str, distributions.BaseDistribution) -> Any

        # TODO(Akiba): if already sampled, return the recorded value
        # TODO(Akiba): check that distribution is the same

        self.storage.set_trial_param_distribution(self.trial_id, name, distribution)

        param_value_in_internal_repr = self.study.sampler.sample(
            self.storage, self.study_id, name, distribution)
        self.storage.set_trial_param(self.trial_id, name, param_value_in_internal_repr)
        param_value = distribution.to_external_repr(param_value_in_internal_repr)
        return param_value

    @property
    def params(self):
        # type: () -> Dict[str, Any]

        return self.storage.get_trial_params(self.trial_id)

    @property
    def user_attrs(self):
        # type: () -> Dict[str, Any]

        return self.storage.get_trial_user_attrs(self.trial_id)
