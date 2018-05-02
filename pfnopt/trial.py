from typing import TYPE_CHECKING  # NOQA

from pfnopt import distributions
from pfnopt import frozen_trial

if TYPE_CHECKING:
    from pfnopt.study import Study  # NOQA
    from typing import Any  # NOQA
    from typing import Dict  # NOQA
    from typing import Optional  # NOQA
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

    def suggest_uniform(self, name, low, high):
        # type: (str, float, float) -> float

        return self._suggest(name, distributions.UniformDistribution(low=low, high=high))

    def suggest_loguniform(self, name, low, high):
        # type: (str, float, float) -> float

        return self._suggest(name, distributions.LogUniformDistribution(low=low, high=high))

    def suggest_categorical(self, name, choices):
        # type: (str, Sequence[T]) -> T

        choices = tuple(choices)
        return self._suggest(name, distributions.CategoricalDistribution(choices=choices))

    def report(self, value, step=None):
        # type: (float, Optional[int]) -> None

        self.storage.set_trial_value(self.trial_id, value)
        if step is not None:
            self.storage.set_trial_intermediate_value(self.trial_id, step, value)

    def complete(self):
        # type: () -> None

        self.storage.set_trial_state(self.trial_id, frozen_trial.State.COMPLETE)

    def should_prune(self, step):
        # type: (int) -> bool

        # TODO(akiba): remove `step` argument

        return self.study.pruner.prune(
            self.storage, self.study_id, self.trial_id, step)

    def set_user_attr(self, key, value):
        # type: (str, Any) -> None

        self.storage.set_trial_user_attr(self.trial_id, key, value)

    def _suggest(self, name, distribution):
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
