from typing import TYPE_CHECKING  # NOQA

from optuna import distributions

if TYPE_CHECKING:
    from optuna.study import Study  # NOQA
    from typing import Any  # NOQA
    from typing import Dict  # NOQA
    from typing import Optional  # NOQA
    from typing import Sequence  # NOQA
    from typing import TypeVar  # NOQA

    T = TypeVar('T', float, str)


class Trial(object):

    """A trial is a process of evaluating an objective function.

    This object is passed to an objective function, and provides interfaces to get parameter
    suggestion, manage the trial's state, and set/get user-defined attributes of the trial.

    Note that this object is seamlessly instantiated and passed to the objective function behind
    Study.run() method (as well as optimize function); hence, in typical use cases,
    library users do not care about instantiation of this object.

    Args:
        study:
            A study object.
        trial_id:
            A trial ID populated by a storage object.

    """

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

    def suggest_discrete_uniform(self, name, low, high, q):
        # type: (str, float, float, float) -> float

        discrete = distributions.DiscreteUniformDistribution(low=low, high=high, q=q)
        return self._suggest(name, discrete)

    def suggest_int(self, name, low, high):
        # type: (str, int, int) -> int

        return int(self._suggest(name, distributions.IntUniformDistribution(low=low, high=high)))

    def suggest_categorical(self, name, choices):
        # type: (str, Sequence[T]) -> T

        choices = tuple(choices)
        return self._suggest(name, distributions.CategoricalDistribution(choices=choices))

    def report(self, value, step=None):
        # type: (float, Optional[int]) -> None

        self.storage.set_trial_value(self.trial_id, value)
        if step is not None:
            self.storage.set_trial_intermediate_value(self.trial_id, step, value)

    def should_prune(self, step):
        # type: (int) -> bool

        # TODO(akiba): remove `step` argument

        return self.study.pruner.prune(
            self.storage, self.study_id, self.trial_id, step)

    def set_user_attr(self, key, value):
        # type: (str, Any) -> None

        self.storage.set_trial_user_attr(self.trial_id, key, value)

    def set_system_attr(self, key, value):
        # type: (str, Any) -> None

        self.storage.set_trial_system_attr(self.trial_id, key, value)

    def _suggest(self, name, distribution):
        # type: (str, distributions.BaseDistribution) -> Any

        param_value_in_internal_repr = self.study.sampler.sample(
            self.storage, self.study_id, name, distribution)

        set_success = self.storage.set_trial_param(
            self.trial_id, name, param_value_in_internal_repr, distribution)
        if not set_success:
            param_value_in_internal_repr = self.storage.get_trial_param(self.trial_id, name)

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

    @property
    def system_attrs(self):
        # type: () -> Dict[str, Any]

        return self.storage.get_trial_system_attrs(self.trial_id)
