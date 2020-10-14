from typing import Any
from typing import Dict
import warnings

import numpy

from optuna._experimental import experimental
from optuna.distributions import BaseDistribution
from optuna.samplers import BaseSampler
from optuna.study import Study
from optuna.trial import FrozenTrial


@experimental("2.3.0")
class PartialFixedSampler(BaseSampler):
    """Sampler that can sample with some parameters fixed.

        .. versionadded:: 2.3.0

    Example:

        After optimizing with :class:`~optuna.samplers.TPESampler`,
        fix the value of ``y`` and optimize again with :class:`~optuna.samplers.CmaEsSampler`.

        .. testcode::

            import optuna

            def objective(trial):
                x = trial.suggest_uniform("x", -1, 1)
                y = trial.suggest_int("y", -1, 1)
                return x ** 2 + y

            study = optuna.create_study()
            study.optimize(objective, n_trials=10)

            best_params = study.best_params
            fixed_params = {"y": best_params["y"]}
            partial_sampler = PartialFixedSampler(fixed_params, study.sampler)

            study.sampler = partial_sampler
            study.optimize(objective, n_trials=10)

    """

    def __init__(self, fixed_params: Dict[str, Any], base_sampler: BaseSampler) -> None:
        self._fixed_params = fixed_params
        self._base_sampler = base_sampler

    def reseed_rng(self) -> None:

        self._rng = numpy.random.RandomState()

    def infer_relative_search_space(
        self,
        study: Study,
        trial: FrozenTrial,
    ) -> Dict[str, BaseDistribution]:

        search_space = self._base_sampler.infer_relative_search_space(study, trial)

        # Remove fixed params from relative search space to return fixed values.
        for param_name in self._fixed_params.keys():
            if param_name in search_space:
                del search_space[param_name]

        return search_space

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: Dict[str, BaseDistribution],
    ) -> Dict[str, Any]:

        # Fixed params are never sampled here.
        return self._base_sampler.sample_relative(study, trial, search_space)

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:

        # Fixed params will be sampled here.

        # If param_name isn't in self._fixed_params.keys(), param_value == None.
        param_value = self._fixed_params.get(param_name)

        if param_value is None:
            # Unfixed params are sampled here.
            return self._base_sampler.sample_independent(
                study, trial, param_name, param_distribution
            )
        else:
            # Fixed params are sampled here.
            # Check if a parameter value is contained in the range of this distribution.
            param_value_in_internal_repr = param_distribution.to_internal_repr(param_value)
            contained = param_distribution._contains(param_value_in_internal_repr)

            if not contained:
                warnings.warn(
                    "Fixed parameter '{}' with value {} is out of range "
                    "for distribution {}.".format(param_name, param_value, param_distribution)
                )
            return param_value

    @property
    def fixed_params(self) -> Dict[str, Any]:
        """Return fixed parameters.

        Returns:
            A dictionary containing fixed parameters.
        """

        return self._fixed_params
