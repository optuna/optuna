from typing import Any
from typing import Dict

import numpy

from optuna import distributions
from optuna.distributions import BaseDistribution
from optuna.samplers import BaseSampler
from optuna.study import Study
from optuna.trial import FrozenTrial


class PartialFixedSampler(BaseSampler):
    """Sampler that can sample with some parameters fixed.

        .. versionadded:: 2.2.0

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
            base_sampler = optuna.samplers.CmaEsSampler()
            partial_sampler = PartialFixedSampler(fixed_params, base_sampler)

            study.sampler = partial_sampler
            study.optimize(objective, n_trials=10)

    .. seealso::

        Please check :class:`optuna.samplers.BaseSampler` page for
        more information on how the sampler works.

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
        param_distribution: distributions.BaseDistribution,
    ) -> Any:

        # Fixed params will be sampled here.
        # If param_name isn't in self._fixed_params.keys(), param_value == None.
        param_value = self._fixed_params.get(param_name)

        if param_value:
            return param_value
        else:
            return self._base_sampler.sample_independent(
                study, trial, param_name, param_distribution
            )

    @property
    def fixed_params(self) -> Dict[str, Any]:
        """Return fixed parameters.

            .. versionadded:: 2.2.0

        Returns:
            A dictionary containing fixed parameters.
        """

        return self._fixed_params
