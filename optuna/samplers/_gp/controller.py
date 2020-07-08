import math
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import numpy as np

from optuna import distributions
from optuna.samplers._gp.acquisition import acquisition_selector
from optuna.samplers._gp.acquisition import BaseAcquisitionFunction
from optuna.samplers._gp.model import BaseModel
from optuna.samplers._gp.model import model_selector
from optuna.samplers._gp.optimizer import BaseOptimizer
from optuna.samplers._gp.optimizer import optimizer_selector
from optuna.study import Study
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial


class _BayesianOptimizationController(object):
    def __init__(
        self,
        search_space: Dict[str, distributions.BaseDistribution],
        model: Union[str, BaseModel] = 'SVGP',
        acquisition: Union[str, BaseAcquisitionFunction] = 'EI',
        optimizer: Union[str, BaseOptimizer] = 'L-BFGS-B',
    ) -> None:

        self._search_space = search_space

        if isinstance(model, BaseModel):
            self._model = model
        else:
            self._model = model_selector(model)

        if isinstance(acquisition, BaseAcquisitionFunction):
            self._acquisition = acquisition
        else:
            self._acquisition = acquisition_selector(acquisition)

        if isinstance(optimizer, BaseOptimizer):
            self._optimizer = optimizer
        else:
            self._optimizer = optimizer_selector(optimizer, self._convert_search_space())

    def tell(self, study: Study, trials: List[FrozenTrial]) -> None:

        xs = []
        ys = []
        for trial in trials:
            if not self._is_compatible(trial):
                continue

            x, y = self._trial_to_observation_pair(study, trial)
            xs.append(x)
            ys.append(y)
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)

        self._model.add_data(xs, ys)

    def ask(self) -> Dict[str, Any]:

        def objective(x):
            return self._acquisition.compute_acq(x=x, model=self._model)

        def derivative(x):
            return self._acquisition.compute_grad(x=x, model=self._model)

        param_values = self._optimizer.optimize(f=objective, df=derivative)
        params = {}
        for (name, distribution), param_value in zip(sorted(self._search_space.items()), param_values):
            if isinstance(distribution, distributions.LogUniformDistribution):
                param_value = math.exp(param_value)
            elif isinstance(distribution, distributions.DiscreteUniformDistribution):
                param_value = param_value * distribution.q + distribution.low
                param_value = float(min(max(param_value, distribution.low), distribution.high))
            elif isinstance(distribution, distributions.IntUniformDistribution):
                param_value = param_value * distribution.step + distribution.low
                param_value = int(min(max(param_value, distribution.low), distribution.high))
            elif isinstance(distribution, distributions.IntLogUniformDistribution):
                param_value = param_value + math.log(distribution.low)
                param_value = int(math.exp(param_value))
            params[name] = param_value

        return params

    def _is_compatible(self, trial: FrozenTrial) -> bool:

        # Thanks to `intersection_search_space()` function, in sequential optimization,
        # the parameters of complete trials are always compatible with the search space.
        #
        # However, in distributed optimization, incompatible trials may complete on a worker
        # just after an intersection search space is calculated on another worker.

        for name, distribution in self._search_space.items():
            if name not in trial.params:
                return False

            distributions.check_distribution_compatibility(distribution, trial.distributions[name])
            param_value = trial.params[name]
            param_internal_value = distribution.to_internal_repr(param_value)
            if not distribution._contains(param_internal_value):
                return False

        return True

    def _trial_to_observation_pair(
        self, study: Study, trial: FrozenTrial
    ) -> Tuple[List[Any], float]:

        param_values = []
        for name, distribution in sorted(self._search_space.items()):
            param_value = trial.params[name]

            if isinstance(distribution, distributions.LogUniformDistribution):
                param_value = math.log(param_value)
            elif isinstance(distribution, distributions.DiscreteUniformDistribution):
                param_value = np.round((param_value - distribution.low) / distribution.q)
            elif isinstance(distribution, distributions.IntUniformDistribution):
                param_value = np.round((param_value - distribution.low) / distribution.step)
            elif isinstance(distribution, distributions.IntLogUniformDistribution):
                param_value = np.round(math.log(param_value) - math.log(distribution.low))

            param_values.append(param_value)

        value = trial.value
        assert value is not None

        if study.direction == StudyDirection.MAXIMIZE:
            value = -value

        return param_values, value

    def _convert_search_space(self) -> np.ndarray:

        bounds = np.zeros((len(self._search_space), 2))

        for i, (_, distribution) in enumerate(sorted(self._search_space.items())):
            if isinstance(distribution, distributions.UniformDistribution):
                low = distribution.low
                high = distribution.high
            elif isinstance(distribution, distributions.LogUniformDistribution):
                low = math.log(distribution.low)
                high = math.log(distribution.high)
            elif isinstance(distribution, distributions.DiscreteUniformDistribution):
                low = 0.
                high = np.round((distribution.high - distribution.low) / distribution.q)
            elif isinstance(distribution, distributions.IntUniformDistribution):
                low = 0.
                high = np.round((distribution.high - distribution.low) / distribution.step)
            elif isinstance(distribution, distributions.IntLogUniformDistribution):
                low = 0.
                high = np.round(math.log(distribution.high) - math.log(distribution.low))
            else:
                raise NotImplementedError(
                    "The distribution {} is not implemented.".format(distribution)
                )

            bounds[i][0] = low
            bounds[i][1] = high

        return bounds
