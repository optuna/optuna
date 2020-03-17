import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor, kernels

import optuna
from optuna import distributions
from optuna.samplers.base import BaseSampler
from optuna.samplers import random
from optuna import samplers
from optuna import type_checking

if type_checking.TYPE_CHECKING:
    from typing import Any  # NOQA
    from typing import Dict  # NOQA
    from typing import Optional  # NOQA

    from optuna.distributions import BaseDistribution  # NOQA
    from optuna.structs import FrozenTrial  # NOQA
    from optuna.study import Study  # NOQA

from optuna import structs
from optuna.structs import StudyDirection

class GPBO(BaseSampler):
    """Simple Bayesian Optimization based on Gaussian Process sampler.

    Example:

        .. code::

            >>> study = optuna.create_study(sampler=GPBO())
            >>> study.optimize(objective, direction='minimize')
    """

    def __init__(self):
        # type: () -> None
        self._random_sampler = random.RandomSampler()

    def infer_relative_search_space(self, study, trial):
        # type: (Study, FrozenTrial) -> Dict[str, BaseDistribution]

        search_space = {}
        for name, distribution in samplers.intersection_search_space(study).items():
            if type(distribution) == optuna.distributions.UniformDistribution:
                search_space[name] = distribution

        return search_space

    def sample_relative(self, study, trial, search_space):
        # type: (Study, FrozenTrial, Dict[str, BaseDistribution]) -> Dict[str, Any]
        if len(search_space) == 0:
            return {}

        complete_trials = [t for t in study.trials if t.state == structs.TrialState.COMPLETE]

        optimizer = _Optimizer(search_space)
        optimizer.tell(study, complete_trials)
        return optimizer.ask()

    def sample_independent(self, study, trial, param_name, param_distribution):
        # type: (Study, FrozenTrial, str, distributions.BaseDistribution) -> Any
        return self._random_sampler.sample_independent(
                study, trial, param_name, param_distribution
            )

class _Optimizer(object):
    def __init__(self, search_space):
        # type: (Dict[str, BaseDistribution], Dict[str, Any]) -> None

        self._search_space = search_space
        kernel = kernels.RBF(0.1)
        self._model = GaussianProcessRegressor(kernel=kernel)

        self._best_value = 0

    def _trial_to_x_y(self, study, trial):
        # type: (Study, FrozenTrial) -> Tuple[List[Any], float]

        param_values = []
        for name, distribution in sorted(self._search_space.items()):
            param_value = trial.params[name]
            param_values.append(param_value)

        value = trial.value
        assert value is not None

        if study.direction == StudyDirection.MAXIMIZE:
            value = -value

        return param_values, value

    def debug(self):
        plt.figure()
        plt.scatter(self._model.X_train_, self._model.y_train_)
        x_grid = np.asarray(np.arange(0, 1, 0.001))
        x_grid = x_grid.reshape(len(x_grid), 1)
        mu, std = self._model.predict(x_grid, return_std=True)
        plt.plot(x_grid, mu)
        plt.fill_between(x_grid[:,0], mu-1.96*std, mu+1.96*std, color='red', alpha=0.5, label='95% Confidence Interval')

        plt.ylim(-2,2)
        plt.savefig('debug/{:02d}.jpg'.format(len(self._model.y_train_)))

    def tell(self, study, complete_trials):
        # type: (Study, List[FrozenTrial]) -> None

        xs = []
        ys = []

        for trial in complete_trials:
            x, y = self._trial_to_x_y(study, trial)
            xs.append(x)
            ys.append(y)

        xs = np.array(xs)
        ys = np.array(ys)
        self._tell(xs, ys)

        self._best_value = np.min(ys)

    def _tell(self, X, y):
        # type: (Any, Any) -> None
 
        self._model.fit(X, y)

    def _ask(self):
        # type: () -> None

        x = []
        for i in range(100):
            param_dict = dict()
            for param_name, param_distribution in self._search_space.items():
                assert type(param_distribution) == optuna.distributions.UniformDistribution
                param_value = np.random.uniform(param_distribution.low, param_distribution.high)
                param_dict[param_name] = param_value

            param_values = []
            for name, distribution in sorted(param_dict.items()):
                param_value = param_dict[name]
                param_values.append(param_value)

            x.append(param_values)

        x = np.array(x)
 
        mu, std =  self._model.predict(x, return_std=True)

        probs = norm.cdf((mu - self._best_value) / (std+1E-9))
        best_idx = np.argmin(probs)

        self.debug()
        return x[best_idx]

    def ask(self):
        # type: () -> Dict[str, Any]

        params = {}
        param_values = self._ask()

        for (name, distribution), value in zip(sorted(self._search_space.items()), param_values):
            params[name] = value

        return params
