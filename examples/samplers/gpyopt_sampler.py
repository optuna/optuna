"""
Optuna example that implements a custom relative sampler that uses GPyOpt as the backend.

Note that this implementation isn't intended to be used for production purposes and
only supports `UniformDistribution` (i.e., `Trial.suggest_uniform` method).

You can run this example as follows:
    $ python gpyopt_sampler.py

"""

import GPyOpt
import numpy as np

import optuna
from optuna import distributions
from optuna.samplers import BaseSampler
from optuna import structs

if optuna.types.TYPE_CHECKING:
    from typing import Any  # NOQA
    from typing import Dict  # NOQA
    from typing import List  # NOQA
    from typing import Optional  # NOQA

    from optuna.distributions import BaseDistribution  # NOQA
    from optuna.structs import FrozenTrial  # NOQA
    from optuna.study import InTrialStudy  # NOQA


class GPyOptSampler(BaseSampler):
    def __init__(self, n_startup_trials=5, seed=None):
        # type: (int, Optional[int]) -> None

        self._n_startup_trials = n_startup_trials
        self._independent_sampler = optuna.samplers.RandomSampler(seed=seed)

    def infer_relative_search_space(self, study, trial):
        # type: (InTrialStudy, FrozenTrial) -> Dict[str, BaseDistribution]

        return optuna.samplers.product_search_space(study)

    def sample_relative(self, study, trial, search_space):
        # type: (InTrialStudy, FrozenTrial, Dict[str, BaseDistribution]) -> Dict[str, Any]

        if search_space == {}:
            return {}

        trials = [t for t in study.trials if t.state == structs.TrialState.COMPLETE]
        if len(trials) < self._n_startup_trials:
            return {}

        optimizer = self._create_gpyopt_optimizer(trials, study, search_space)

        params = optimizer.suggest_next_locations()
        params = {p: params[0, i] for i, p in enumerate(search_space.keys())}

        return params

    def _create_gpyopt_optimizer(
            self,
            trials,  # type: List[FrozenTrial]
            study,  # type: InTrialStudy
            search_space,  # type: Dict[str, BaseDistribution]
    ):
        # type: (...) -> GPyOpt.methods.BayesianOptimization

        domain = []
        for param_name, param_distribution in search_space.items():
            if not isinstance(param_distribution, distributions.UniformDistribution):
                raise NotImplementedError(
                    "The distribution {} of parameter '{}' is unsupported.".format(
                        param_distribution, param_name))

            type = 'continuous'
            range = (param_distribution.low, param_distribution.high)
            domain.append({'name': param_name, 'type': type, 'domain': range})

        if study.direction == structs.StudyDirection.MINIMIZE:
            sign = -1
        else:
            sign = 1

        X = np.array([[t.params[p] for p in search_space.keys()] for t in trials])
        Y = np.array([[sign * t.value] for t in trials])

        return GPyOpt.methods.BayesianOptimization(f=None, domain=domain, X=X, Y=Y)

    def sample_independent(self, study, trial, param_name, param_distribution):
        # type: (InTrialStudy, FrozenTrial, str, BaseDistribution) -> Any

        return self._independent_sampler.sample_independent(study, trial, param_name,
                                                            param_distribution)


# Define a simple 2-dimensional objective function whose minimum value is -1 when (x, y) = (0, -1).
def objective(trial):
    x = trial.suggest_uniform('x', -100, 100)
    y = trial.suggest_uniform('y', -1, 1)
    return x**2 + y


if __name__ == '__main__':
    # Run optimization by using `GPyOptSampler`.
    sampler = GPyOptSampler()
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=20)

    print('Best trial:')
    print('  Value: ', study.best_trial.value)
    print('  Params: ')
    for key, value in study.best_trial.params.items():
        print('    {}: {}'.format(key, value))
