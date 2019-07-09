import numpy as np

import optuna
from optuna import distributions
from optuna.samplers import BaseSampler
from optuna import structs

if optuna.types.TYPE_CHECKING:
    from typing import Any  # NOQA
    from typing import Dict  # NOQA
    from typing import Optional  # NOQA

    from optuna.distributions import BaseDistribution  # NOQA
    from optuna.structs import FrozenTrial  # NOQA
    from optuna.study import InTrialStudy  # NOQA

COOLDOWN_FACTOR = 0.9
NEIGHBOR_RANGE_FACTOR = 0.1


class SimulatedAnnealingSampler(BaseSampler):
    def __init__(self, temperature=100, seed=None):
        # type: (int, Optional[int]) -> None

        self._rng = np.random.RandomState(seed)
        self._independent_sampler = optuna.samplers.RandomSampler(seed=seed)
        self._temperature = temperature
        self._current_params = {}  # type: Dict[str, Any]

    def infer_relative_search_space(self, study, trial):
        # type: (InTrialStudy, FrozenTrial) -> Dict[str, BaseDistribution]

        return optuna.samplers.product_search_space(study)

    def sample_relative(self, study, trial, search_space):
        # type: (InTrialStudy, FrozenTrial, Dict[str, BaseDistribution]) -> Dict[str, Any]

        if search_space == {}:
            return {}

        prev_trial = self._last_complete_trial(study)
        if self._rng.uniform(0, 1) <= self._probability(study, prev_trial.value, study.best_value):
            self._current_params = prev_trial.params

        params = self._sample_neighbor_params(search_space)
        self._temperature *= COOLDOWN_FACTOR

        return params

    def _sample_neighbor_params(self, search_space):
        # type: (Dict[str, BaseDistribution]) -> Dict[str, Any]

        params = {}
        for param_name, param_distribution in search_space.items():
            if isinstance(param_distribution, distributions.UniformDistribution):
                current_value = self._current_params[param_name]
                width = (param_distribution.high - param_distribution.low) * NEIGHBOR_RANGE_FACTOR
                neighbor_low = max(current_value - width, param_distribution.low)
                neighbor_high = min(current_value + width, param_distribution.high)
                params[param_name] = self._rng.uniform(neighbor_low, neighbor_high)
            else:
                raise NotImplementedError(
                    'Unsupported distribution {}.'.format(param_distribution))

        return params

    def _probability(self, study, prev_value, best_value):
        # type: (InTrialStudy, float, float) -> float

        if study.direction == structs.StudyDirection.MINIMIZE and prev_value <= best_value:
            return 1.0
        elif study.direction == structs.StudyDirection.MAXIMIZE and prev_value >= best_value:
            return 1.0
        else:
            return np.exp(-abs(best_value - prev_value) / self._temperature)

    def _last_complete_trial(self, study):
        # type: (InTrialStudy) -> Optional[FrozenTrial]

        complete_trials = [t for t in study.trials if t.state == structs.TrialState.COMPLETE]
        if len(complete_trials) == 0:
            return None
        else:
            return complete_trials[-1]

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
    # Run optimization by using `SimulatedAnnealingSampler`.
    sampler = SimulatedAnnealingSampler()
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=100)

    print('Best trial:')
    print('  Value: ', study.best_trial.value)
    print('  Params: ')
    for key, value in study.best_trial.params.items():
        print('    {}: {}'.format(key, value))
