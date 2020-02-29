import abc
from typing import Dict
from typing import Tuple

import numpy as np

from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.samplers import intersection_search_space
from optuna.structs import TrialState
from optuna.study import Study


class BaseImportanceEvaluator(object, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def get_param_importances(self, study: Study) -> Dict[str, float]:
        raise NotImplementedError


def _get_search_space(study: Study) -> Dict[str, BaseDistribution]:
    return intersection_search_space(study, ordered_dict=True)


def _get_trial_data(
        study: Study, search_space: Dict[str, BaseDistribution]) -> Tuple[np.ndarray, np.ndarray]:
    trials = [t for t in study.trials if t.state == TrialState.COMPLETE]

    n_trials = len(trials)
    n_params = len(search_space)

    assert n_trials > 0
    assert n_params > 0

    params = np.empty((n_trials, n_params), dtype=np.float64)
    values = np.empty((n_trials,), dtype=np.float64)

    for i, trial in enumerate(trials):
        trial_params = trial.params
        for j, (name, distribution) in enumerate(search_space.items()):
            param = trial_params[name]
            if isinstance(distribution, CategoricalDistribution):
                param = distribution.to_internal_repr(param)
            params[i, j] = param
        values[i] = trial.value

    return params, values
