from collections import OrderedDict
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

import numpy

from optuna._transform import _SearchSpaceTransform
from optuna.distributions import BaseDistribution
from optuna.importance._base import _get_distributions
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
from optuna.visualization._utils import _filter_nonfinite


class _StudyInfo:
    def __init__(
        self,
        non_single_distributions: Dict[str, BaseDistribution],
        single_distributions: Dict[str, BaseDistribution],
        trans: Optional[_SearchSpaceTransform],  # None if there is no non_single_distributions
        trans_params: numpy.ndarray,
        trans_values: numpy.ndarray,
    ):
        self.non_single_distributions = non_single_distributions
        self.single_distributions = single_distributions
        self.trans = trans
        self.trans_params = trans_params
        self.trans_values = trans_values


def _gather_study_info(
    study: Study,
    *,
    params: Optional[List[str]] = None,
    target: Optional[Callable[[FrozenTrial], float]] = None,
) -> _StudyInfo:

    if target is None and study._is_multi_objective():
        raise ValueError(
            "If the `study` is being used for multi-objective optimization, "
            "please specify the `target`. For example, use "
            "`target=lambda t: t.values[0]` for the first objective value."
        )

    distributions = _get_distributions(study, params)

    # Imporance values for parameter distributions with a single value is always set to 0.
    single_distributions = {name: dist for name, dist in distributions.items() if dist.single()}
    non_single_distributions = {
        name: dist for name, dist in distributions.items() if not dist.single()
    }

    if len(non_single_distributions) == 0:
        return _StudyInfo(
            non_single_distributions=non_single_distributions,
            single_distributions=single_distributions,
            trans=None,
            trans_params=numpy.empty((0, 0), dtype=numpy.float64),
            trans_values=numpy.empty(0, dtype=numpy.float64),
        )

    trials = []
    for trial in _filter_nonfinite(
        study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)), target=target
    ):
        if any(name not in trial.params for name in non_single_distributions.keys()):
            continue
        trials.append(trial)

    trans = _SearchSpaceTransform(
        non_single_distributions, transform_log=False, transform_step=False
    )

    n_trials = len(trials)
    trans_params = numpy.empty((n_trials, trans.bounds.shape[0]), dtype=numpy.float64)
    trans_values = numpy.empty(n_trials, dtype=numpy.float64)

    for trial_idx, trial in enumerate(trials):
        trans_params[trial_idx] = trans.transform(trial.params)
        trans_values[trial_idx] = trial.value if target is None else target(trial)

    return _StudyInfo(
        non_single_distributions=non_single_distributions,
        single_distributions=single_distributions,
        trans=trans,
        trans_params=trans_params,
        trans_values=trans_values,
    )


def _gather_importance_values(
    non_single_importances: Dict[str, float],
    single_distributions: Dict[str, BaseDistribution],
    single_importance_value: float = 0.0,
) -> Dict[str, float]:
    single_importances = {name: single_importance_value for name in single_distributions.keys()}
    importances = {**non_single_importances, **single_importances}
    sorted_importances = OrderedDict(
        reversed(
            sorted(importances.items(), key=lambda name_and_importance: name_and_importance[1])
        )
    )
    return sorted_importances
