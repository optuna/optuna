import abc
from collections import OrderedDict
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np

from optuna._experimental import experimental
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.samplers import intersection_search_space
from optuna.study import Study
from optuna.trial import TrialState


@experimental("1.3.0")
class BaseImportanceEvaluator(object, metaclass=abc.ABCMeta):
    """Abstract parameter importance evaluator.

    """

    @abc.abstractmethod
    def evaluate(self, study: Study, params: Optional[List[str]]) -> Dict[str, float]:
        """Evaluate parameter importances based on completed trials in the given study.

        .. note::

            This method is not meant to be called by library users.

        .. seealso::

            Please refer to :func:`~optuna.importance.get_param_importances` for how a concrete
            evaluator should implement this method.

        Args:
            study:
                An optimized study.
            params:
                A list of names of parameters to assess.
                If :obj:`None`, all parameters that are present in all of the completed trials are
                assessed.

        Returns:
            An :class:`collections.OrderedDict` where the keys are parameter names and the values
            are assessed importances.
        """
        # TODO(hvy): Reconsider the interface as logic might violate DRY among multiple evaluators.
        raise NotImplementedError


def _get_distributions(study: Study, params: Optional[List[str]]) -> Dict[str, BaseDistribution]:
    _check_evaluate_args(study, params)

    if params is None:
        return intersection_search_space(study, ordered_dict=True)

    # New temporary required to pass mypy. Seems like a bug.
    params_not_none = params
    assert params_not_none is not None

    # Compute the search space based on the subset of trials containing all parameters.
    distributions = None
    for trial in study.trials:
        if trial.state != TrialState.COMPLETE:
            continue

        trial_distributions = trial.distributions
        if not all(name in trial_distributions for name in params_not_none):
            continue

        if distributions is None:
            distributions = dict(
                filter(
                    lambda name_and_distribution: name_and_distribution[0] in params_not_none,
                    trial_distributions.items(),
                )
            )
            continue

        if any(
            trial_distributions[name] != distribution
            for name, distribution in distributions.items()
        ):
            raise ValueError(
                "Parameters importances cannot be assessed with dynamic search spaces if "
                "parameters are specified. Specified parameters: {}.".format(params)
            )

    assert distributions is not None  # Required to pass mypy.
    distributions = OrderedDict(
        sorted(distributions.items(), key=lambda name_and_distribution: name_and_distribution[0])
    )
    return distributions


def _get_study_data(
    study: Study, distributions: Dict[str, BaseDistribution]
) -> Tuple[np.ndarray, np.ndarray]:
    trials = []
    for trial in study.trials:
        if trial.state != TrialState.COMPLETE:
            continue
        if any(name not in trial.params for name in distributions.keys()):
            continue
        trials.append(trial)

    n_trials = len(trials)
    n_params = len(distributions)

    params = np.empty((n_trials, n_params), dtype=np.float64)
    values = np.empty((n_trials,), dtype=np.float64)

    for i, trial in enumerate(trials):
        trial_params = trial.params
        for j, (name, distribution) in enumerate(distributions.items()):
            param = trial_params[name]
            if isinstance(distribution, CategoricalDistribution):
                param = distribution.to_internal_repr(param)
            params[i, j] = param
        values[i] = trial.value

    return params, values


def _check_evaluate_args(study: Study, params: Optional[List[str]]) -> None:
    completed_trials = list(filter(lambda t: t.state == TrialState.COMPLETE, study.trials))
    if len(completed_trials) == 0:
        raise ValueError("Cannot evaluate parameter importances without completed trials.")
    if len(completed_trials) == 1:
        raise ValueError("Cannot evaluate parameter importances with only a single trial.")

    if params is not None:
        if not isinstance(params, (list, tuple)):
            raise TypeError(
                "Parameters must be specified as a list. Actual parameters: {}.".format(params)
            )
        if any(not isinstance(p, str) for p in params):
            raise TypeError(
                "Parameters must be specified by their names with strings. Actual parameters: "
                "{}.".format(params)
            )

        if len(params) > 0:
            at_least_one_trial = False
            for trial in completed_trials:
                if all(p in trial.distributions for p in params):
                    at_least_one_trial = True
                    break
            if not at_least_one_trial:
                raise ValueError(
                    "Study must contain completed trials with all specified parameters. "
                    "Specified parameters: {}.".format(params)
                )
