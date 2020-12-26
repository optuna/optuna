import abc
from collections import OrderedDict
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import TYPE_CHECKING

from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import DiscreteUniformDistribution
from optuna.distributions import IntLogUniformDistribution
from optuna.distributions import IntUniformDistribution
from optuna.distributions import LogUniformDistribution
from optuna.distributions import UniformDistribution
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


if TYPE_CHECKING:
    from optuna.study import Study


class BaseImportanceEvaluator(object, metaclass=abc.ABCMeta):
    """Abstract parameter importance evaluator."""

    @abc.abstractmethod
    def evaluate(
        self,
        study: "Study",
        params: Optional[List[str]] = None,
        *,
        target: Optional[Callable[[FrozenTrial], float]] = None,
    ) -> Dict[str, float]:
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
            target:
                A function to specify the value to evaluate importances.
                If it is :obj:`None` and ``study`` is being used for single-objective optimization,
                the objective values are used.

                .. note::
                    Specify this argument if ``study`` is being used for multi-objective
                    optimization.

        Returns:
            An :class:`collections.OrderedDict` where the keys are parameter names and the values
            are assessed importances.

        Raises:
            :exc:`ValueError`:
                If ``target`` is :obj:`None` and ``study`` is being used for multi-objective
                optimization.
        """
        # TODO(hvy): Reconsider the interface as logic might violate DRY among multiple evaluators.
        raise NotImplementedError


def _get_info_distribution(distribution: BaseDistribution) -> Dict[str, Any]:
    info_distribution = {}
    if isinstance(distribution, CategoricalDistribution):
        info_distribution = {
            "is_categorical": True,
            "distribution": distribution,
        }
    elif isinstance(
        distribution,
        (
            DiscreteUniformDistribution,
            IntLogUniformDistribution,
            IntUniformDistribution,
            LogUniformDistribution,
            UniformDistribution,
        ),
    ):
        info_distribution = {
            "is_categorical": False,
            "low": distribution.low,
            "high": distribution.high,
        }

    return info_distribution


def _get_distributions(study: "Study", params: Optional[List[str]]) -> Dict[str, BaseDistribution]:
    _check_evaluate_args(study, params)
    info_distributions = None
    for trial in reversed(study._storage.get_all_trials(study._study_id, deepcopy=False)):
        if trial.state != TrialState.COMPLETE:
            continue

        if params is not None:
            trial_distributions = trial.distributions
            if not all(name in trial_distributions for name in params):
                continue

        if info_distributions is None:
            info_distributions = {}
            for param_name, param_distribution in trial.distributions.items():
                if (params is None) or (params is not None and param_name in params):
                    info_distributions[param_name] = _get_info_distribution(param_distribution)
            continue

        delete_list = []
        for param_name, info_distribution in info_distributions.items():
            if param_name not in trial.distributions:
                delete_list.append(param_name)
            else:
                trial_info_distribution = _get_info_distribution(trial.distributions[param_name])
                if (
                    trial_info_distribution["is_categorical"]
                    != info_distribution["is_categorical"]
                ):
                    delete_list.append(param_name)
                elif trial_info_distribution["is_categorical"] is True:
                    # CategoricalDistribution does not support dynamic value space.
                    if trial_info_distribution["distribution"] != trial.distributions[param_name]:
                        delete_list.append(param_name)
                else:
                    info_distribution["low"] = min(
                        trial_info_distribution["low"], info_distribution["low"]
                    )
                    info_distribution["high"] = max(
                        trial_info_distribution["high"], info_distribution["high"]
                    )

        if params is not None and len(delete_list) > 0:
            raise ValueError(
                "Parameters importances cannot be assessed with dynamic search spaces if "
                "parameters are specified. Specified parameters: {}.".format(params)
            )
        else:
            for param_name in delete_list:
                del info_distributions[param_name]

    distributions = {}
    for param, info_distribution in info_distributions.items():
        if info_distribution["is_categorical"] is True:
            distributions[param] = info_distribution["distribution"]
        else:
            distributions[param] = UniformDistribution(
                info_distribution["low"], info_distribution["high"]
            )

    assert distributions is not None  # Required to pass mypy.
    distributions = OrderedDict(
        sorted(distributions.items(), key=lambda name_and_distribution: name_and_distribution[0])
    )
    return distributions


def _check_evaluate_args(study: "Study", params: Optional[List[str]]) -> None:
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
