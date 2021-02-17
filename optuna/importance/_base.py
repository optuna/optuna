import abc
from collections import OrderedDict
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import TYPE_CHECKING

from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
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


def _get_distributions(study: "Study", params: Optional[List[str]]) -> Dict[str, BaseDistribution]:
    _check_evaluate_args(study, params)
    distributions = None
    for trial in study.trials:
        if trial.state != TrialState.COMPLETE:
            continue

        if params is not None:
            trial_distributions = trial.distributions
            if not all(name in trial_distributions for name in params):
                continue

        if distributions is None:
            distributions = {}
            for param_name, param_distribution in trial.distributions.items():
                if (params is None) or (params is not None and param_name in params):
                    distributions[param_name] = param_distribution
            continue

        delete_list = []
        for param_name, param_distribution in distributions.items():
            if param_name not in trial.distributions:
                delete_list.append(param_name)
            else:
                trial_param_distribution = trial.distributions[param_name]
                if isinstance(param_distribution, CategoricalDistribution):
                    continue
                if not isinstance(
                    trial_param_distribution, CategoricalDistribution
                ) and not isinstance(param_distribution, CategoricalDistribution):
                    param_distribution.low = min(
                        trial_param_distribution.low, param_distribution.low
                    )
                    param_distribution.high = max(
                        trial_param_distribution.high, param_distribution.high
                    )
                else:
                    delete_list.append(param_name)

        for param_name in delete_list:
            del distributions[param_name]

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
