import abc
from collections import OrderedDict
from typing import Callable
from typing import Collection
from typing import Dict
from typing import List
from typing import Union

import numpy

from optuna._transform import _SearchSpaceTransform
from optuna.distributions import BaseDistribution
from optuna.trial import FrozenTrial


class BaseImportanceEvaluator(object, metaclass=abc.ABCMeta):
    """Abstract parameter importance evaluator."""

    @abc.abstractmethod
    def evaluate(
        self,
        completed_trials: List[FrozenTrial],
        params: List[str],
        *,
        target: Callable[[FrozenTrial], float],
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
                Can also be used for other trial attributes, such as
                the duration, like ``target=lambda t: t.duration.total_seconds()``.

        Returns:
            An :class:`collections.OrderedDict` where the keys are parameter names and the values
            are assessed importances.

        """
        # TODO(hvy): Reconsider the interface as logic might violate DRY among multiple evaluators.
        raise NotImplementedError


def _get_distributions(
    completed_trials: List[FrozenTrial], params: List[str]
) -> Dict[str, BaseDistribution]:
    filtered_trials = (trial for trial in completed_trials if set(params) <= set(trial.params))

    first_trial = next(filtered_trials)
    distributions = {name: first_trial.distributions[name] for name in params}

    # Check that all remaining trials have the same distributions.
    if any(
        trial.distributions[name] != distribution
        for trial in filtered_trials
        for name, distribution in distributions.items()
    ):
        raise ValueError(
            "Parameters importances cannot be assessed with dynamic search spaces if "
            "parameters are specified. Specified parameters: {}.".format(params)
        )

    return OrderedDict(
        sorted(distributions.items(), key=lambda name_and_distribution: name_and_distribution[0])
    )


def _get_filtered_trials(
    completed_trials: List[FrozenTrial],
    params: Collection[str],
    target: Callable[[FrozenTrial], float],
) -> List[FrozenTrial]:
    return [
        trial
        for trial in completed_trials
        if set(params) <= set(trial.params) and numpy.isfinite(target(trial))
    ]


def _param_importances_to_dict(
    params: Collection[str], param_importances: Union[numpy.ndarray, float]
) -> Dict[str, float]:
    return {
        name: value
        for name, value in zip(params, numpy.broadcast_to(param_importances, (len(params),)))
    }


def _get_trans_params(trials: List[FrozenTrial], trans: _SearchSpaceTransform) -> numpy.ndarray:
    return numpy.array([trans.transform(trial.params) for trial in trials])


def _get_target_values(
    trials: List[FrozenTrial], target: Callable[[FrozenTrial], float]
) -> numpy.ndarray:
    return numpy.array([target(trial) for trial in trials])
