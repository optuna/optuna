from collections import OrderedDict
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

from optuna.importance._base import BaseImportanceEvaluator
from optuna.importance._fanova import FanovaImportanceEvaluator
from optuna.importance._mean_decrease_impurity import MeanDecreaseImpurityImportanceEvaluator
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


__all__ = [
    "BaseImportanceEvaluator",
    "FanovaImportanceEvaluator",
    "MeanDecreaseImpurityImportanceEvaluator",
    "get_param_importances",
]


def _check_evaluate_args(completed_trials: List[FrozenTrial], params: Optional[List[str]]) -> None:
    if len(completed_trials) == 0:
        raise ValueError("Cannot evaluate parameter importances without completed trials.")
    if len(completed_trials) == 1:
        raise ValueError("Cannot evaluate parameter importances with only a single trial.")

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


def get_param_importances(
    study: Study,
    *,
    evaluator: Optional[BaseImportanceEvaluator] = None,
    params: Optional[List[str]] = None,
    target: Optional[Callable[[FrozenTrial], float]] = None,
) -> Dict[str, float]:
    """Evaluate parameter importances based on completed trials in the given study.

    The parameter importances are returned as a dictionary where the keys consist of parameter
    names and their values importances.
    The importances are represented by floating point numbers that sum to 1.0 over the entire
    dictionary.
    The higher the value, the more important.
    The returned dictionary is of type :class:`collections.OrderedDict` and is ordered by
    its values in a descending order.

    If ``params`` is :obj:`None`, all parameter that are present in all of the completed trials are
    assessed.
    This implies that conditional parameters will be excluded from the evaluation.
    To assess the importances of conditional parameters, a :obj:`list` of parameter names can be
    specified via ``params``.
    If specified, only completed trials that contain all of the parameters will be considered.
    If no such trials are found, an error will be raised.

    If the given study does not contain completed trials, an error will be raised.

    .. note::

        If ``params`` is specified as an empty list, an empty dictionary is returned.

    .. seealso::

        See :func:`~optuna.visualization.plot_param_importances` to plot importances.

    Args:
        study:
            An optimized study.
        evaluator:
            An importance evaluator object that specifies which algorithm to base the importance
            assessment on.
            Defaults to
            :class:`~optuna.importance.FanovaImportanceEvaluator`.
        params:
            A list of names of parameters to assess.
            If :obj:`None`, all parameters that are present in all of the completed trials are
            assessed.
        target:
            A function to specify the value to evaluate importances.
            If it is :obj:`None` and ``study`` is being used for single-objective optimization,
            the objective values are used. Can also be used for other trial attributes, such as
            the duration, like ``target=lambda t: t.duration.total_seconds()``.

            .. note::
                Specify this argument if ``study`` is being used for multi-objective
                optimization. For example, to get the hyperparameter importance of the first
                objective, use ``target=lambda t: t.values[0]`` for the target parameter.

    Returns:
        An :class:`collections.OrderedDict` where the keys are parameter names and the values are
        assessed importances. The importances will be sorted in a descending order.

    """

    completed_trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
    if len(completed_trials) == 0:
        raise ValueError("The study does not contain completed trials.")

    if evaluator is None:
        evaluator = FanovaImportanceEvaluator()

    if not isinstance(evaluator, BaseImportanceEvaluator):
        raise TypeError("Evaluator must be a subclass of BaseImportanceEvaluator.")

    if target is None:
        if study._is_multi_objective():
            raise ValueError(
                "If the `study` is being used for multi-objective optimization, "
                "please specify the `target`. For example, use "
                "`target=lambda t: t.values[0]` for the first objective value."
            )

        def default_target(trial: FrozenTrial) -> float:
            val: Optional[float] = trial.value
            assert val is not None
            return val

        target = default_target

    if params is None:
        # Find intersection spaces of all parameters.
        # We don't use `intersection_search_space` here because
        # we don't want to call `study.get_trials` multiple times.
        dists: Any
        dists = completed_trials[0].distributions.items()
        for trial in completed_trials[1:]:
            # We use lambda to capture the value of `trial.distributions`.
            dists = (
                lambda d: (
                    (param, dist) for param, dist in dists if param in d and dist == d[param]
                )
            )(trial.distributions)
        params = list(param for param, dist in dists)

    assert params is not None

    _check_evaluate_args(completed_trials, params)

    if len(params) == 0:
        return OrderedDict()

    importances = evaluator.evaluate(completed_trials, params=params, target=target)

    # Sort the importances in descending order.
    return OrderedDict(
        reversed(
            sorted(importances.items(), key=lambda name_and_importance: name_and_importance[1])
        )
    )
