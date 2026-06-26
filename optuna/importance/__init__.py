from __future__ import annotations

from typing import TYPE_CHECKING

from optuna._experimental import warn_experimental_argument
from optuna.importance._base import BaseImportanceEvaluator
from optuna.importance._fanova import FanovaImportanceEvaluator
from optuna.importance._mean_decrease_impurity import MeanDecreaseImpurityImportanceEvaluator
from optuna.importance._ped_anova import PedAnovaImportanceEvaluator


if TYPE_CHECKING:
    from collections.abc import Callable

    from optuna.study import Study
    from optuna.trial import FrozenTrial

__all__ = [
    "BaseImportanceEvaluator",
    "FanovaImportanceEvaluator",
    "MeanDecreaseImpurityImportanceEvaluator",
    "PedAnovaImportanceEvaluator",
    "get_param_importances",
]


def get_param_importances(
    study: Study,
    *,
    evaluator: BaseImportanceEvaluator | None = None,
    params: list[str] | None = None,
    target: Callable[[FrozenTrial], float] | None = None,
    normalize: bool = True,
) -> dict[str, float]:
    """Evaluate parameter importances based on completed trials in the given study.

    The parameter importances are returned as a dictionary where the keys consist of parameter
    names and their values importances.
    The importances are represented by non-negative floating point numbers, where higher values
    mean that the parameters are more important.
    The returned dictionary is ordered by its values in a descending order.
    By default, the sum of the importance values are normalized to 1.0.

    With the default evaluator, :class:`~optuna.importance.PedAnovaImportanceEvaluator`,
    ``params=None`` assesses all parameters that appear in completed trials, including conditional
    parameters.
    Other evaluators assess only parameters that are present in all of the completed trials and
    therefore exclude conditional parameters.
    If ``params`` is specified, only the specified parameters are assessed.
    When using an evaluator other than :class:`~optuna.importance.PedAnovaImportanceEvaluator`,
    at least one completed trial must contain all specified parameters.

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
            Defaults to :class:`~optuna.importance.PedAnovaImportanceEvaluator`.
        params:
            A list of names of parameters to assess.
            If :obj:`None`, all parameters that are present in all of the completed trials are
            assessed.
        target:
            A function that returns the value used to evaluate importances.
            If :obj:`None`, objective values are used for single-objective optimization.
            For multi-objective optimization, :obj:`None` is supported only by
            :class:`~optuna.importance.PedAnovaImportanceEvaluator`. When using another
            evaluator, specify ``target``, for example ``target=lambda t: t.values[0]``, to
            evaluate importances for a specific objective.
        normalize:
            A boolean option to specify whether the sum of the importance values should be
            normalized to 1.0.
            Defaults to :obj:`True`.

            .. note::
                Added in v3.0.0 as an experimental feature. The interface may change in newer
                versions without prior notice. See
                https://github.com/optuna/optuna/releases/tag/v3.0.0.

    Returns:
        A :obj:`dict` where the keys are parameter names and the values are assessed importances.

    """
    if evaluator is None:
        evaluator = PedAnovaImportanceEvaluator()

    if not isinstance(evaluator, BaseImportanceEvaluator):
        raise TypeError("Evaluator must be a subclass of BaseImportanceEvaluator.")

    res = evaluator.evaluate(study, params=params, target=target)
    if normalize:
        s = sum(res.values())
        if s == 0.0:
            n_params = len(res)
            return dict((param, 1.0 / n_params) for param in res.keys())
        else:
            return dict((param, value / s) for (param, value) in res.items())
    else:
        warn_experimental_argument("normalize")
        return res
