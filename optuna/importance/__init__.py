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

    By default, this function uses
    :class:`~optuna.importance.PedAnovaImportanceEvaluator`.
    For details on this evaluator, please refer to the following papers:

    - `PED-ANOVA: Efficiently Quantifying Hyperparameter Importance in Arbitrary Subspaces
      <https://arxiv.org/abs/2304.10255>`__ (IJCAI 2023)
    - `Conditional PED-ANOVA: Hyperparameter Importance in Hierarchical & Dynamic Search Spaces
      <https://arxiv.org/abs/2601.20800>`__ (KDD 2026)

    When using this evaluator in your project, please cite both papers.

    With the default evaluator, :class:`~optuna.importance.PedAnovaImportanceEvaluator`,
    ``params=None`` assesses all parameters that appear in completed trials, including conditional
    parameters.
    Other evaluators assess only parameters that are present in all of the completed trials and
    therefore exclude conditional parameters.
    If ``params`` is specified, only the specified parameters are assessed.
    When using :class:`~optuna.importance.PedAnovaImportanceEvaluator`, each specified parameter
    must appear in at least one completed trial.
    When using other evaluators, at least one completed trial must contain all specified
    parameters.


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
            If :obj:`None`, :class:`~optuna.importance.PedAnovaImportanceEvaluator` assesses all
            parameters that appear in completed trials, including conditional parameters, while
            other evaluators assess parameters present in all completed trials.
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

    Example:
        .. testcode::

            import optuna


            def objective(trial: optuna.trial.Trial) -> float:
                x = trial.suggest_int("x", 0, 2)
                y = trial.suggest_float("y", -1.0, 1.0)
                z = trial.suggest_float("z", 0.0, 1.5)
                return x**2 + y**3 - z**4

            sampler = optuna.samplers.RandomSampler(seed=42)
            study = optuna.create_study(sampler=sampler)
            study.optimize(objective, n_trials=100)

            importances = optuna.importance.get_param_importances(study)

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
