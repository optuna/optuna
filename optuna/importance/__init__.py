from typing import Dict
from typing import List
from typing import Optional

from optuna import _experimental
from optuna.importance._base import BaseImportanceEvaluator
from optuna.importance._fanova import FanovaImportanceEvaluator  # NOQA
from optuna.importance._sklearn import PermutationImportanceEvaluator  # NOQA
from optuna.importance._sklearn import RandomForestFeatureImportanceEvaluator
from optuna.structs import TrialState
from optuna.study import Study


@_experimental.experimental("1.2.0")
def get_param_importances(
        study: Study, evaluator: BaseImportanceEvaluator = None,
        params: Optional[List[str]] = None) -> Dict[str, float]:
    """Compute parameter importances based on an optimized study.

    Args:
        study:
            An optimized study.
        evaluator:
            Importance evaluator object. Defaults to
            :class:`~optuna.importance._sklearn.RandomForestFeatureImportanceEvaluator`.
        params:
            Names of the parameters to evaluate.

    Returns:
        A dictionary where keys are parameter names and values are their importances in floats.
    """

    # TODO(hvy): Support specifying parameters.
    if params is not None:
        raise NotImplementedError

    if not any(t for t in study.trials if t.state == TrialState.COMPLETE):
        raise ValueError('Cannot evaluate parameter importances without completed trials.')

    if evaluator is None:
        evaluator = RandomForestFeatureImportanceEvaluator()

    if not isinstance(evaluator, BaseImportanceEvaluator):
        raise TypeError('Evaluator must be a subclass of BaseImportanceEvaluator.')

    return evaluator.get_param_importances(study)
