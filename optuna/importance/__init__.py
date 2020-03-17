from typing import Dict
from typing import List
from typing import Optional

from optuna._experimental import experimental
from optuna.importance._base import BaseImportanceEvaluator
from optuna.importance._fanova import FanovaImportanceEvaluator
from optuna.study import Study


@experimental("1.3.0")
def get_param_importances(
    study: Study, evaluator: BaseImportanceEvaluator = None, params: Optional[List[str]] = None
) -> Dict[str, float]:
    """Compute parameter importances based on an optimized study.

    Args:
        study:
            An optimized study.
        evaluator:
            Importance evaluator object. Defaults to
            :class:`~optuna.importance._sklearn.FanovaImportanceEvaluator`.
        params:
            Names of the parameters to evaluate.

    Returns:
        A dictionary where keys are parameter names and values are their importances in floats.
    """
    if evaluator is None:
        evaluator = FanovaImportanceEvaluator()

    if not isinstance(evaluator, BaseImportanceEvaluator):
        raise TypeError("Evaluator must be a subclass of BaseImportanceEvaluator.")

    return evaluator.evaluate(study, params=params)
