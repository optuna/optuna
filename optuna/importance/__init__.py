from typing import Dict
from typing import List
from typing import Optional

from optuna._experimental import experimental
from optuna.importance import _fanova
from optuna.study import Study


@experimental("1.2.0")
def get_param_importance(study: Study, params: Optional[List[str]] = None) -> Dict[str, float]:
    """Compute parameter importances based on an optimized study.

    Args:
        study:
            An optimized study.
        params:
            Names of the parameters to evaluate.

    Returns:
        A dictionary where keys are parameter names and values are their importances in floats.
    """

    evaluator = _fanova._Fanova(study)
    return evaluator.get_param_importance()
