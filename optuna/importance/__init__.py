from typing import Dict
from typing import List
from typing import Optional
from typing import Type

from optuna._experimental import experimental
from optuna.importance import _base
from optuna.importance import _fanova
from optuna.importance import _sklearn
from optuna.structs import TrialState
from optuna.study import Study


@experimental("1.2.0")
def get_param_importance(
        study: Study, evaluator: str = 'random_forest_feature_importance',
        params: Optional[List[str]] = None) -> Dict[str, float]:
    """Compute parameter importances based on an optimized study.

    Args:
        study:
            An optimized study.
        params:
            Names of the parameters to evaluate.

    Returns:
        A dictionary where keys are parameter names and values are their importances in floats.
    """

    # TODO(hvy): Support specifying parameters.
    if params is not None:
        raise NotImplementedError

    trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if len(trials) == 0:
        raise ValueError('Cannot evaluate parameter importance without complete d trials.')

    evaluator_cls = None  # type: Optional[Type[_base._BaseImportanceEvaluator]]
    if evaluator == 'random_forest_feature_importance':
        evaluator_cls = _sklearn._RandomForestFeatureImportance
    elif evaluator == 'permutation_importance':
        evaluator_cls = _sklearn._PermutationImportance
    elif evaluator == 'fanova':
        evaluator_cls = _fanova._Fanova
    else:
        # TODO(hvy): Make message more verbose.
        raise ValueError('Unsupported evaluator {}.'.format(evaluator))

    assert evaluator_cls is not None
    evaltr = evaluator_cls()

    return evaltr.get_param_importance(study)
