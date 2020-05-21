from typing import Dict
from typing import List
from typing import Optional

from optuna._experimental import experimental
from optuna.importance._base import BaseImportanceEvaluator
from optuna.importance._fanova import FanovaImportanceEvaluator  # NOQA
from optuna.importance._mean_decrease_impurity import MeanDecreaseImpurityImportanceEvaluator
from optuna.study import Study


@experimental("1.3.0")
def get_param_importances(
    study: Study, evaluator: BaseImportanceEvaluator = None, params: Optional[List[str]] = None
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

    Args:
        study:
            An optimized study.
        evaluator:
            An importance evaluator object that specifies which algorithm to base the importance
            assessment on.
            Defaults to
            :class:`~optuna.importance._mean_decrease_impurity.MeanDecreaseImpurityImportanceEvaluator`.
        params:
            A list of names of parameters to assess.
            If :obj:`None`, all parameters that are present in all of the completed trials are
            assessed.

    Returns:
        An :class:`collections.OrderedDict` where the keys are parameter names and the values are
        assessed importances.
    """
    if evaluator is None:
        evaluator = MeanDecreaseImpurityImportanceEvaluator()

    if not isinstance(evaluator, BaseImportanceEvaluator):
        raise TypeError("Evaluator must be a subclass of BaseImportanceEvaluator.")

    return evaluator.evaluate(study, params=params)
