from __future__ import annotations

from typing import Callable
from typing import List
from typing import Optional

import numpy

from optuna._imports import try_import
from optuna._transform import _SearchSpaceTransform
from optuna.importance._base import _before_evaluate
from optuna.importance._base import _get_trans_params
from optuna.importance._base import _param_importances_to_dict
from optuna.importance._base import _sort_dict_by_importance
from optuna.importance._base import BaseImportanceEvaluator
from optuna.study import Study
from optuna.trial import FrozenTrial


with try_import() as _imports:
    from sklearn.ensemble import RandomForestRegressor


class MeanDecreaseImpurityImportanceEvaluator(BaseImportanceEvaluator):
    """Mean Decrease Impurity (MDI) parameter importance evaluator.

    This evaluator fits fits a random forest regression model that predicts the objective values
    of :class:`~optuna.trial.TrialState.COMPLETE` trials given their parameter configurations.
    Feature importances are then computed using MDI.

    .. note::

        This evaluator requires the `sklearn <https://scikit-learn.org/stable/>`_ Python package
        and is based on `sklearn.ensemble.RandomForestClassifier.feature_importances_
        <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier.feature_importances_>`_.

    Args:
        n_trees:
            Number of trees in the random forest.
        max_depth:
            The maximum depth of each tree in the random forest.
        seed:
            Seed for the random forest.
    """

    def __init__(
        self, *, n_trees: int = 64, max_depth: int = 64, seed: Optional[int] = None
    ) -> None:
        _imports.check()

        self._forest = RandomForestRegressor(
            n_estimators=n_trees,
            max_depth=max_depth,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=seed,
        )
        self._trans_params = numpy.empty(0)
        self._trans_values = numpy.empty(0)
        self._param_names: List[str] = list()

    def evaluate(
        self,
        study: Study,
        params: list[str] | None = None,
        *,
        target: Callable[[FrozenTrial], float] | None = None,
    ) -> dict[str, float]:
        params, distributions, trials, target_values = _before_evaluate(study, params, target)
        if len(params) == 0:
            return {}

        trans = _SearchSpaceTransform(distributions, transform_log=False, transform_step=False)
        trans_params: numpy.ndarray = _get_trans_params(trials, trans)

        forest = self._forest
        forest.fit(X=trans_params, y=target_values)
        feature_importances = forest.feature_importances_

        # Untransform feature importances to param importances
        # by adding up relevant feature importances.
        param_importances = numpy.zeros(len(params))
        numpy.add.at(param_importances, trans.encoded_column_to_column, feature_importances)
        return _sort_dict_by_importance(_param_importances_to_dict(params, param_importances))
