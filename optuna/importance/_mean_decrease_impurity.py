from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

import numpy

from optuna._imports import try_import
from optuna._transform import _SearchSpaceTransform
from optuna.importance._base import _feature_importances_to_param_importances
from optuna.importance._base import _get_distributions
from optuna.importance._base import _get_filtered_trials
from optuna.importance._base import _get_target_values
from optuna.importance._base import _get_trans_params
from optuna.importance._base import _param_importances_to_dict
from optuna.importance._base import BaseImportanceEvaluator
from optuna.study import Study
from optuna.trial import FrozenTrial


with try_import() as _imports:
    from sklearn.ensemble import RandomForestRegressor


class MeanDecreaseImpurityImportanceEvaluator(BaseImportanceEvaluator):
    """Mean Decrease Impurity (MDI) parameter importance evaluator.

    This evaluator fits a random forest that predicts objective values given hyperparameter
    configurations. Feature importances are then computed using MDI.

    .. note::

        This evaluator requires the `sklean <https://scikit-learn.org/stable/>`_ Python package and
        is based on `sklearn.ensemble.RandomForestClassifier.feature_importances_
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

    def evaluate(
        self,
        study: Study,
        params: List[str],
        target: Callable[[FrozenTrial], float],
    ) -> Dict[str, float]:

        distributions = _get_distributions(study, params=params)

        if len(distributions) == 0:
            return {}

        trials: List[FrozenTrial] = _get_filtered_trials(study, params=params, target=target)
        trans = _SearchSpaceTransform(distributions, transform_log=False, transform_step=False)
        trans_params: numpy.ndarray = _get_trans_params(trials, trans)
        values: numpy.ndarray = _get_target_values(trials, target)

        forest = self._forest
        forest.fit(X=trans_params, y=values)
        feature_importances = forest.feature_importances_
        param_importances = _feature_importances_to_param_importances(feature_importances, trans)

        return _param_importances_to_dict(params, param_importances)
