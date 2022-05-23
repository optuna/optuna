from collections import OrderedDict
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

import numpy

from optuna._imports import try_import
from optuna.importance._base import BaseImportanceEvaluator
from optuna.importance._utils import _gather_importance_values
from optuna.importance._utils import _gather_study_info
from optuna.importance._utils import _StudyInfo
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
        self._trans_params = numpy.empty(0)
        self._trans_values = numpy.empty(0)
        self._param_names: List[str] = list()

    def evaluate(
        self,
        study: Study,
        params: Optional[List[str]] = None,
        *,
        target: Optional[Callable[[FrozenTrial], float]] = None,
    ) -> Dict[str, float]:

        info: _StudyInfo
        info = _gather_study_info(study, params=params, target=target)

        n_params = len(info.non_single_distributions)

        importances = OrderedDict()
        if n_params > 0:
            assert info.trans is not None

            self._forest.fit(info.trans_params, info.trans_values)

            feature_importances = self._forest.feature_importances_
            feature_importances_reduced = numpy.zeros(n_params)
            numpy.add.at(
                feature_importances_reduced,
                info.trans.encoded_column_to_column,
                feature_importances,
            )

            param_names = list(info.non_single_distributions.keys())
            for i in feature_importances_reduced.argsort()[::-1]:
                importances[param_names[i]] = feature_importances_reduced[i].item()

        return _gather_importance_values(importances, info.single_distributions)
