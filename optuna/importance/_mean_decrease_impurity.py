from collections import OrderedDict
from typing import Dict
from typing import List
from typing import Optional

import numpy

from optuna._imports import try_import
from optuna._transform import _Transform
from optuna.importance._base import _get_distributions
from optuna.importance._base import BaseImportanceEvaluator
from optuna.study import Study
from optuna.trial import TrialState


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

    def evaluate(self, study: Study, params: Optional[List[str]] = None) -> Dict[str, float]:
        distributions = _get_distributions(study, params)
        if len(distributions) == 0:
            return OrderedDict()

        trials = []
        for trial in study.trials:
            if trial.state != TrialState.COMPLETE:
                continue
            if any(name not in trial.params for name in distributions.keys()):
                continue
            trials.append(trial)

        trans = _Transform(trials, distributions, transform_log=False)
        trans_params = trans.params
        trans_values = trans.values
        encoded_cols_to_cols = trans._encoder.encoded_cols_to_cols
        assert encoded_cols_to_cols is not None

        if trans_params.size == 0:  # `params` were given but as an empty list.
            return OrderedDict()

        forest = self._forest
        forest.fit(trans_params, trans_values)
        feature_importances = forest.feature_importances_
        feature_importances_reduced = numpy.zeros(len(distributions))
        numpy.add.at(feature_importances_reduced, encoded_cols_to_cols, feature_importances)

        param_importances = OrderedDict()
        param_names = list(distributions.keys())
        for i in feature_importances_reduced.argsort()[::-1]:
            param_importances[param_names[i]] = feature_importances_reduced[i].item()

        return param_importances
