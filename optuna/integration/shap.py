from collections import OrderedDict
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

import numpy as np

from optuna._experimental import experimental_class
from optuna._imports import try_import
from optuna._transform import _SearchSpaceTransform
from optuna.importance._base import _get_distributions
from optuna.importance._base import _get_filtered_trials
from optuna.importance._base import _get_target_values
from optuna.importance._base import _get_trans_params
from optuna.importance._base import _param_importances_to_dict
from optuna.importance._base import _sort_dict_by_importance
from optuna.importance._base import BaseImportanceEvaluator
from optuna.study import Study
from optuna.trial import FrozenTrial


with try_import() as _imports:
    from shap import TreeExplainer
    from sklearn.ensemble import RandomForestRegressor


@experimental_class("3.0.0")
class ShapleyImportanceEvaluator(BaseImportanceEvaluator):
    """Shapley (SHAP) parameter importance evaluator.

    This evaluator fits a random forest regression model that predicts the objective values
    of :class:`~optuna.trial.TrialState.COMPLETE` trials given their parameter configurations.
    Feature importances are then computed as the mean absolute SHAP values.

    .. note::

        This evaluator requires the `sklearn <https://scikit-learn.org/stable/>`_ Python package
        and `SHAP <https://shap.readthedocs.io/en/stable/index.html>`_.
        The model for the SHAP calculation is based on `sklearn.ensemble.RandomForestClassifier
        <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>`_.

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

        # Use the RandomForest as the surrogate model to evaluate the feature importances.
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
        params: Optional[List[str]] = None,
        *,
        target: Optional[Callable[[FrozenTrial], float]] = None,
    ) -> Dict[str, float]:

        if target is None and study._is_multi_objective():
            raise ValueError(
                "If the `study` is being used for multi-objective optimization, "
                "please specify the `target`. For example, use "
                "`target=lambda t: t.values[0]` for the first objective value."
            )

        distributions = _get_distributions(study, params=params)
        if params is None:
            params = list(distributions.keys())
        assert params is not None
        if len(params) == 0:
            return OrderedDict()

        trials: List[FrozenTrial] = _get_filtered_trials(study, params=params, target=target)
        trans = _SearchSpaceTransform(distributions, transform_log=False, transform_step=False)
        trans_params: np.ndarray = _get_trans_params(trials, trans)
        target_values: np.ndarray = _get_target_values(trials, target)

        forest = self._forest
        forest.fit(X=trans_params, y=target_values)

        # Create Tree Explainer object that can calculate shap values.
        explainer = TreeExplainer(forest)

        # Generate SHAP values for the parameters during the trials.
        feature_shap_values: np.ndarray = explainer.shap_values(trans_params)
        param_shap_values = np.zeros((len(trials), len(params)))
        np.add.at(param_shap_values.T, trans.encoded_column_to_column, feature_shap_values.T)

        # Calculate the mean absolute SHAP value for each parameter.
        # List of tuples ("feature_name": mean_abs_shap_value).
        mean_abs_shap_values = np.abs(param_shap_values).mean(axis=0)

        return _sort_dict_by_importance(_param_importances_to_dict(params, mean_abs_shap_values))
