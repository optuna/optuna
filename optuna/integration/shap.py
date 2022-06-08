from collections import OrderedDict
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

import numpy as np

from optuna._experimental import experimental_class
from optuna._imports import try_import
from optuna.importance._base import BaseImportanceEvaluator
from optuna.importance._mean_decrease_impurity import MeanDecreaseImpurityImportanceEvaluator
from optuna.study import Study
from optuna.trial import FrozenTrial


with try_import() as _imports:
    from shap import TreeExplainer


@experimental_class("3.0.0")
class ShapleyImportanceEvaluator(BaseImportanceEvaluator):
    """Shapley (SHAP) parameter importance evaluator.

    This evaluator fits a random forest that predicts objective values given hyperparameter
    configurations. Feature importances are then computed as the mean absolute SHAP values.

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
        self._backend_evaluator = MeanDecreaseImpurityImportanceEvaluator(
            n_trees=n_trees, max_depth=max_depth, seed=seed
        )
        # Use the TreeExplainer from the SHAP module.
        self._explainer: TreeExplainer = None

    def evaluate(
        self,
        study: Study,
        params: Optional[List[str]] = None,
        *,
        target: Optional[Callable[[FrozenTrial], float]] = None,
    ) -> Dict[str, float]:

        # Train a RandomForest from the backend evaluator.
        self._backend_evaluator.evaluate(study=study, params=params, target=target)

        # Create Tree Explainer object that can calculate shap values.
        self._explainer = TreeExplainer(self._backend_evaluator._forest)

        # Generate SHAP values for the parameters during the trials.
        shap_values = self._explainer.shap_values(self._backend_evaluator._trans_params)

        # Calculate the mean absolute SHAP value for each parameter.
        # List of tuples ("feature_name": mean_abs_shap_value).
        mean_abs_shap_values = list(
            zip(self._backend_evaluator._param_names, np.abs(shap_values).mean(axis=0))
        )

        # Use the mean absolute SHAP values as the feature importance.
        mean_abs_shap_values.sort(key=lambda t: t[1], reverse=True)
        feature_importances = OrderedDict(mean_abs_shap_values)

        return feature_importances
