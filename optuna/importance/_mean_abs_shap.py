from collections import OrderedDict
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

import pandas as pd

from optuna._imports import try_import
from optuna.importance._mean_decrease_impurity import MeanDecreaseImpurityImportanceEvaluator
from optuna.study import Study
from optuna.trial import FrozenTrial


with try_import() as _imports:
    from shap import TreeExplainer


class ShapleyImportanceEvaluator(MeanDecreaseImpurityImportanceEvaluator):
    """Shapley (SHAP) parameter importance evaluator.

    This evaluator fits a random forest that predicts objective values given hyperparameter
    configurations. Feature importances are then computed using the mean absolute SHAP value.

    .. note::

        This evaluator requires the `sklean <https://scikit-learn.org/stable/>`_ Python package and
        `SHAP <https://shap.readthedocs.io/en/latest/index.html>`_.
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

        MeanDecreaseImpurityImportanceEvaluator.__init__(
            self, n_trees=n_trees, max_depth=max_depth, seed=seed
        )
        # Explainer from SHAP
        self._explainer: TreeExplainer = None

    def evaluate(
        self,
        study: Study,
        params: Optional[List[str]] = None,
        *,
        target: Optional[Callable[[FrozenTrial], float]] = None,
    ) -> Dict[str, float]:

        # Train a RandomForest from the parent class
        super().evaluate(study=study, params=params, target=target)

        # Create Tree Explainer object that can calculate shap values
        self._explainer = TreeExplainer(self._forest)

        # Generate SHAP values for the parameters during the trials
        shap_values = self._explainer.shap_values(self._trans_params)
        df_shap = pd.DataFrame(shap_values, columns=self._param_names)

        # Calculate the mean absolute SHAP value for each parameter
        mean_abs_shap_values = []
        for param in df_shap.columns:
            mean_abs_shap_values.append((param, df_shap[param].abs().mean()))

        # Use the mean absolute SHAP values as the feature importance
        mean_abs_shap_values.sort(key=lambda t: t[1], reverse=True)
        feature_importances = OrderedDict(mean_abs_shap_values)

        return feature_importances
