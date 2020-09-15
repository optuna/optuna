from collections import OrderedDict
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import ValuesView

import numpy

from optuna._imports import try_import
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.importance._base import _get_distributions
from optuna.importance._base import _get_study_data
from optuna.importance._base import BaseImportanceEvaluator
from optuna.study import Study


with try_import() as _imports:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import OneHotEncoder


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
        params_data, values_data = _get_study_data(study, distributions)

        if params_data.size == 0:  # `params` were given but as an empty list.
            return OrderedDict()

        params_data, cols_to_raw_cols = _encode_categorical(params_data, distributions.values())

        forest = self._forest
        forest.fit(params_data, values_data)
        feature_importances = forest.feature_importances_
        feature_importances_reduced = numpy.zeros(len(distributions))
        numpy.add.at(feature_importances_reduced, cols_to_raw_cols, feature_importances)

        param_importances = OrderedDict()
        param_names = list(distributions.keys())
        for i in feature_importances_reduced.argsort()[::-1]:
            param_importances[param_names[i]] = feature_importances_reduced[i].item()

        return param_importances


def _encode_categorical(
    params_data: numpy.ndarray, distributions: ValuesView[BaseDistribution]
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    # Transform the `params_data` matrix by expanding categorical integer-valued columns to one-hot
    # encoding matrices. Note that the resulting matrix can be sparse and potentially very big.

    numerical_cols = []
    categorical_cols = []
    categories = []

    for col, distribution in enumerate(distributions):
        if isinstance(distribution, CategoricalDistribution):
            categorical_cols.append(col)
            categories.append(list(range(len(distribution.choices))))
        else:
            numerical_cols.append(col)
    assert col == params_data.shape[1] - 1

    col_transformer = ColumnTransformer(
        [("_categorical", OneHotEncoder(categories=categories, sparse=False), categorical_cols)],
        remainder="passthrough",
    )
    # All categorical one-hot columns are placed before the numerical columns in
    # `ColumnTransformer.fit_transform`.
    params_data = col_transformer.fit_transform(params_data)

    # `cols_to_raw_cols["column index in transformed matrix"]
    #     == "column index in original matrix"`
    cols_to_raw_cols = numpy.empty((params_data.shape[1],), dtype=numpy.int32)

    i = 0
    for categorical_col, cats in zip(categorical_cols, categories):
        for _ in range(len(cats)):
            cols_to_raw_cols[i] = categorical_col
            i += 1
    for numerical_col in numerical_cols:
        cols_to_raw_cols[i] = numerical_col
        i += 1
    assert i == cols_to_raw_cols.size

    return params_data, cols_to_raw_cols
