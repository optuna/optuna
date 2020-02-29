from collections import OrderedDict
from typing import Dict
from typing import Tuple
from typing import ValuesView

import numpy as np

from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.importance._base import _get_search_space
from optuna.importance._base import _get_trial_data
from optuna.importance._base import BaseImportanceEvaluator
from optuna.study import Study

try:
    from sklearn import __version__ as _sklearn_version
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder
    _available = True
except ImportError as e:
    _import_error = e
    _available = False

if _available:
    if _sklearn_version >= '0.22.0':
        from sklearn.inspection import permutation_importance
    else:
        # TODO(hvy): Raise proper error when trying to use evaluator.
        permutation_importance = None


def _transform_params_categorical_to_one_hot(
    params: np.ndarray,
    distributions: ValuesView[BaseDistribution],
) -> Tuple[np.ndarray, np.ndarray]:
    # Transform the `params` matrix by expanding categorical integer-valued columns to one-hot
    # encoding matrices. Note that the resulting matrix can be sparse and potetially very big.

    numerical_cols = []
    categorical_cols = []
    categorical_cols_n_uniques = {}

    for i, distribution in enumerate(distributions):
        if isinstance(distribution, CategoricalDistribution):
            categorical_cols_n_uniques[i] = np.unique(params[:, i]).size
            categorical_cols.append(i)
        else:
            numerical_cols.append(i)

    col_transformer = ColumnTransformer(
        [('_categorical', OneHotEncoder(sparse=False), categorical_cols)], remainder='passthrough')
    # All categorical one-hot columns are placed before the numerical columns in
    # `ColumnTransformer.fit_transform`.
    params = col_transformer.fit_transform(params)

    # `transformed_cols_to_original_cols["column index in transformed matrix"]
    #     == "column index in original matrix"`
    transformed_cols_to_original_cols = np.empty((params.shape[1],), dtype=np.int32)

    i = 0
    for categorical_col in categorical_cols:
        for _ in range(categorical_cols_n_uniques[categorical_col]):
            transformed_cols_to_original_cols[i] = categorical_col
            i += 1
    for numerical_col in numerical_cols:
        transformed_cols_to_original_cols[i] = numerical_col
        i += 1
    assert i == transformed_cols_to_original_cols.size

    return params, transformed_cols_to_original_cols


class RandomForestFeatureImportanceEvaluator(BaseImportanceEvaluator):

    def get_param_importances(self, study: Study) -> Dict[str, float]:
        search_space = _get_search_space(study)
        params, values = _get_trial_data(study, search_space)
        params_transformed, transformed_cols_to_original_cols = (
            _transform_params_categorical_to_one_hot(params, search_space.values()))

        regr = RandomForestRegressor(n_estimators=16, random_state=0)
        regr.fit(params_transformed, values)

        feature_importances = regr.feature_importances_

        feature_importances_reduced = np.zeros((len(search_space),), dtype=np.float32)
        np.add.at(
            feature_importances_reduced, transformed_cols_to_original_cols, feature_importances)

        param_importances = OrderedDict()
        param_names = list(search_space.keys())
        for i in np.asarray(feature_importances_reduced).argsort():
            param_importances[param_names[i]] = feature_importances_reduced[i].item()

        return param_importances


class PermutationImportanceEvaluator(BaseImportanceEvaluator):

    def get_param_importances(self, study: Study) -> Dict[str, float]:
        search_space = _get_search_space(study)
        params, values = _get_trial_data(study, search_space)
        params_transformed, transformed_cols_to_original_cols = (
            _transform_params_categorical_to_one_hot(params, search_space.values()))

        params_train, params_test, values_train, values_test = train_test_split(
            params_transformed, values, test_size=0.2)

        regr = RandomForestRegressor(n_estimators=16, random_state=0)
        regr.fit(params_train, values_train)
        # TODO(hvy): Warn if predictions on test data is too inaccurate.

        result = permutation_importance(regr, params_test, values_test, n_repeats=10)
        perm_importance = np.abs(result.importances_mean)

        feature_importances_reduced = np.zeros((len(search_space),), dtype=np.float32)
        np.add.at(
            feature_importances_reduced, transformed_cols_to_original_cols, perm_importance)

        param_importances = OrderedDict()
        param_names = list(search_space.keys())
        for i in feature_importances_reduced.argsort():
            param_importances[param_names[i]] = feature_importances_reduced[i].item()

        return param_importances
