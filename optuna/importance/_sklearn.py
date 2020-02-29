from collections import OrderedDict
from typing import Dict

import numpy as np

from optuna.importance._base import BaseImportanceEvaluator
from optuna.importance._base import _get_search_space
from optuna.importance._base import _get_trial_data
from optuna.study import Study

try:
    from sklearn import __version__ as _sklearn_version
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
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


class RandomForestFeatureImportanceEvaluator(BaseImportanceEvaluator):

    def get_param_importance(self, study: Study) -> Dict[str, float]:
        search_space = _get_search_space(study)
        params, values = _get_trial_data(study, search_space)

        regr = RandomForestRegressor(n_estimators=16)
        regr.fit(params, values)

        feature_importances = regr.feature_importances_

        param_importances = OrderedDict()
        param_names = list(search_space.keys())
        for i in feature_importances.argsort():
            param_importances[param_names[i]] = feature_importances[i]

        return param_importances


class PermutationImportanceEvaluator(BaseImportanceEvaluator):

    def get_param_importance(self, study: Study) -> Dict[str, float]:
        search_space = _get_search_space(study)
        params, values = _get_trial_data(study, search_space)

        params_train, params_test, values_train, values_test = train_test_split(
            params, values, test_size=0.2)

        regr = RandomForestRegressor(n_estimators=16)
        regr.fit(params_train, values_train)
        # TODO(hvy): Warn if predictions on test data is too inaccurate.

        result = permutation_importance(regr, params_test, values_test, n_repeats=10)
        perm_importance = np.abs(result.importances_mean)

        param_importances = OrderedDict()
        param_names = list(search_space.keys())
        for i in perm_importance.argsort():
            param_importances[param_names[i]] = perm_importance[i]

        return param_importances
