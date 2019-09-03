import warnings

import lightgbm as lgb

from optuna import type_checking
from optuna.integration.lightgbm_autotune.optimize import LGBMAutoTune


if type_checking.TYPE_CHECKING:
    from type_checking import Any  # NOQA
    from type_checking import Optional  # NOQA


class LGBMModel(lgb.LGBMModel):
    """LightGBM wrapper to tune hyperparameters.

    Detail: `pydoc lightgbm.LGBMModel`

    Example:

        .. code::

            param = {'objective': 'binary', 'metric': 'binary_error'}
            lgb.train(param, dtrain, valid_sets[d_val])

    """

    def fit(self, *args, **kwargs):  # NOQA
        if 'is_tuned' not in self.__dict__:
            warnings.warn('Not tuned hyperparameter yet. Call `tune()` before `fit()`.')
        return super().fit(*args, **kwargs)

    def tune(self, X, y,
             sample_weight=None,
             init_score=None,
             group=None,
             eval_set=None,
             eval_names=None,
             eval_sample_weight=None,
             eval_class_weight=None,
             eval_init_score=None,
             eval_group=None,
             eval_metric=None,
             early_stopping_rounds=None,
             verbose=True,
             feature_name='auto',
             categorical_feature='auto',
             callbacks=None):
        self.is_tuned = True

        trn_data = lgb.Dataset(X, label=y)

        if eval_set is None:
            raise ValueError("`eval_set` is required for hyperparameter-tuning.")
        elif early_stopping_rounds is None:
            raise ValueError("`early_stopping_rounds` is required for hyperparameter-tuning.")
        if type(eval_set) is tuple:
            valid_sets = lgb.Dataset(eval_set[0], label=eval_set[1])
        elif type(eval_set) is list:
            valid_sets = [lgb.Dataset(e[0], label=e[1]) for e in eval_set]
        else:
            raise ValueError("given `eval_set` value is unknown type")

        auto_booster = LGBMAutoTune(
            self.get_params(),
            trn_data,
            valid_sets=valid_sets,
            valid_names=eval_names,
            early_stopping_rounds=early_stopping_rounds)

        self._booster = auto_booster.run()

        self.set_params(**auto_booster.get_params())

        return self._booster


class LGBMClassifier(lgb.LGBMClassifier):
    def fit(self, *args, **kwargs):
        if 'is_tuned' not in self.__dict__:
            warnings.warn('Not tuned hyperparameter yet. Call `tune()` before `fit()`.')
        return super().fit(*args, **kwargs)

    def tune(self, X, y, sample_weight=None, init_score=None,
             eval_set=None, eval_names=None, eval_sample_weight=None,
             eval_class_weight=None, eval_init_score=None, eval_group=None,
             eval_metric=None, early_stopping_rounds=None, verbose=True,
             feature_name='auto', categorical_feature='auto', callbacks=None):
        self.is_tuned = True

        trn_data = lgb.Dataset(X, label=y)

        if eval_set is None:
            raise ValueError("`eval_set` is required for hyperparameter-tuning.")
        elif early_stopping_rounds is None:
            raise ValueError("`early_stopping_rounds` is required for hyperparameter-tuning.")
        if type(eval_set) is tuple:
            valid_sets = lgb.Dataset(eval_set[0], label=eval_set[1])
        elif type(eval_set) is list:
            valid_sets = [lgb.Dataset(e[0], label=e[1]) for e in eval_set]
        else:
            raise ValueError("given `eval_set` value is unknown type")

        auto_booster = LGBMAutoTune(
            self.get_params(),
            trn_data,
            valid_sets=valid_sets,
            valid_names=eval_names,
            early_stopping_rounds=early_stopping_rounds)
        self._booster = auto_booster.run()
        self.set_params(**auto_booster.get_params())
        return self._booster


class LGBMRegressor(lgb.LGBMRegressor):
    def fit(self, *args, **kwargs):
        if 'is_tuned' not in self.__dict__:
            warnings.warn('Not tuned hyperparameter yet. Call `tune()` before `fit()`.')
        return super().fit(*args, **kwargs)

    def tune(self, X, y,
             sample_weight=None,
             init_score=None,
             eval_set=None,
             eval_names=None,
             eval_sample_weight=None,
             eval_init_score=None,
             eval_metric=None,
             early_stopping_rounds=None,
             verbose=True,
             feature_name='auto',
             categorical_feature='auto',
             callbacks=None):
        self.is_tuned = True

        trn_data = lgb.Dataset(X, label=y)

        if eval_set is None:
            raise ValueError("`eval_set` is required for hyperparameter-tuning.")
        elif early_stopping_rounds is None:
            raise ValueError("`early_stopping_rounds` is required for hyperparameter-tuning.")
        if type(eval_set) is tuple:
            valid_sets = lgb.Dataset(eval_set[0], label=eval_set[1])
        elif type(eval_set) is list:
            valid_sets = [lgb.Dataset(e[0], label=e[1]) for e in eval_set]
        else:
            raise ValueError("given `eval_set` value is unknown type")

        auto_booster = LGBMAutoTune(
            self.get_params(),
            trn_data,
            valid_sets=valid_sets,
            valid_names=eval_names,
            early_stopping_rounds=early_stopping_rounds)
        self._booster = auto_booster.run()
        self.set_params(**auto_booster.get_params())
        return self._booster
