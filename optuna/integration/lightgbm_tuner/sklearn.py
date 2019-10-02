import warnings

import lightgbm as lgb

from optuna.integration.lightgbm_tuner.optimize import LightGBMTuner
from optuna import type_checking


if type_checking.TYPE_CHECKING:
    from type_checking import Any  # NOQA
    from type_checking import Callable  # NOQA
    from type_checking import Dict  # NOQA
    from type_checking import List  # NOQA
    from type_checking import Optional  # NOQA
    from type_checking import Tuple  # NOQA
    from type_checking import Union  # NOQA

    import numpy as np  # NOQA
    from scipy.sparse.compressed import _cs_matrix  # NOQA


class LGBMModel(lgb.LGBMModel):
    """LightGBM wrapper to tune hyperparameters.

    Detail: `pydoc lightgbm.LGBMModel`

    Example:

        .. code::

            param = {'objective': 'binary', 'metric': 'binary_error'}
            lgb.train(param, dtrain, valid_sets[d_val])
    """

    def fit(self, *args, **kwargs):
        # type: (List[Any], Dict[str, Any]) -> lgb.Booster

        if 'is_tuned' not in self.__dict__:
            warnings.warn('Not tuned hyperparameter yet. Call `tune()` before `fit()`.')
        return super(LGBMModel, self).fit(*args, **kwargs)

    def tune(
            self,
            X,  # type: Union[np.ndarray, _cs_matrix]
            y,  # type: Union[np.ndarray, _cs_matrix]
            sample_weight=None,  # type: Optional[np.ndarray]
            init_score=None,  # type: Optional[np.ndarray]
            group=None,  # type: Optional[np.ndarray]
            eval_set=None,  # type: Optional[List[Tuple[Any, Any]]]
            eval_names=None,  # type: Optional[List[str]]
            eval_sample_weight=None,  # type: Optional[List[np.ndarray]]
            eval_class_weight=None,  # type: Any
            eval_init_score=None,  # type: Optional[List[np.ndarray]]
            eval_group=None,  # type: Any
            eval_metric=None,  # type: Any
            early_stopping_rounds=None,  # type: Optional[int]
            verbose=True,  # type: bool
            feature_name='auto',  # type: Union[List[str], str]
            categorical_feature='auto',  # type: Union[List[str], str]
            callbacks=None,  # type: Optional[Callable[Any, Any]]
    ):
        # type: (...) -> lgb.Booster

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

        auto_booster = LightGBMTuner(
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
        # type: (List[Any], Dict[str, Any]) -> lgb.Booster

        if 'is_tuned' not in self.__dict__:
            warnings.warn('Not tuned hyperparameter yet. Call `tune()` before `fit()`.')
        return super(LGBMClassifier, self).fit(*args, **kwargs)

    def tune(
            self,
            X,  # type: Union[np.ndarray, _cs_matrix]
            y,  # type: Union[np.ndarray, _cs_matrix]
            sample_weight=None,  # type: Optional[np.ndarray]
            init_score=None,  # type: Optional[np.ndarray]
            eval_set=None,  # type: Optional[List[Tuple[Any, Any]]]
            eval_names=None,  # type: Optional[List[str]]
            eval_sample_weight=None,  # type: Optional[List[np.ndarray]]
            eval_class_weight=None,  # type: Any
            eval_init_score=None,  # type: Optional[List[np.ndarray]]
            eval_group=None,  # type: Any
            eval_metric=None,  # type: Any
            early_stopping_rounds=None,  # type: Optional[int]
            verbose=True,  # type: bool
            feature_name='auto',  # type: Union[List[str], str]
            categorical_feature='auto',  # type: Union[List[str], str]
            callbacks=None,  # type: Any
    ):
        # type: (...) -> lgb.Booster

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

        auto_booster = LightGBMTuner(
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
        # type: (List[Any], Dict[str, Any]) -> lgb.Booster

        if 'is_tuned' not in self.__dict__:
            warnings.warn('Not tuned hyperparameter yet. Call `tune()` before `fit()`.')
        return super(LGBMRegressor, self).fit(*args, **kwargs)

    def tune(
            self,
            X,  # type: Union[np.ndarray, _cs_matrix]
            y,  # type: Union[np.ndarray, _cs_matrix]
            sample_weight=None,  # type: Optional[np.ndarray]
            init_score=None,  # type: Optional[np.ndarray]
            eval_set=None,  # type: Optional[List[Tuple[Any, Any]]]
            eval_names=None,  # type: Optional[List[str]]
            eval_sample_weight=None,  # type: Optional[List[np.ndarray]]
            eval_init_score=None,  # type: Optional[List[np.ndarray]]
            eval_metric=None,  # type: Any
            early_stopping_rounds=None,  # type: Optional[int]
            verbose=True,  # type: bool
            feature_name='auto',  # type: Union[List[str], str]
            categorical_feature='auto',  # type: Union[List[str], str]
            callbacks=None,  # type: Any
    ):
        # type: (...) -> lgb.Booster

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

        auto_booster = LightGBMTuner(
            self.get_params(),
            trn_data,
            valid_sets=valid_sets,
            valid_names=eval_names,
            early_stopping_rounds=early_stopping_rounds)
        self._booster = auto_booster.run()
        self.set_params(**auto_booster.get_params())
        return self._booster
