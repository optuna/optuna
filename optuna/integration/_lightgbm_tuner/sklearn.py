from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import is_classifier
from sklearn.base import RegressorMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_array
from sklearn.utils import check_consistent_length
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import _assert_all_finite
from sklearn.utils.validation import _num_samples
from sklearn.utils.validation import column_or_1d

from optuna import logging
from optuna import study as study_module
from optuna.integration._lightgbm_tuner import optimize


if lgb.__version__ >= "2.3":
    from lightgbm.sklearn import _EvalFunctionWrapper
    from lightgbm.sklearn import _ObjectiveFunctionWrapper
else:
    from lightgbm.sklearn import _eval_function_wrapper as _EvalFunctionWrapper
    from lightgbm.sklearn import _objective_function_wrapper as _ObjectiveFunctionWrapper

__all__ = ["LGBMModel", "LGBMClassifier", "LGBMRegressor"]

OneDimArrayLikeType = Union[np.ndarray, pd.Series]
TwoDimArrayLikeType = Union[np.ndarray, pd.DataFrame, spmatrix]

RandomStateType = Union[int, np.random.RandomState]


def check_X(
    X: TwoDimArrayLikeType, estimator: Optional[BaseEstimator] = None, **kwargs: Any
) -> TwoDimArrayLikeType:
    """Check ``X``.

    Args:
        X:
            Data.

        estimator:
            Object to use to fit the data.

        **kwargs:
            Other keywords passed to ``sklearn.utils.check_array``.

    Returns:
        X:
            Converted and validated data.
    """
    if not isinstance(X, pd.DataFrame):
        X = check_array(X, estimator=estimator, **kwargs)

    _, actual_n_features = X.shape
    expected_n_features = getattr(estimator, "n_features_", actual_n_features)

    if actual_n_features != expected_n_features:
        raise ValueError(
            "`n_features` must be {} but was {}.".format(expected_n_features, actual_n_features)
        )

    return X


def check_fit_params(
    X: TwoDimArrayLikeType,
    y: OneDimArrayLikeType,
    sample_weight: Optional[OneDimArrayLikeType] = None,
    estimator: Optional[BaseEstimator] = None,
    **kwargs: Any
) -> Tuple[TwoDimArrayLikeType, OneDimArrayLikeType, OneDimArrayLikeType]:
    """Check ``X``, ``y`` and ``sample_weight``.

    Args:
        X:
            Data.

        y:
            Target.

        sample_weight:
            Weights of data.

        estimator:
            Object to use to fit the data.

        **kwargs:
            Other keywords passed to ``sklearn.utils.check_array``.

    Returns:
        X:
            Converted and validated data.

        y:
            Converted and validated target.

        sample_weight:
            Converted and validated weights of data.
    """
    X = check_X(X, estimator=estimator, **kwargs)

    if not isinstance(y, pd.Series):
        y = column_or_1d(y, warn=True)

    _assert_all_finite(y)

    if is_classifier(estimator):
        check_classification_targets(y)

    if sample_weight is None:
        n_samples = _num_samples(X)
        sample_weight = np.ones(n_samples)

    sample_weight = np.asarray(sample_weight)

    class_weight = getattr(estimator, "class_weight", None)

    if class_weight is not None:
        sample_weight *= compute_sample_weight(class_weight, y)

    check_consistent_length(X, y, sample_weight)

    return X, y, sample_weight


class LGBMModel(lgb.LGBMModel):
    """Base class for models."""

    def __init__(
        self,
        boosting_type: str = "gbdt",
        num_leaves: int = 31,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        subsample_for_bin: int = 200000,
        objective: Optional[Union[Callable, str]] = None,
        class_weight: Optional[Union[Dict[str, float], str]] = None,
        min_split_gain: float = 0.0,
        min_child_weight: float = 1e-03,
        min_child_samples: int = 20,
        subsample: float = 1.0,
        subsample_freq: int = 0,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        random_state: Optional[RandomStateType] = None,
        n_jobs: int = -1,
        silent: bool = True,
        importance_type: str = "split",
        study: Optional[study_module.Study] = None,
        timeout: Optional[int] = None,
        model_dir: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(
            boosting_type=boosting_type,
            class_weight=class_weight,
            colsample_bytree=colsample_bytree,
            importance_type=importance_type,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_samples=min_child_samples,
            min_child_weight=min_child_weight,
            min_split_gain=min_split_gain,
            num_leaves=num_leaves,
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            objective=objective,
            random_state=random_state,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            silent=silent,
            subsample=subsample,
            subsample_for_bin=subsample_for_bin,
            subsample_freq=subsample_freq,
            **kwargs,
        )

        self.study = study
        self.timeout = timeout
        self.model_dir = model_dir

    def _check_is_fitted(self) -> None:
        getattr(self, "n_features_")

    def _get_objective(self) -> str:
        if isinstance(self.objective, str):
            return self.objective

        if self._n_classes is None:
            return "regression"
        elif self._n_classes > 2:
            return "multiclass"
        else:
            return "binary"

    def fit(
        self,
        X: TwoDimArrayLikeType,
        y: OneDimArrayLikeType,
        sample_weight: Optional[OneDimArrayLikeType] = None,
        init_score: Optional[OneDimArrayLikeType] = None,
        group: Optional[OneDimArrayLikeType] = None,
        eval_set: Optional[List[Tuple[TwoDimArrayLikeType, OneDimArrayLikeType]]] = None,
        eval_names: Optional[List[str]] = None,
        eval_sample_weight: Optional[List[OneDimArrayLikeType]] = None,
        eval_class_weight: Optional[List[Union[Dict[str, float], str]]] = None,
        eval_init_score: Optional[List[OneDimArrayLikeType]] = None,
        eval_group: Optional[List[OneDimArrayLikeType]] = None,
        eval_metric: Optional[Union[Callable, List[str], str]] = None,
        early_stopping_rounds: Optional[int] = 10,
        verbose: bool = True,
        feature_name: Union[List[str], str] = "auto",
        categorical_feature: Union[List[int], List[str], str] = "auto",
        callbacks: Optional[List[Callable]] = None,
        init_model: Optional[Union[lgb.Booster, lgb.LGBMModel, str]] = None,
        **fit_params: Any
    ) -> "LGBMModel":
        """Fit the model according to the given training data.

        Args:
            X:
                Training data.

            y:
                Target.

            sample_weight:
                Weights of training data.

            init_score:
                Init score of training data.

            group:
                Group data of training data.

            eval_set:
                List of (X, y) tuple pairs to use as validation sets.

            eval_names:
                Names of eval_set.

            eval_sample_weight:
                Weights of eval data.

            eval_class_weight:
                Class weights of eval data.

            eval_init_score:
                Init score of eval data.

            eval_group:
                Group data of eval data.

            eval_metric:
                Evaluation metric. See
                https:/lightgbm.readthedocs.io/en/latest/Parameters.html#metric.

            early_stopping_rounds:
                Used to activate early stopping. The model will train until the
                validation score stops improving.

            verbose:
                If True, the eval metric on the eval set is printed at each
                boosting stage. If int, the eval metric on the eval set is
                printed at every verbose boosting stage.

            feature_name:
                Feature names. If 'auto' and data is pandas DataFrame, data
                columns names are used.

            categorical_feature:
                Categorical features. If list of int, interpreted as indices.
                If list of strings, interpreted as feature names. If 'auto' and
                data is pandas DataFrame, pandas categorical columns are used.
                All values in categorical features should be less than int32
                max value (2147483647). Large values could be memory consuming.
                Consider using consecutive integers starting from zero. All
                negative values in categorical features will be treated as
                missing values.

            callbacks:
                List of callback functions that are applied at each iteration.

            init_model:
                Filename of LightGBM model, Booster instance or LGBMModel
                instance used for continue training.

            **fit_params:
                Always ignored. This parameter exists for compatibility.

        Returns:
            self:
                Return self.
        """
        logger = logging.get_logger(__name__)

        X, y, sample_weight = check_fit_params(
            X,
            y,
            sample_weight=sample_weight,
            accept_sparse=True,
            ensure_min_samples=2,
            estimator=self,
            force_all_finite=False,
        )

        n_samples, self._n_features = X.shape

        self._n_features_in = self._n_features

        for key, value in fit_params.items():
            logger.warning("{}={} will be ignored.".format(key, value))

        params = self.get_params()

        if (
            not any(verbose_alias in params for verbose_alias in ("verbose", "verbosity"))
            and self.silent
        ):
            params["verbose"] = -1

        for attr in (
            "class_weight",
            "importance_type",
            "n_estimators",
            "silent",
            "study",
            "timeout",
            "model_dir",
        ):
            params.pop(attr, None)

        params["objective"] = self._get_objective()

        if self._n_classes is not None and self._n_classes > 2:
            params["num_classes"] = self._n_classes

        if callable(eval_metric):
            params["metric"] = "None"
            feval = _EvalFunctionWrapper(eval_metric)

        elif isinstance(eval_metric, list):
            raise ValueError("eval_metric is not allowed to be a list.")

        else:
            if eval_metric is None:
                params["metric"] = params["objective"]
            else:
                params["metric"] = eval_metric

            feval = None

        fobj = _ObjectiveFunctionWrapper(self.objective) if callable(self.objective) else None

        init_model = init_model.booster_ if isinstance(init_model, lgb.LGBMModel) else init_model

        train_set = lgb.Dataset(
            X,
            label=y,
            weight=sample_weight,
            group=group,
            init_score=init_score,
            feature_name=feature_name,
            categorical_feature=categorical_feature,
        )

        valid_sets = []

        if eval_set is not None:

            def _get_meta_data(collection: Optional[List], i: int) -> Any:
                if collection is None:
                    return None
                elif isinstance(collection, list):
                    return collection[i] if len(collection) > i else None
                elif isinstance(collection, dict):
                    return collection.get(i, None)
                else:
                    raise TypeError(
                        "eval_sample_weight, eval_class_weight, eval_init_score, and eval_group "
                        "should be dict or list"
                    )

            for i, (X_valid, y_valid) in enumerate(eval_set):
                valid_class_weight = _get_meta_data(eval_class_weight, i)
                valid_group = _get_meta_data(eval_group, i)
                valid_init_score = _get_meta_data(eval_init_score, i)
                valid_weight = _get_meta_data(eval_sample_weight, i)

                if valid_class_weight is not None:
                    valid_class_sample_weight = compute_sample_weight(valid_class_weight, y_valid)

                    if valid_weight is None or len(valid_weight) == 0:
                        valid_weight = valid_class_sample_weight
                    else:
                        valid_weight *= valid_class_sample_weight

                valid_set = lgb.Dataset(
                    X_valid,
                    label=y_valid,
                    weight=valid_weight,
                    group=valid_group,
                    init_score=valid_init_score,
                    feature_name=feature_name,
                    categorical_feature=categorical_feature,
                )

            valid_sets.append(valid_set)

        evals_result = {}  # type: Dict[Any, Any]
        tuner = optimize.LightGBMTuner(
            params,
            train_set,
            num_boost_round=self.n_estimators,
            valid_sets=valid_sets,
            valid_names=eval_names,
            fobj=fobj,
            feval=feval,
            # TODO(Kon): Pass init_model to LightGBMTuner
            # init_model=init_model,
            early_stopping_rounds=early_stopping_rounds,
            evals_result=evals_result,
            verbose_eval=verbose,
            callbacks=callbacks,
            time_budget=self.timeout,
            study=self.study,
            model_dir=self.model_dir,
        )

        tuner.run()

        self._Booster = tuner.get_best_booster()
        self._best_iteration = (
            None if early_stopping_rounds is None else self._Booster.best_iteration
        )
        self._best_score = self._Booster.best_score
        self._evals_result = evals_result if evals_result else None
        self._objective = params["objective"]

        self._Booster.free_dataset()

        return self


class LGBMClassifier(LGBMModel, ClassifierMixin):
    """LightGBM classifier using Optuna.

    Args:
        boosting_type:
            Boosting type.

            - 'dart', Dropouts meet Multiple Additive Regression Trees,
            - 'gbdt', traditional Gradient Boosting Decision Tree,
            - 'goss', Gradient-based One-Side Sampling,
            - 'rf', Random Forest.

        num_leaves:
            Maximum tree leaves for base learners.

        max_depth:
            Maximum depth of each tree. -1 means no limit.

        learning_rate:
            Learning rate. You can use ``callbacks`` parameter of ``fit``
            method to shrinkadapt learning rate in training using
            ``reset_parameter`` callback. Note, that this will ignore the
            ``learning_rate`` argument in training.

        n_estimators:
            Maximum number of iterations of the boosting process. a.k.a.
            ``num_boost_round``.

        subsample_for_bin:
            Number of samples for constructing bins.

        objective:
            Objective function.

        class_weight:
            Weights associated with classes in the form
            ``{class_label: weight}``. This parameter is used only for
            multi-class classification task. For binary classification task you
            may use ``is_unbalance`` or ``scale_pos_weight`` parameters. The
            'balanced' mode uses the values of y to automatically adjust
            weights inversely proportional to class frequencies in the input
            data as ``n_samples  (n_classes * np.bincount(y))``. If None, all
            classes are supposed to have weight one. Note, that these weights
            will be multiplied with ``sample_weight`` if ``sample_weight`` is
            specified.

        min_split_gain:
            Minimum loss reduction required to make a further partition on a
            leaf node of the tree.

        min_child_weight:
            Minimum sum of instance weight (hessian) needed in a child (leaf).

        min_child_samples:
            Minimum number of data needed in a child (leaf).

        subsample:
            Subsample ratio of the training instance.

        subsample_freq:
            Frequence of subsample. <=0 means no enable.

        colsample_bytree:
            Subsample ratio of columns when constructing each tree.

        reg_alpha:
            L1 regularization term on weights.

        reg_lambda:
            L2 regularization term on weights.

        random_state:
            Seed of the pseudo random number generator. If int, this is the
            seed used by the random number generator. If
            ``numpy.random.RandomState`` object, this is the random number
            generator. If None, the global random state from ``numpy.random``
            is used.

        n_jobs:
            Number of parallel jobs. -1 means using all processors.

        silent:
            If true, print messages while running boosting.

        importance_type:
            Type of feature importances. If 'split', result contains numbers of
            times the feature is used in a model. If 'gain', result contains
            total gains of splits which use the feature.

        study:
            Study corresponds to the optimization task. If None, a new study is
            created.

        timeout:
            Time limit in seconds for the search of appropriate models. If
            None, the study is executed without time limitation. If
            ``n_trials`` is also set to None, the study continues to create
            trials until it receives a termination signal such as Ctrl+C or
            SIGTERM. This trades off runtime vs quality of the solution.

        model_dir:
            Directory for storing the files generated during training.

        **kwargs:
            Other parameters for the model. See
            http:/lightgbm.readthedocs.io/en/latest/Parameters.html for more
            parameters. Note, that **kwargs is not supported in sklearn, so it
            may cause unexpected issues.

    Attributes:
        encoder_:
            Label encoder.

    Examples:
        >>> import optuna.integration.lightgbm as lgb
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.model_selection import train_test_split
        >>> clf = lgb.LGBMClassifier(random_state=0)
        >>> X, y = load_iris(return_X_y=True)
        >>> X, X_valid, y, y_valid = train_test_split(X, y, random_state=0)
        >>> clf.fit(X, y, eval_set=[(X_valid, y_valid)])
        LGBMClassifier(...)
        >>> y_pred = clf.predict(X)
    """

    @property
    def classes_(self) -> np.ndarray:
        """Get the class labels."""
        self._check_is_fitted()

        return self._classes

    @property
    def n_classes_(self) -> int:
        """Get the number of classes."""
        self._check_is_fitted()

        return self._n_classes

    def fit(
        self,
        X: TwoDimArrayLikeType,
        y: OneDimArrayLikeType,
        sample_weight: Optional[OneDimArrayLikeType] = None,
        init_score: Optional[OneDimArrayLikeType] = None,
        group: Optional[OneDimArrayLikeType] = None,
        eval_set: Optional[List[Tuple[TwoDimArrayLikeType, OneDimArrayLikeType]]] = None,
        eval_names: Optional[List[str]] = None,
        eval_sample_weight: Optional[List[OneDimArrayLikeType]] = None,
        eval_class_weight: Optional[List[Union[Dict[str, float], str]]] = None,
        eval_init_score: Optional[List[OneDimArrayLikeType]] = None,
        eval_group: Optional[List[OneDimArrayLikeType]] = None,
        eval_metric: Optional[Union[Callable, List[str], str]] = None,
        early_stopping_rounds: Optional[int] = 10,
        verbose: bool = True,
        feature_name: Union[List[str], str] = "auto",
        categorical_feature: Union[List[int], List[str], str] = "auto",
        callbacks: Optional[List[Callable]] = None,
        init_model: Optional[Union[lgb.Booster, lgb.LGBMModel, str]] = None,
        **fit_params: Any
    ) -> "LGBMClassifier":
        """Docstring is inherited from the LGBMModel."""
        self.encoder_ = LabelEncoder()

        y = self.encoder_.fit_transform(y)

        self._classes = self.encoder_.classes_
        self._n_classes = len(self.encoder_.classes_)

        return super().fit(
            X,
            y,
            sample_weight=sample_weight,
            init_score=init_score,
            group=group,
            eval_set=eval_set,
            eval_names=eval_names,
            eval_sample_weight=eval_sample_weight,
            eval_class_weight=eval_class_weight,
            eval_init_score=eval_init_score,
            eval_group=eval_group,
            eval_metric=eval_metric,
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose,
            feature_name=feature_name,
            categorical_feature=categorical_feature,
            callbacks=callbacks,
            init_model=init_model,
            **fit_params,
        )

    fit.__doc__ = LGBMModel.fit.__doc__

    def predict(
        self,
        X: TwoDimArrayLikeType,
        raw_score: bool = False,
        num_iteration: Optional[int] = None,
        pred_leaf: bool = False,
        pred_contrib: bool = False,
        **predict_params: Any
    ) -> np.ndarray:
        """Docstring is inherited from the LGBMModel."""
        result = self.predict_proba(
            X,
            raw_score=raw_score,
            num_iteration=num_iteration,
            pred_leaf=pred_leaf,
            pred_contrib=pred_contrib,
            **predict_params,
        )

        if raw_score or pred_leaf or pred_contrib:
            return result

        class_index = np.argmax(result, axis=1)

        return self.encoder_.inverse_transform(class_index)

    predict.__doc__ = LGBMModel.predict.__doc__

    def predict_proba(
        self,
        X: TwoDimArrayLikeType,
        raw_score: bool = False,
        num_iteration: Optional[int] = None,
        pred_leaf: bool = False,
        pred_contrib: bool = False,
        **predict_params: Any
    ) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X:
                Data.

            raw_score:
                If True, return raw scores.

            num_iteration:
                Limit number of iterations in the prediction. If None, if the
                best iteration exists, it is used; otherwise, all trees are
                used. If <=0, all trees are used (no limits).

            pred_leaf:
                If True, return leaf indices.

            pred_contrib:
                If True, return feature contributions.

            **predict_params:
                Ignored if refit is set to False.

        Returns:
            p:
                Class probabilities, raw scores, leaf indices or feature
                contributions.
        """
        result = super().predict(
            X,
            raw_score=raw_score,
            num_iteration=num_iteration,
            pred_leaf=pred_leaf,
            pred_contrib=pred_contrib,
            **predict_params,
        )

        if self._n_classes > 2 or raw_score or pred_leaf or pred_contrib:
            return result

        preds = result.reshape(-1, 1)

        return np.concatenate([1.0 - preds, preds], axis=1)


class LGBMRegressor(LGBMModel, RegressorMixin):
    """LightGBM regressor using Optuna.

    Args:
        boosting_type:
            Boosting type.

            - 'dart', Dropouts meet Multiple Additive Regression Trees,
            - 'gbdt', traditional Gradient Boosting Decision Tree,
            - 'goss', Gradient-based One-Side Sampling,
            - 'rf', Random Forest.

        num_leaves:
            Maximum tree leaves for base learners.

        max_depth:
            Maximum depth of each tree. -1 means no limit.

        learning_rate:
            Learning rate. You can use ``callbacks`` parameter of ``fit``
            method to shrinkadapt learning rate in training using
            ``reset_parameter`` callback. Note, that this will ignore the
            ``learning_rate`` argument in training.

        n_estimators:
            Maximum number of iterations of the boosting process. a.k.a.
            ``num_boost_round``.

        subsample_for_bin:
            Number of samples for constructing bins.

        objective:
            Objective function.

        class_weight:
            Weights associated with classes in the form
            ``{class_label: weight}``. This parameter is used only for
            multi-class classification task. For binary classification task you
            may use ``is_unbalance`` or ``scale_pos_weight`` parameters. The
            'balanced' mode uses the values of y to automatically adjust
            weights inversely proportional to class frequencies in the input
            data as ``n_samples  (n_classes * np.bincount(y))``. If None, all
            classes are supposed to have weight one. Note, that these weights
            will be multiplied with ``sample_weight`` if ``sample_weight`` is
            specified.

        min_split_gain:
            Minimum loss reduction required to make a further partition on a
            leaf node of the tree.

        min_child_weight:
            Minimum sum of instance weight (hessian) needed in a child (leaf).

        min_child_samples:
            Minimum number of data needed in a child (leaf).

        subsample:
            Subsample ratio of the training instance.

        subsample_freq:
            Frequence of subsample. <=0 means no enable.

        colsample_bytree:
            Subsample ratio of columns when constructing each tree.

        reg_alpha:
            L1 regularization term on weights.

        reg_lambda:
            L2 regularization term on weights.

        random_state:
            Seed of the pseudo random number generator. If int, this is the
            seed used by the random number generator. If
            ``numpy.random.RandomState`` object, this is the random number
            generator. If None, the global random state from ``numpy.random``
            is used.

        n_jobs:
            Number of parallel jobs. -1 means using all processors.

        silent:
            If true, print messages while running boosting.

        importance_type:
            Type of feature importances. If 'split', result contains numbers of
            times the feature is used in a model. If 'gain', result contains
            total gains of splits which use the feature.

        study:
            Study corresponds to the optimization task. If None, a new study is
            created.

        timeout:
            Time limit in seconds for the search of appropriate models. If
            None, the study is executed without time limitation. If
            ``n_trials`` is also set to None, the study continues to create
            trials until it receives a termination signal such as Ctrl+C or
            SIGTERM. This trades off runtime vs quality of the solution.

        model_dir:
            Directory for storing the files generated during training.

        **kwargs:
            Other parameters for the model. See
            http:/lightgbm.readthedocs.io/en/latest/Parameters.html for more
            parameters. Note, that **kwargs is not supported in sklearn, so it
            may cause unexpected issues.

        >>> import optuna.integration.lightgbm as lgb
        >>> from sklearn.datasets import load_boston
        >>> from sklearn.model_selection import train_test_split
        >>> reg = lgb.LGBMRegressor(random_state=0)
        >>> X, y = load_boston(return_X_y=True)
        >>> X, X_valid, y, y_valid = train_test_split(X, y, random_state=0)
        >>> reg.fit(X, y, eval_set=[(X_valid, y_valid)])
        LGBMRegressor(...)
        >>> y_pred = reg.predict(X)
    """
