import copy
import time

from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Tuple
from typing import Union

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn

from lightgbm.engine import _CVBooster
from scipy.sparse import spmatrix
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import is_classifier
from sklearn.base import RegressorMixin
from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection import check_cv as sklearn_check_cv
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_array
from sklearn.utils import check_consistent_length
from sklearn.utils import check_random_state
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import _assert_all_finite
from sklearn.utils.validation import _num_samples
from sklearn.utils.validation import column_or_1d

from optuna import distributions
from optuna import integration
from optuna import logging
from optuna import samplers
from optuna import study as study_module
from optuna import trial as trial_module

if lgb.__version__ >= "2.3":
    from lightgbm.sklearn import _EvalFunctionWrapper
    from lightgbm.sklearn import _ObjectiveFunctionWrapper
else:
    from lightgbm.sklearn import _eval_function_wrapper as _EvalFunctionWrapper
    from lightgbm.sklearn import _objective_function_wrapper as _ObjectiveFunctionWrapper

if sklearn.__version__ >= "0.22":
    from sklearn.utils import _safe_indexing
else:
    from sklearn.utils import safe_indexing as _safe_indexing

__all__ = ["LGBMModel", "LGBMClassifier", "LGBMRegressor"]

CVType = Union[BaseCrossValidator, int, List[Tuple]]

LightGBMCallbackEnvType = NamedTuple(
    "LightGBMCallbackEnv",
    [
        ("model", _CVBooster),
        ("params", Dict[str, Any]),
        ("iteration", int),
        ("begin_iteration", int),
        ("end_iteration", int),
        ("evaluation_result_list", List),
    ],
)

OneDimArrayLikeType = Union[np.ndarray, pd.Series]
TwoDimArrayLikeType = Union[np.ndarray, pd.DataFrame, spmatrix]

RandomStateType = Union[int, np.random.RandomState]

MAX_INT = np.iinfo(np.int32).max

OBJECTIVE2METRIC = {
    # classification
    "binary": "binary_logloss",
    "multiclass": "multi_logloss",
    "softmax": "multi_logloss",
    "multiclassova": "multi_logloss",
    "multiclass_ova": "multi_logloss",
    "ova": "multi_logloss",
    "ovr": "multi_logloss",
    # regression
    "mean_absoluter_error": "l1",
    "mae": "l1",
    "regression_l1": "l1",
    "l2_root": "l2",
    "mean_squared_error": "l2",
    "mse": "l2",
    "regression": "l2",
    "regression_l2": "l2",
    "root_mean_squared_error": "l2",
    "rmse": "l2",
    "huber": "huber",
    "fair": "fair",
    "poisson": "poisson",
    "quantile": "quantile",
    "mean_absolute_percentage_error": "mape",
    "mape": "mape",
    "gamma": "gamma",
    "tweedie": "tweedie",
}


def check_cv(
    cv: CVType = 5, y: Optional[OneDimArrayLikeType] = None, classifier: bool = False,
) -> BaseCrossValidator:
    """Check ``cv``.

    Args:
        cv:
            Cross-validation strategy.

        y:
            Target.

        classifier:
            If the task is a classification task, ``StratifiedKFold`` will be
            used.

    Returns:
        cv:
            Converted cross-validation strategy.
    """
    if classifier and isinstance(cv, int):
        _, counts = np.unique(y, return_counts=True)
        cv = max(2, min(cv, *counts))

    return sklearn_check_cv(cv, y, classifier)


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


def _is_higher_better(metric: str) -> bool:
    return metric in ["auc"]


class _LightGBMExtractionCallback(object):
    def __init__(self) -> None:
        self._best_iteration = None  # type: Optional[int]
        self._boosters = None  # type: Optional[List[lgb.Booster]]

    def __call__(self, env: LightGBMCallbackEnvType) -> None:
        self._best_iteration = env.iteration + 1
        self._boosters = env.model.boosters


class _Objective(object):
    def __init__(
        self,
        params: Dict[str, Any],
        dataset: lgb.Dataset,
        eval_name: str,
        is_higher_better: bool,
        n_samples: int,
        callbacks: Optional[List[Callable]] = None,
        categorical_feature: Union[List[int], List[str], str] = "auto",
        cv: Optional[CVType] = None,
        early_stopping_rounds: Optional[int] = None,
        enable_pruning: bool = False,
        feature_name: Union[List[str], str] = "auto",
        feval: Optional[Callable] = None,
        fobj: Optional[Callable] = None,
        init_model: Optional[Union[lgb.Booster, lgb.LGBMModel, str]] = None,
        n_estimators: int = 100,
        param_distributions: Optional[Dict[str, distributions.BaseDistribution]] = None,
    ) -> None:
        self.callbacks = callbacks
        self.categorical_feature = categorical_feature
        self.cv = cv
        self.dataset = dataset
        self.early_stopping_rounds = early_stopping_rounds
        self.enable_pruning = enable_pruning
        self.eval_name = eval_name
        self.feature_name = feature_name
        self.feval = feval
        self.fobj = fobj
        self.init_model = init_model
        self.is_higher_better = is_higher_better
        self.n_estimators = n_estimators
        self.n_samples = n_samples
        self.params = params
        self.param_distributions = param_distributions

    def __call__(self, trial: trial_module.Trial) -> float:
        params = self._get_params(trial)  # type: Dict[str, Any]
        dataset = copy.copy(self.dataset)
        callbacks = self._get_callbacks(trial)  # type: List[Callable]
        eval_hist = lgb.cv(
            params,
            dataset,
            callbacks=callbacks,
            categorical_feature=self.categorical_feature,
            early_stopping_rounds=self.early_stopping_rounds,
            feature_name=self.feature_name,
            feval=self.feval,
            fobj=self.fobj,
            folds=self.cv,
            init_model=self.init_model,
            num_boost_round=self.n_estimators,
        )  # Dict[str, List[float]]
        best_iteration = callbacks[0]._best_iteration  # type: ignore

        trial.set_user_attr("best_iteration", best_iteration)

        value = eval_hist["{}-mean".format(self.eval_name)][-1]  # type: float
        is_best_trial = True  # type: bool

        try:
            is_best_trial = (value < trial.study.best_value) ^ self.is_higher_better
        except ValueError:
            pass

        if is_best_trial:
            boosters = callbacks[0]._boosters  # type: ignore
            representations = []  # type: List[str]

            for b in boosters:
                b.free_dataset()
                representations.append(b.model_to_string())

            trial.study.set_user_attr("representations", representations)

        return value

    def _get_callbacks(self, trial: trial_module.Trial) -> List[Callable]:
        extraction_callback = _LightGBMExtractionCallback()  # type: _LightGBMExtractionCallback
        callbacks = [extraction_callback]  # type: List[Callable]

        if self.enable_pruning:
            pruning_callback = integration.LightGBMPruningCallback(
                trial, self.eval_name
            )  # type: integration.LightGBMPruningCallback

            callbacks.append(pruning_callback)

        if self.callbacks is not None:
            callbacks += self.callbacks

        return callbacks

    def _get_params(self, trial: trial_module.Trial) -> Dict[str, Any]:
        params = self.params.copy()  # type: Dict[str, Any]

        if self.param_distributions is None:
            params["feature_fraction"] = trial.suggest_discrete_uniform(
                "feature_fraction", 0.1, 1.0, 0.05
            )
            params["max_depth"] = trial.suggest_int("max_depth", 1, 7)
            params["num_leaves"] = trial.suggest_int("num_leaves", 2, 2 ** params["max_depth"])
            # See https://github.com/Microsoft/LightGBM/issues/907
            params["min_data_in_leaf"] = trial.suggest_int(
                "min_data_in_leaf", 1, max(1, int(self.n_samples / params["num_leaves"])),
            )
            params["lambda_l1"] = trial.suggest_loguniform("lambda_l1", 1e-09, 10.0)
            params["lambda_l2"] = trial.suggest_loguniform("lambda_l2", 1e-09, 10.0)

            if params["boosting_type"] != "goss":
                params["bagging_fraction"] = trial.suggest_discrete_uniform(
                    "bagging_fraction", 0.5, 0.95, 0.05
                )
                params["bagging_freq"] = trial.suggest_int("bagging_freq", 1, 10)

            return params

        for name, distribution in self.param_distributions.items():
            params[name] = trial._suggest(name, distribution)

        return params


class _VotingBooster(object):
    @property
    def feature_name(self) -> List[str]:
        return self.boosters[0].feature_name

    def __init__(self, boosters: List[lgb.Booster], weights: Optional[np.ndarray] = None) -> None:
        self.boosters = boosters
        self.weights = weights

    @classmethod
    def from_representations(
        cls, representations: List[str], weights: Optional[np.ndarray] = None
    ) -> "_VotingBooster":
        if lgb.__version__ >= "2.3":
            boosters = [
                lgb.Booster(model_str=model_str, silent=True) for model_str in representations
            ]
        else:
            boosters = [
                lgb.Booster(params={"model_str": model_str}) for model_str in representations
            ]

        return cls(boosters, weights=weights)

    def feature_importance(self, **kwargs: Any) -> np.ndarray:
        results = [b.feature_importance(**kwargs) for b in self.boosters]

        return np.average(results, axis=0, weights=self.weights)

    def predict(self, X: TwoDimArrayLikeType, **kwargs: Any) -> np.ndarray:
        results = [b.predict(X, **kwargs) for b in self.boosters]

        return np.average(results, axis=0, weights=self.weights)


class LGBMModel(lgb.LGBMModel):
    """Base class for models."""

    @property
    def best_index_(self) -> int:
        """Get the best trial's number."""
        self._check_is_fitted()

        return self.study_.best_trial.number

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
        refit: bool = True,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        random_state: Optional[RandomStateType] = None,
        n_jobs: int = -1,
        importance_type: str = "split",
        cv: CVType = 5,
        enable_pruning: bool = False,
        n_trials: int = 40,
        param_distributions: Optional[Dict[str, distributions.BaseDistribution]] = None,
        study: Optional[study_module.Study] = None,
        timeout: Optional[float] = None,
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
            subsample=subsample,
            subsample_for_bin=subsample_for_bin,
            subsample_freq=subsample_freq,
            **kwargs
        )

        self.cv = cv
        self.enable_pruning = enable_pruning
        self.n_trials = n_trials
        self.param_distributions = param_distributions
        self.refit = refit
        self.study = study
        self.timeout = timeout

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

    def _get_random_state(self) -> Optional[int]:
        if self.random_state is None or isinstance(self.random_state, int):
            return self.random_state

        random_state = check_random_state(self.random_state)

        return random_state.randint(0, MAX_INT)

    def _make_booster(
        self,
        params: Dict[str, Any],
        dataset: lgb.Dataset,
        representations: List[str],
        num_boost_round: int,
        folds: List[Tuple],
        fobj: Optional[Callable] = None,
        feature_name: Union[List[str], str] = "auto",
        categorical_feature: Union[List[int], List[str], str] = "auto",
        callbacks: Optional[List[Callable]] = None,
        init_model: Optional[Union[lgb.Booster, lgb.LGBMModel, str]] = None,
    ) -> Union[_VotingBooster, lgb.Booster]:
        if self.refit:
            booster = lgb.train(
                params,
                dataset,
                num_boost_round=num_boost_round,
                fobj=fobj,
                feature_name=feature_name,
                categorical_feature=categorical_feature,
                callbacks=callbacks,
                init_model=init_model,
            )

            booster.free_dataset()

            return booster

        sample_weight = dataset.get_weight()
        weights = np.array([np.sum(sample_weight[train]) for train, _ in folds])

        booster = _VotingBooster.from_representations(representations, weights=weights)

        return booster

    def fit(
        self,
        X: TwoDimArrayLikeType,
        y: OneDimArrayLikeType,
        sample_weight: Optional[OneDimArrayLikeType] = None,
        group: Optional[OneDimArrayLikeType] = None,
        eval_metric: Optional[Union[Callable, str]] = None,
        early_stopping_rounds: Optional[int] = 10,
        feature_name: Union[List[str], str] = "auto",
        categorical_feature: Union[List[int], List[str], str] = "auto",
        callbacks: Optional[List[Callable]] = None,
        init_model: Optional[Union[lgb.Booster, lgb.LGBMModel, str]] = None,
        groups: Optional[OneDimArrayLikeType] = None,
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

            group:
                Group data of training data.

            eval_metric:
                Evaluation metric. See
                https:/lightgbm.readthedocs.io/en/latest/Parameters.html#metric.

            early_stopping_rounds
                Used to activate early stopping. The model will train until the
                validation score stops improving.

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

            groups:
                Group labels for the samples used while splitting the dataset
                into traintest set. If ``group`` is not None, this parameter is
                ignored.

            **fit_params:
                Always ignored. This parameter exists for compatibility.

        Returns:
            self:
                Return self.
        """
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

        is_classifier = self._estimator_type == "classifier"
        cv = check_cv(self.cv, y, is_classifier)

        seed = self._get_random_state()

        params = self.get_params()

        for attr in (
            "class_weight",
            "cv",
            "enable_pruning",
            "importance_type",
            "n_estimators",
            "n_trials",
            "param_distributions",
            "refit",
            "study",
            "timeout",
        ):
            params.pop(attr, None)

        params["objective"] = self._get_objective()
        params["random_state"] = seed
        params["verbose"] = -1

        if self._n_classes is not None and self._n_classes > 2:
            params["num_classes"] = self._n_classes

        if callable(eval_metric):
            params["metric"] = "None"
            feval = _EvalFunctionWrapper(eval_metric)
            eval_name, _, is_higher_better = eval_metric(y, y)

        else:
            if eval_metric is None:
                params["metric"] = OBJECTIVE2METRIC[params["objective"]]
            else:
                params["metric"] = eval_metric

            feval = None
            eval_name = params["metric"]
            is_higher_better = _is_higher_better(params["metric"])

        if callable(self.objective):
            fobj = _ObjectiveFunctionWrapper(self.objective)
        else:
            fobj = None

        if isinstance(init_model, lgb.LGBMModel):
            init_model = init_model.booster_

        if self.study is None:
            sampler = samplers.TPESampler(seed=seed)

            self.study_ = study_module.create_study(
                direction="maximize" if is_higher_better else "minimize", sampler=sampler,
            )

        else:
            self.study_ = self.study

        # See https://github.com/microsoft/LightGBM/issues/2319
        if group is None and groups is not None:
            indices = np.argsort(groups)
            X = _safe_indexing(X, indices)
            y = _safe_indexing(y, indices)
            sample_weight = _safe_indexing(sample_weight, indices)
            groups = _safe_indexing(groups, indices)
            _, group = np.unique(groups, return_counts=True)

        dataset = lgb.Dataset(X, label=y, group=group, weight=sample_weight)

        objective = _Objective(
            params,
            dataset,
            eval_name,
            is_higher_better,
            n_samples,
            callbacks=callbacks,
            categorical_feature=categorical_feature,
            cv=cv,
            early_stopping_rounds=early_stopping_rounds,
            enable_pruning=self.enable_pruning,
            feature_name=feature_name,
            feval=feval,
            fobj=fobj,
            init_model=init_model,
            n_estimators=self.n_estimators,
            param_distributions=self.param_distributions,
        )

        logger = logging.get_logger(__name__)

        logger.info("Searching the best hyperparameters...")

        start_time = time.perf_counter()

        self.study_.optimize(objective, catch=(), n_trials=self.n_trials, timeout=self.timeout)

        elapsed_time = time.perf_counter() - start_time

        self._best_iteration = self.study_.best_trial.user_attrs["best_iteration"]
        self._best_score = self.study_.best_value
        self._objective = params["objective"]
        self.best_params_ = {**params, **self.study_.best_params}
        self.n_splits_ = cv.get_n_splits(X, y, groups=groups)

        logger.info(
            "Finished hyperparemeter search! "
            "(elapsed time: {:.3f} sec.) "
            "The best_iteration is {}.".format(elapsed_time, self._best_iteration)
        )

        folds = cv.split(X, y, groups=groups)
        representations = self.study_.user_attrs["representations"]

        logger.info("Making booster(s)...")

        start_time = time.perf_counter()

        self._Booster = self._make_booster(
            self.best_params_,
            dataset,
            representations,
            self._best_iteration,
            folds,
            fobj=fobj,
            feature_name=feature_name,
            categorical_feature=categorical_feature,
            callbacks=callbacks,
            init_model=init_model,
        )

        elapsed_time = time.perf_counter() - start_time

        logger.info(
            "Finished making booster(s)! " "(elapsed time: {:.3f} sec.)".format(elapsed_time)
        )

        if self.refit:
            self.refit_time_ = elapsed_time

        return self

    def predict(
        self, X: TwoDimArrayLikeType, num_iteration: Optional[int] = None, **predict_params: Any
    ) -> np.ndarray:
        """Predict using the fitted model.

        Args:
            X:
                Data.

            num_iteration:
                Limit number of iterations in the prediction. If None, if the
                best iteration exists, it is used; otherwise, all trees are
                used. If <=0, all trees are used (no limits).

            **predict_params:
                Always ignored. This parameter exists for compatibility.

        Returns:
            y_pred:
                Predicted values.
        """
        X = check_X(X, accept_sparse=True, estimator=self, force_all_finite=False)

        return self.booster_.predict(X, num_iteration=num_iteration)


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

        importance_type:
            Type of feature importances. If 'split', result contains numbers of
            times the feature is used in a model. If 'gain', result contains
            total gains of splits which use the feature.

        cv:
            Cross-validation strategy. Possible inputs for cv are:

            - integer to specify the number of folds in a CV splitter,
            - a CV splitter,
            - an iterable yielding (train, test) splits as arrays of indices.

            If int, ``sklearn.model_selection.StratifiedKFold`` is used.

        enable_pruning:
            If True, pruning is performed.

        n_trials:
            Number of trials. If None, there is no limitation on the number of
            trials. If ``timeout`` is also set to None, the study continues to
            create trials until it receives a termination signal such as Ctrl+C
            or SIGTERM. This trades off runtime vs quality of the solution.

        param_distributions:
            Dictionary where keys are parameters and values are distributions.
            Distributions are assumed to implement the optuna distribution
            interface. If None, ``num_leaves``, ``max_depth``,
            ``min_child_samples``, ``subsample``, ``subsample_freq``,
            ``colsample_bytree``, ``reg_alpha`` and ``reg_lambda`` are
            searched.

        refit:
            If True, refit the estimator with the best found hyperparameters.

        study:
            Study corresponds to the optimization task. If None, a new study is
            created.

        timeout:
            Time limit in seconds for the search of appropriate models. If
            None, the study is executed without time limitation. If
            ``n_trials`` is also set to None, the study continues to create
            trials until it receives a termination signal such as Ctrl+C or
            SIGTERM. This trades off runtime vs quality of the solution.

        **kwargs:
            Other parameters for the model. See
            http:/lightgbm.readthedocs.io/en/latest/Parameters.html for more
            parameters. Note, that **kwargs is not supported in sklearn, so it
            may cause unexpected issues.

    Attributes:
        best_iteration_:
            Number of iterations as selected by early stopping.

        best_params_:
            Parameters of the best trial in the ``Study``.

        best_score_:
            Mean cross-validated score of the best estimator.

        booster_:
            Trained booster.

        encoder_:
            Label encoder.

        n_features_:
            Number of features of fitted model.

        n_splits_:
            Number of cross-validation splits.

        objective_:
            Concrete objective used while fitting this model.

        study_:
            Actual study.

        refit_time_:
            Time for refitting the best estimator. This is present only if
            ``refit`` is set to True.

    Examples:
        >>> import optuna.integration.lightgbm as lgb
        >>> from sklearn.datasets import load_iris
        >>> clf = lgb.LGBMClassifier(random_state=0)
        >>> X, y = load_iris(return_X_y=True)
        >>> clf.fit(X, y)
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
        group: Optional[OneDimArrayLikeType] = None,
        eval_metric: Optional[Union[Callable, str]] = None,
        early_stopping_rounds: Optional[int] = 10,
        feature_name: Union[List[str], str] = "auto",
        categorical_feature: Union[List[int], List[str], str] = "auto",
        callbacks: Optional[List[Callable]] = None,
        init_model: Optional[Union[lgb.Booster, lgb.LGBMModel, str]] = None,
        groups: Optional[OneDimArrayLikeType] = None,
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
            group=group,
            eval_metric=eval_metric,
            early_stopping_rounds=early_stopping_rounds,
            feature_name=feature_name,
            categorical_feature=categorical_feature,
            callbacks=callbacks,
            init_model=init_model,
            groups=groups,
            **fit_params
        )

    fit.__doc__ = LGBMModel.fit.__doc__

    def predict(
        self, X: TwoDimArrayLikeType, num_iteration: Optional[int] = None, **predict_params: Any
    ) -> np.ndarray:
        """Docstring is inherited from the LGBMModel."""
        probas = self.predict_proba(X, num_iteration=num_iteration, **predict_params)
        class_index = np.argmax(probas, axis=1)

        return self.encoder_.inverse_transform(class_index)

    predict.__doc__ = LGBMModel.predict.__doc__

    def predict_proba(
        self, X: TwoDimArrayLikeType, num_iteration: Optional[int] = None, **predict_params: Any
    ) -> np.ndarray:
        """Predict class probabilities for data.

        Args:
            X:
                Data.

            num_iteration:
                Limit number of iterations in the prediction. If None, if the
                best iteration exists, it is used; otherwise, all trees are
                used. If <=0, all trees are used (no limits).

            **predict_params:
                Always ignored. This parameter exists for compatibility.

        Returns:
            p:
                Class probabilities of data.
        """
        preds = super().predict(X, num_iteration=num_iteration, **predict_params)

        if self._n_classes > 2:
            return preds

        else:
            preds = preds.reshape(-1, 1)

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

        importance_type:
            Type of feature importances. If 'split', result contains numbers of
            times the feature is used in a model. If 'gain', result contains
            total gains of splits which use the feature.

        cv:
            Cross-validation strategy. Possible inputs for cv are:

            - integer to specify the number of folds in a CV splitter,
            - a CV splitter,
            - an iterable yielding (train, test) splits as arrays of indices.

            If int, ``sklearn.model_selection.StratifiedKFold`` is used.

        enable_pruning:
            If True, pruning is performed.

        n_trials:
            Number of trials. If None, there is no limitation on the number of
            trials. If ``timeout`` is also set to None, the study continues to
            create trials until it receives a termination signal such as Ctrl+C
            or SIGTERM. This trades off runtime vs quality of the solution.

        param_distributions:
            Dictionary where keys are parameters and values are distributions.
            Distributions are assumed to implement the optuna distribution
            interface. If None, ``num_leaves``, ``max_depth``,
            ``min_child_samples``, ``subsample``, ``subsample_freq``,
            ``colsample_bytree``, ``reg_alpha`` and ``reg_lambda`` are
            searched.

        refit:
            If True, refit the estimator with the best found hyperparameters.

        study:
            Study corresponds to the optimization task. If None, a new study is
            created.

        timeout:
            Time limit in seconds for the search of appropriate models. If
            None, the study is executed without time limitation. If
            ``n_trials`` is also set to None, the study continues to create
            trials until it receives a termination signal such as Ctrl+C or
            SIGTERM. This trades off runtime vs quality of the solution.

        **kwargs:
            Other parameters for the model. See
            http:/lightgbm.readthedocs.io/en/latest/Parameters.html for more
            parameters. Note, that **kwargs is not supported in sklearn, so it
            may cause unexpected issues.

    Attributes:
        best_iteration_:
            Number of iterations as selected by early stopping.

        best_params_:
            Parameters of the best trial in the ``Study``.

        best_score_:
            Mean cross-validated score of the best estimator.

        booster_:
            Trained booster.

        n_features_:
            Number of features of fitted model.

        n_splits_:
            Number of cross-validation splits.

        objective_:
            Concrete objective used while fitting this model.

        study_:
            Actual study.

        refit_time_:
            Time for refitting the best estimator. This is present only if
            ``refit`` is set to True.

    Examples:
        >>> import optuna.integration.lightgbm as lgb
        >>> from sklearn.datasets import load_boston
        >>> reg = lgb.LGBMRegressor(random_state=0)
        >>> X, y = load_boston(return_X_y=True)
        >>> reg.fit(X, y)
        LGBMRegressor(...)
        >>> y_pred = reg.predict(X)
    """
