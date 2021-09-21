from logging import DEBUG
from logging import INFO
from logging import WARNING
from numbers import Integral
from numbers import Number
from time import time
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Mapping
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import scipy as sp
from scipy.sparse import spmatrix

from optuna import distributions
from optuna import logging
from optuna import samplers
from optuna import study as study_module
from optuna import TrialPruned
from optuna._experimental import experimental
from optuna._imports import try_import
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import Trial


with try_import() as _imports:
    import pandas as pd
    import sklearn
    from sklearn.base import BaseEstimator
    from sklearn.base import clone
    from sklearn.base import is_classifier
    from sklearn.metrics import check_scoring
    from sklearn.metrics._scorer import _check_multimetric_scoring
    from sklearn.model_selection import BaseCrossValidator
    from sklearn.model_selection import check_cv
    from sklearn.model_selection import cross_validate
    from sklearn.model_selection._validation import _score
    from sklearn.utils import check_random_state
    from sklearn.utils.metaestimators import _safe_split

    if sklearn.__version__ >= "0.22":
        from sklearn.utils import _safe_indexing as sklearn_safe_indexing
    else:
        from sklearn.utils import safe_indexing as sklearn_safe_indexing
    from sklearn.utils.validation import check_is_fitted

if not _imports.is_successful():
    BaseEstimator = object  # NOQA

ArrayLikeType = Union[List, np.ndarray, "pd.Series", spmatrix]
OneDimArrayLikeType = Union[List[float], np.ndarray, "pd.Series"]
TwoDimArrayLikeType = Union[List[List[float]], np.ndarray, "pd.DataFrame", spmatrix]
IterableType = Union[List, "pd.DataFrame", np.ndarray, "pd.Series", spmatrix, None]
IndexableType = Union[Iterable, None]

_logger = logging.get_logger(__name__)


def _check_fit_params(
    X: TwoDimArrayLikeType, fit_params: Dict, indices: OneDimArrayLikeType
) -> Dict:

    fit_params_validated = {}
    for key, value in fit_params.items():

        # NOTE Original implementation:
        # https://github.com/scikit-learn/scikit-learn/blob/ \
        # 2467e1b84aeb493a22533fa15ff92e0d7c05ed1c/sklearn/utils/validation.py#L1324-L1328
        # Scikit-learn does not accept non-iterable inputs.
        # This line is for keeping backward compatibility.
        # (See: https://github.com/scikit-learn/scikit-learn/issues/15805)
        if not _is_arraylike(value) or _num_samples(value) != _num_samples(X):
            fit_params_validated[key] = value
        else:
            fit_params_validated[key] = _make_indexable(value)
            fit_params_validated[key] = _safe_indexing(fit_params_validated[key], indices)
    return fit_params_validated


# NOTE Original implementation:
# https://github.com/scikit-learn/scikit-learn/blob/ \
# 8caa93889f85254fc3ca84caa0a24a1640eebdd1/sklearn/utils/validation.py#L131-L135
def _is_arraylike(x: Any) -> bool:

    return hasattr(x, "__len__") or hasattr(x, "shape") or hasattr(x, "__array__")


# NOTE Original implementation:
# https://github.com/scikit-learn/scikit-learn/blob/ \
# 8caa93889f85254fc3ca84caa0a24a1640eebdd1/sklearn/utils/validation.py#L217-L234
def _make_indexable(iterable: IterableType) -> IndexableType:

    tocsr_func = getattr(iterable, "tocsr", None)
    if tocsr_func is not None and sp.sparse.issparse(iterable):
        return tocsr_func(iterable)
    elif hasattr(iterable, "__getitem__") or hasattr(iterable, "iloc"):
        return iterable
    elif iterable is None:
        return iterable
    return np.array(iterable)


def _num_samples(x: ArrayLikeType) -> int:

    # NOTE For dask dataframes
    # https://github.com/scikit-learn/scikit-learn/blob/ \
    # 8caa93889f85254fc3ca84caa0a24a1640eebdd1/sklearn/utils/validation.py#L155-L158
    x_shape = getattr(x, "shape", None)
    if x_shape is not None:
        if isinstance(x_shape[0], Integral):
            return int(x_shape[0])

    try:
        return len(x)
    except TypeError:
        raise TypeError("Expected sequence or array-like, got %s." % type(x)) from None


def _safe_indexing(
    X: Union[OneDimArrayLikeType, TwoDimArrayLikeType], indices: OneDimArrayLikeType
) -> Union[OneDimArrayLikeType, TwoDimArrayLikeType]:

    if X is None:
        return X

    return sklearn_safe_indexing(X, indices)


class _MultiMetricMixin:
    """A mixin class for utilities.

    As per optuna implementation, some functionalities that
    ``sklearn.model_selection.BaseSearchCV`` provides for multi-metric evaluation
    are shared by ``_Objective`` and :class:`~optuna.integration.OptunaSearchCV`. Those functions are
    organized in this mixin class.
    """

    # NOTE Original implementation:
    # https://github.com/scikit-learn/scikit-learn/blob/ \
    # 2beed5584/sklearn/model_selection/_search.py#L706-L721
    @classmethod
    def _check_refit_for_multimetric(cls, scores: Union[Callable[..., float], Callable[..., Dict[str, float]]], refit: Union[bool, str, Callable]) -> None:
        """Check `refit` is compatible with `scores` is valid"""
        multimetric_refit_msg = (
            "For multi-metric scoring, the parameter refit must be set to a "
            "scorer key or a callable to refit an estimator with the best "
            "parameter setting on the whole data and make the best_* "
            "attributes available for that metric. If this is not needed, "
            f"refit should be set to False explicitly. {refit!r} was "
            "passed."
        )

        valid_refit_dict = isinstance(refit, str) and (
            refit in scores or f"test_{refit}" in scores
        )

        if refit is not False and not valid_refit_dict and not callable(refit):
            raise ValueError(multimetric_refit_msg)


class _Objective(MultiMetricMixin, object):
    """Callable that implements objective function.

    Args:
        estimator:
            Object to use to fit the data. This is assumed to implement the
            scikit-learn estimator interface. Either this needs to provide
            ``score``, or ``scoring`` must be passed.

        param_distributions:
            Dictionary where keys are parameters and values are distributions.
            Distributions are assumed to implement the optuna distribution
            interface.

        X:
            Training data.

        y:
            Target variable.

        cv:
            Cross-validation strategy.

        enable_pruning:
            If :obj:`True`, pruning is performed in the case where the
            underlying estimator supports ``partial_fit``.

        error_score:
            Value to assign to the score if an error occurs in fitting. If
            'raise', the error is raised. If numeric,
            ``sklearn.exceptions.FitFailedWarning`` is raised. This does not
            affect the refit step, which will always raise the error.

        fit_params:
            Parameters passed to ``fit`` one the estimator.

        groups:
            Group labels for the samples used while splitting the dataset into
            train/validation set.

        max_iter:
            Maximum number of epochs. This is only used if the underlying
            estimator supports ``partial_fit``.

        refit:
            If :obj:`True`, refit the estimator with the best found
            hyperparameters. The refitted estimator is made available at the
            ``best_estimator_`` attribute and permits using ``predict``
            directly.

            For multiple metric evaluation, this needs to be a :class:`str` denoting
            the scorer that would be used to find the best parameters for
            refitting the estimator at the end.

        return_train_score:
            If :obj:`True`, training scores will be included. Computing
            training scores is used to get insights on how different
            hyperparameter settings impact the overfitting/underfitting
            trade-off. However computing training scores can be
            computationally expensive and is not strictly required to select
            the hyperparameters that yield the best generalization
            performance.

        scoring:
            Scorer function or a dict which maps the scorer key to the scorer
            callable.
    """

    def __init__(
        self,
        estimator: "BaseEstimator",
        param_distributions: Mapping[str, distributions.BaseDistribution],
        X: TwoDimArrayLikeType,
        y: Optional[Union[OneDimArrayLikeType, TwoDimArrayLikeType]],
        cv: "BaseCrossValidator",
        enable_pruning: bool,
        error_score: Union[Number, float, str],
        fit_params: Dict[str, Any],
        groups: Optional[OneDimArrayLikeType],
        max_iter: int,
        refit: Union[bool, str, Callable],
        return_train_score: bool,
        scoring: Optional[
            Union[
                str,
                Callable[..., float],
                List[str],
                Tuple[str],
                Callable[..., Dict[str, float]],
                Dict[str, Callable[..., float]],
            ]
        ] = None,
    ) -> None:

        self.cv = cv
        self.enable_pruning = enable_pruning
        self.error_score = error_score
        self.estimator = estimator
        self.fit_params = fit_params
        self.groups = groups
        self.max_iter = max_iter
        self.param_distributions = param_distributions
        self.refit = refit
        self.return_train_score = return_train_score
        self.scoring = scoring
        self.X = X
        self.y = y

    def __call__(self, trial: Trial) -> float:

        estimator = clone(self.estimator)
        params = self._get_params(trial)

        estimator.set_params(**params)

        if self.enable_pruning:
            scores = self._cross_validate_with_pruning(trial, estimator)
        else:
            scores = cross_validate(
                estimator,
                self.X,
                self.y,
                cv=self.cv,
                error_score=self.error_score,
                fit_params=self.fit_params,
                groups=self.groups,
                return_train_score=self.return_train_score,
                scoring=self.scoring,
            )
            self.multimetric_ = "test_score" not in scores

            # check refit_metric now for a callabe scorer that is multimetric
            if callable(self.scoring) and self.multimetric_:
                self._check_refit_for_multimetric(scores, self.refit)

        self._store_scores(trial, scores)

        if self.multimetric_:
            return trial.user_attrs[f"mean_test_{self.refit}"]
        else:
            return trial.user_attrs["mean_test_score"]

    def _cross_validate_with_pruning(
        self, trial: Trial, estimator: "BaseEstimator"
    ) -> Mapping[str, OneDimArrayLikeType]:

        if is_classifier(estimator):
            partial_fit_params = self.fit_params.copy()
            classes = np.unique(self.y)

            partial_fit_params.setdefault("classes", classes)

        else:
            partial_fit_params = self.fit_params

        n_splits = self.cv.get_n_splits(self.X, self.y, groups=self.groups)
        estimators = [clone(estimator) for _ in range(n_splits)]
        scores = {
            "fit_time": np.zeros(n_splits),
            "score_time": np.zeros(n_splits),
        }

        for step in range(self.max_iter):
            for i, (train, test) in enumerate(self.cv.split(self.X, self.y, groups=self.groups)):
                out = self._partial_fit_and_score(estimators[i], train, test, partial_fit_params)

                # NOTE Original implementaiton:
                # https://github.com/scikit-learn/scikit-learn/blob/ \
                # 642127806a830346886a0337fcbefedc871159c0/sklearn/model_selection/_search.py#L844-L852

                # multimetric is determined here because in the case of a callable
                # self.scoring the return type is only known after calling
                self.multimetric_ = isinstance(out[0], dict)

                # Initialize scores if necessary
                if not self.multimetric_:
                    if "test_score" not in scores:
                        scores["test_score"] = np.empty(n_splits)
                    if self.return_train_score and "train_score" not in scores:
                        scores["train_score"] = np.empty(n_splits)
                else:
                    metricnames = list(out[0].keys())
                    for metricname in metricnames:
                        if f"test_{metricname}" not in scores:
                            scores[f"test_{metricname}"] = np.empty(n_splits)
                        if self.return_train_score and f"train_{metricname}" not in scores:
                            scores[f"train_{metricname}"] = np.empty(n_splits)

                if self.return_train_score:
                    if not self.multimetric_:
                        scores["train_score"][i] = out.pop(0)
                    else:
                        out_pop0 = out.pop(0)
                        for metricname in metricnames:
                            scores[f"train_{metricname}"][i] = out_pop0[metricname]

                if not self.multimetric_:
                    scores["test_score"][i] = out[0]
                else:
                    for metricname in metricnames:
                        scores[f"train_{metricname}"][i] = out[0][metricname]
                scores["fit_time"][i] += out[1]
                scores["score_time"][i] += out[2]

                # NOTE Original implementaiton:
                # https://github.com/scikit-learn/scikit-learn/blob/ \
                # 642127806a830346886a0337fcbefedc871159c0/sklearn/model_selection/_search.py#L844-L852

                # check refit_metric now for a callabe scorer that is multimetric
                if callable(self.scoring) and self.multimetric_:
                    self._check_refit_for_multimetric(out[0], self.refit)

            if self.multimetric_:
                intermediate_value = np.nanmean(scores[f"test_{self.refit}"])
            else:
                intermediate_value = np.nanmean(scores["test_score"])

            trial.report(intermediate_value, step=step)

            if trial.should_prune():
                self._store_scores(trial, scores)

                raise TrialPruned("trial was pruned at iteration {}.".format(step))

        return scores

    def _get_params(self, trial: Trial) -> Dict[str, Any]:

        return {
            name: trial._suggest(name, distribution)
            for name, distribution in self.param_distributions.items()
        }

    def _partial_fit_and_score(
        self,
        estimator: "BaseEstimator",
        train: List[int],
        test: List[int],
        partial_fit_params: Dict[str, Any],
    ) -> List[Union[Number, Dict]]:

        X_train, y_train = _safe_split(estimator, self.X, self.y, train)
        X_test, y_test = _safe_split(estimator, self.X, self.y, test, train_indices=train)

        start_time = time()

        # NOTE Original Implementation:
        # https://github.com/scikit-learn/scikit-learn/blob/ \
        # 642127806a830346886a0337fcbefedc871159c0/sklearn/model_selection/_validation.py#L677-L706
        try:
            estimator.partial_fit(X_train, y_train, **partial_fit_params)

        except Exception as e:
            fit_time = time() - start_time
            score_time = 0.0
            if self.error_score == "raise":
                raise e
            elif isinstance(self.error_score, Number):
                if isinstance(self.scoring, dict):
                    test_scores: Union[Number, Dict[str, Number]] = {name: self.error_score for name in self.scoring}
                    if self.return_train_score:
                        train_scores = test_scores.copy()
                else:
                    test_scores = self.error_score
                    if self.return_train_score:
                        train_scores = self.error_score
            else:
                raise ValueError("error_score must be 'raise' or numeric.") from e

        else:
            fit_time = time() - start_time
            test_scores = _score(estimator, X_test, y_test, self.scoring, self.error_score)
            score_time = time() - fit_time - start_time

            if self.return_train_score:
                train_scores = _score(estimator, X_train, y_train, self.scoring, self.error_score)

        # Required for type checking but is never expected to fail.
        assert isinstance(fit_time, Number)
        assert isinstance(score_time, Number)

        ret = [test_scores, fit_time, score_time]

        if self.return_train_score:
            ret.insert(0, train_scores)

        return ret

    def _store_scores(self, trial: Trial, scores: Mapping[str, OneDimArrayLikeType]) -> None:
        for name, array in scores.items():
            if name.startswith("test_") or name.startswith("train_"):
                for i, score in enumerate(array):
                    trial.set_user_attr("split{}_{}".format(i, name), score)

            trial.set_user_attr("mean_{}".format(name), np.nanmean(array))
            trial.set_user_attr("std_{}".format(name), np.nanstd(array))


@experimental("0.17.0")
class OptunaSearchCV(BaseEstimator, MultiMetricMixin):
    """Hyperparameter search with cross-validation.

    Args:
        estimator:
            Object to use to fit the data. This is assumed to implement the
            scikit-learn estimator interface. Either this needs to provide
            ``score``, or ``scoring`` must be passed.

        param_distributions:
            Dictionary where keys are parameters and values are distributions.
            Distributions are assumed to implement the optuna distribution
            interface.

        cv:
            Cross-validation strategy. Possible inputs for cv are:

            - integer to specify the number of folds in a CV splitter,
            - a CV splitter,
            - an iterable yielding (train, validation) splits as arrays of indices.

            For integer, if :obj:`estimator` is a classifier and :obj:`y` is
            either binary or multiclass,
            ``sklearn.model_selection.StratifiedKFold`` is used. otherwise,
            ``sklearn.model_selection.KFold`` is used.

        enable_pruning:
            If :obj:`True`, pruning is performed in the case where the
            underlying estimator supports ``partial_fit``.

        error_score:
            Value to assign to the score if an error occurs in fitting. If
            'raise', the error is raised. If numeric,
            ``sklearn.exceptions.FitFailedWarning`` is raised. This does not
            affect the refit step, which will always raise the error.

        max_iter:
            Maximum number of epochs. This is only used if the underlying
            estimator supports ``partial_fit``.

        n_jobs:
            Number of :obj:`threading` based parallel jobs. :obj:`-1` means
            using the number is set to CPU count.

                .. note::
                    ``n_jobs`` allows parallelization using :obj:`threading` and may suffer from
                    `Python's GIL <https://wiki.python.org/moin/GlobalInterpreterLock>`_.
                    It is recommended to use :ref:`process-based parallelization<distributed>`
                    if ``func`` is CPU bound.

                .. warning::
                    Deprecated in v2.7.0. This feature will be removed in the future.
                    It is recommended to use :ref:`process-based parallelization<distributed>`.
                    The removal of this feature is currently scheduled for v4.0.0, but this
                    schedule is subject to change.
                    See https://github.com/optuna/optuna/releases/tag/v2.7.0.

        n_trials:
            Number of trials. If :obj:`None`, there is no limitation on the
            number of trials. If :obj:`timeout` is also set to :obj:`None`,
            the study continues to create trials until it receives a
            termination signal such as Ctrl+C or SIGTERM. This trades off
            runtime vs quality of the solution.

        random_state:
            Seed of the pseudo random number generator. If int, this is the
            seed used by the random number generator. If
            ``numpy.random.RandomState`` object, this is the random number
            generator. If :obj:`None`, the global random state from
            ``numpy.random`` is used.

        refit:
            If :obj:`True`, refit the estimator with the best found
            hyperparameters. The refitted estimator is made available at the
            ``best_estimator_`` attribute and permits using ``predict``
            directly.

            For multiple metric evaluation, this needs to be a :class:`str` denoting
            the scorer that would be used to find the best parameters for
            refitting the estimator at the end.

        return_train_score:
            If :obj:`True`, training scores will be included. Computing
            training scores is used to get insights on how different
            hyperparameter settings impact the overfitting/underfitting
            trade-off. However computing training scores can be
            computationally expensive and is not strictly required to select
            the hyperparameters that yield the best generalization
            performance.

        scoring:
            String, callable, list, tuple or dict to evaluate the
            predictions on the validation data.

            If ``scoring`` represents a single score, one can use:
                - a single string;
                - a callable that returns a single value.

            If ``scoring`` represents multiple scores, one can use:
                - a list or tuple of unique strings;
                - a callable returning a dictionary where the keys are the
                    metric names and the values are the metric scores;
                - a dictionary with metric names as keys and callables a
                values.

            If :obj:`None`, ``score`` on the estimator is used.

                .. note::
                    Even when `scoring` represents multiple scores, the
                    optimization shall be performed on the single score
                    specified by ``refit``. For optimization with multiple
                    scores, please refer to Tutorial<multi_objective>

        study:
            Study corresponds to the optimization task. If :obj:`None`, a new
            study is created.

        subsample:
            Proportion of samples that are used during hyperparameter search.

            - If int, then draw ``subsample`` samples.
            - If float, then draw ``subsample`` * ``X.shape[0]`` samples.

        timeout:
            Time limit in seconds for the search of appropriate models. If
            :obj:`None`, the study is executed without time limitation. If
            :obj:`n_trials` is also set to :obj:`None`, the study continues to
            create trials until it receives a termination signal such as
            Ctrl+C or SIGTERM. This trades off runtime vs quality of the
            solution.

        verbose:
            Verbosity level. The higher, the more messages.

    Attributes:
        best_estimator_:
            Estimator that was chosen by the search. This is present only if
            ``refit`` is set to :obj:`True`.

        multimetric_:
            Whether or not the scorers compute several metrics.

        n_splits_:
            Number of cross-validation splits.

        refit_time_:
            Time for refitting the best estimator. This is present only if
            ``refit`` is set to :obj:`True`.

        sample_indices_:
            Indices of samples that are used during hyperparameter search.

        scorer_:
            Scorer function or a dict which maps the scorer key to the scorer
            callable.

        study_:
            Actual study.

    Examples:

        .. testcode::

            import optuna
            from sklearn.datasets import load_iris
            from sklearn.svm import SVC

            clf = SVC(gamma="auto")
            param_distributions = {"C": optuna.distributions.LogUniformDistribution(1e-10, 1e10)}
            optuna_search = optuna.integration.OptunaSearchCV(clf, param_distributions)
            X, y = load_iris(return_X_y=True)
            optuna_search.fit(X, y)
            y_pred = optuna_search.predict(X)
    """

    _required_parameters = ["estimator", "param_distributions"]

    @property
    def _estimator_type(self) -> str:

        return self.estimator._estimator_type

    @property
    def best_index_(self) -> int:
        """Index which corresponds to the best candidate parameter setting."""

        df = self.trials_dataframe()

        return df["value"].idxmin()

    @property
    def best_params_(self) -> Dict[str, Any]:
        """Parameters of the best trial in the :class:`~optuna.study.Study`."""

        self._check_is_fitted()

        return self.study_.best_params

    @property
    def best_score_(self) -> float:
        """Mean cross-validated score of the best estimator."""

        self._check_is_fitted()

        return self.study_.best_value

    @property
    def best_trial_(self) -> FrozenTrial:
        """Best trial in the :class:`~optuna.study.Study`."""

        self._check_is_fitted()

        return self.study_.best_trial

    @property
    def classes_(self) -> OneDimArrayLikeType:
        """Class labels."""

        self._check_is_fitted()

        return self.best_estimator_.classes_

    @property
    def n_trials_(self) -> int:
        """Actual number of trials."""

        return len(self.trials_)

    @property
    def trials_(self) -> List[FrozenTrial]:
        """All trials in the :class:`~optuna.study.Study`."""

        self._check_is_fitted()

        return self.study_.trials

    @property
    def user_attrs_(self) -> Dict[str, Any]:
        """User attributes in the :class:`~optuna.study.Study`."""

        self._check_is_fitted()

        return self.study_.user_attrs

    @property
    def decision_function(self) -> Callable[..., Union[OneDimArrayLikeType, TwoDimArrayLikeType]]:
        """Call ``decision_function`` on the best estimator.

        This is available only if the underlying estimator supports
        ``decision_function`` and ``refit`` is set to :obj:`True`.
        """

        self._check_is_fitted()

        return self.best_estimator_.decision_function

    @property
    def inverse_transform(self) -> Callable[..., TwoDimArrayLikeType]:
        """Call ``inverse_transform`` on the best estimator.

        This is available only if the underlying estimator supports
        ``inverse_transform`` and ``refit`` is set to :obj:`True`.
        """

        self._check_is_fitted()

        return self.best_estimator_.inverse_transform

    @property
    def predict(self) -> Callable[..., Union[OneDimArrayLikeType, TwoDimArrayLikeType]]:
        """Call ``predict`` on the best estimator.

        This is available only if the underlying estimator supports ``predict``
        and ``refit`` is set to :obj:`True`.
        """

        self._check_is_fitted()

        return self.best_estimator_.predict

    @property
    def predict_log_proba(self) -> Callable[..., TwoDimArrayLikeType]:
        """Call ``predict_log_proba`` on the best estimator.

        This is available only if the underlying estimator supports
        ``predict_log_proba`` and ``refit`` is set to :obj:`True`.
        """

        self._check_is_fitted()

        return self.best_estimator_.predict_log_proba

    @property
    def predict_proba(self) -> Callable[..., TwoDimArrayLikeType]:
        """Call ``predict_proba`` on the best estimator.

        This is available only if the underlying estimator supports
        ``predict_proba`` and ``refit`` is set to :obj:`True`.
        """

        self._check_is_fitted()

        return self.best_estimator_.predict_proba

    @property
    def score_samples(self) -> Callable[..., OneDimArrayLikeType]:
        """Call ``score_samples`` on the best estimator.

        This is available only if the underlying estimator supports
        ``score_samples`` and ``refit`` is set to :obj:`True`.
        """

        self._check_is_fitted()

        return self.best_estimator_.score_samples

    @property
    def set_user_attr(self) -> Callable[..., None]:
        """Call ``set_user_attr`` on the :class:`~optuna.study.Study`."""

        self._check_is_fitted()

        return self.study_.set_user_attr

    @property
    def transform(self) -> Callable[..., TwoDimArrayLikeType]:
        """Call ``transform`` on the best estimator.

        This is available only if the underlying estimator supports
        ``transform`` and ``refit`` is set to :obj:`True`.
        """

        self._check_is_fitted()

        return self.best_estimator_.transform

    @property
    def trials_dataframe(self) -> Callable[..., "pd.DataFrame"]:
        """Call ``trials_dataframe`` on the :class:`~optuna.study.Study`."""

        self._check_is_fitted()

        return self.study_.trials_dataframe

    def __init__(
        self,
        estimator: "BaseEstimator",
        param_distributions: Mapping[str, distributions.BaseDistribution],
        cv: Optional[Union["BaseCrossValidator", int]] = 5,
        enable_pruning: bool = False,
        error_score: Union[Number, float, str] = np.nan,
        max_iter: int = 1000,
        n_jobs: int = 1,
        n_trials: int = 10,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        refit: Union[bool, str, Callable] = True,
        return_train_score: bool = False,
        scoring: Optional[
            Union[
                str,
                Callable[..., float],
                List[str],
                Tuple[str],
                Callable[..., Dict[str, float]],
                Dict[str, Callable[..., float]],
            ]
        ] = None,
        study: Optional[study_module.Study] = None,
        subsample: Union[float, int] = 1.0,
        timeout: Optional[float] = None,
        verbose: int = 0,
    ) -> None:

        _imports.check()

        self.cv = cv
        self.enable_pruning = enable_pruning
        self.error_score = error_score
        self.estimator = estimator
        self.max_iter = max_iter
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.param_distributions = param_distributions
        self.random_state = random_state
        self.refit = refit
        self.return_train_score = return_train_score
        self.scoring = scoring
        self.study = study
        self.subsample = subsample
        self.timeout = timeout
        self.verbose = verbose

    def _check_is_fitted(self) -> None:

        attributes = ["n_splits_", "sample_indices_", "scorer_", "study_"]

        if self.refit:
            attributes += ["best_estimator_", "refit_time_"]

        check_is_fitted(self, attributes)
    
    # NOTE Original implementation:
    # https://github.com/scikit-learn/scikit-learn/blob/ \
    # 28ef9973362257e0627bd39db9b788c93f49362f/sklearn/model_selection/_search.py#L348-L355
    def _check_refit(self, attr: str) -> None:
        if not self.refit:
            raise AttributeError(
                f"This {type(self).__name__} instance was initialized with "
                f"`refit=False`. {attr} is available only after refitting on the best "
                "parameters. You can refit an estimator manually using the "
                "`best_params_` attribute"
            )

    def _check_params(self) -> None:

        if not hasattr(self.estimator, "fit"):
            raise ValueError("estimator must be a scikit-learn estimator.")

        if type(self.param_distributions) is not dict:
            raise ValueError("param_distributions must be a dictionary.")

        for name, distribution in self.param_distributions.items():
            if not isinstance(distribution, distributions.BaseDistribution):
                raise ValueError("Value of {} must be a optuna distribution.".format(name))

        if self.enable_pruning and not hasattr(self.estimator, "partial_fit"):
            raise ValueError("estimator must support partial_fit.")

        if self.max_iter <= 0:
            raise ValueError("max_iter must be > 0, got {}.".format(self.max_iter))

        if self.study is not None and self.study.direction != StudyDirection.MAXIMIZE:
            raise ValueError("direction of study must be 'maximize'.")

    def _more_tags(self) -> Dict[str, bool]:

        return {"non_deterministic": True, "no_validation": True}

    def _refit(
        self,
        X: TwoDimArrayLikeType,
        y: Optional[Union[OneDimArrayLikeType, TwoDimArrayLikeType]] = None,
        **fit_params: Any,
    ) -> "OptunaSearchCV":

        n_samples = _num_samples(X)

        self.best_estimator_ = clone(self.estimator)

        try:
            self.best_estimator_.set_params(**self.study_.best_params)
        except ValueError as e:
            _logger.exception(e)

        _logger.info("Refitting the estimator using {} samples...".format(n_samples))

        start_time = time()

        self.best_estimator_.fit(X, y, **fit_params)

        self.refit_time_ = time() - start_time

        _logger.info("Finished refitting! (elapsed time: {:.3f} sec.)".format(self.refit_time_))

        return self

    def fit(
        self,
        X: TwoDimArrayLikeType,
        y: Optional[Union[OneDimArrayLikeType, TwoDimArrayLikeType]] = None,
        groups: Optional[OneDimArrayLikeType] = None,
        **fit_params: Any,
    ) -> "OptunaSearchCV":
        """Run fit with all sets of parameters.

        Args:
            X:
                Training data.

            y:
                Target variable.

            groups:
                Group labels for the samples used while splitting the dataset
                into train/validation set.

            **fit_params:
                Parameters passed to ``fit`` on the estimator.

        Returns:
            self:
                Return self.
        """

        self._check_params()

        random_state = check_random_state(self.random_state)
        max_samples = self.subsample
        n_samples = _num_samples(X)
        old_level = _logger.getEffectiveLevel()

        if self.verbose > 1:
            _logger.setLevel(DEBUG)
        elif self.verbose > 0:
            _logger.setLevel(INFO)
        else:
            _logger.setLevel(WARNING)

        self.sample_indices_ = np.arange(n_samples)

        if type(max_samples) is float:
            max_samples = int(max_samples * n_samples)

        if max_samples < n_samples:
            self.sample_indices_ = random_state.choice(
                self.sample_indices_, max_samples, replace=False
            )

            self.sample_indices_.sort()

        X_res = _safe_indexing(X, self.sample_indices_)
        y_res = _safe_indexing(y, self.sample_indices_)
        groups_res = _safe_indexing(groups, self.sample_indices_)
        fit_params_res = fit_params

        if fit_params_res is not None:
            fit_params_res = _check_fit_params(X, fit_params, self.sample_indices_)

        classifier = is_classifier(self.estimator)
        cv = check_cv(self.cv, y_res, classifier)

        self.n_splits_ = cv.get_n_splits(X_res, y_res, groups=groups_res)

        if callable(self.scoring):
            self.scorer_ = self.scoring
        elif self.scoring is None or isinstance(self.scoring, str):
            self.scorer_ = check_scoring(self.estimator, self.scoring)
        else:
            self.scorer_ = _check_multimetric_scoring(self.estimator, self.scoring)
            self._check_refit_for_multimetric(self.scorer_, self.refit)

        if self.study is None:
            seed = random_state.randint(0, np.iinfo("int32").max)
            sampler = samplers.TPESampler(seed=seed)

            self.study_ = study_module.create_study(direction="maximize", sampler=sampler)

        else:
            self.study_ = self.study

        objective = _Objective(
            self.estimator,
            self.param_distributions,
            X_res,
            y_res,
            cv,
            self.enable_pruning,
            self.error_score,
            fit_params_res,
            groups_res,
            self.max_iter,
            self.refit,
            self.return_train_score,
            self.scorer_,
        )

        _logger.info(
            "Searching the best hyperparameters using {} "
            "samples...".format(_num_samples(self.sample_indices_))
        )

        self.study_.optimize(
            objective, n_jobs=self.n_jobs, n_trials=self.n_trials, timeout=self.timeout
        )

        _logger.info("Finished hyperparemeter search!")

        self.multimetric_ = objective.multimetric_

        if self.refit:
            self._refit(X, y, **fit_params)

        _logger.setLevel(old_level)

        return self

    # NOTE Original implementation:
    # https://github.com/scikit-learn/scikit-learn/blob/ \
    # 28ef9973362257e0627bd39db9b788c93f49362f/sklearn/model_selection/_search.py#L434-L476
    def score(
        self,
        X: TwoDimArrayLikeType,
        y: Optional[Union[OneDimArrayLikeType, TwoDimArrayLikeType]] = None,
    ) -> Union[Number, Dict[str, Number]]:
        """Return the score on the given data, if the estimator has been refit.

        This uses the score defined by ``scoring`` where provided, and the
        ``best_estimator_.score`` method otherwise.

        Args:
            X:
                Data.

            y:
                Target variable.

        Returns:
            score:
                Scaler score.
        """
        self._check_refit("score")
        self._check_is_fitted()
        if self.scorer_ is None:
            raise ValueError(
                "No score function explicitly defined, "
                "and the estimator doesn't provide one %s" % self.best_estimator_
            )
        if isinstance(self.scorer_, dict):
            if self.multimetric_:
                scorer = self.scorer_[self.refit]
            else:
                scorer = self.scorer_
            return scorer(self.best_estimator_, X, y)

        # callable
        score = self.scorer_(self.best_estimator_, X, y)
        if self.multimetric_:
            score = score[self.refit]
        return score
