from logging import DEBUG, INFO, WARNING
from numbers import Number
from time import perf_counter
from typing import Any, Callable, Dict, List # NOQA

import numpy as np
import pandas as pd # NOQA
from optuna import distributions
from optuna import logging
from optuna import samplers
from optuna import structs
from optuna import study
from optuna import trial as trial_module # NOQA

try:
    from sklearn.base import (
        BaseEstimator,
        MetaEstimatorMixin,
        clone,
        is_classifier
    )
    from sklearn.metrics import check_scoring
    from sklearn.model_selection import check_cv, cross_validate
    from sklearn.utils import check_random_state
    from sklearn.utils.metaestimators import _safe_split
    from sklearn.utils.validation import check_is_fitted

    _available = True

except ImportError as e:
    _import_error = e
    _available = False


def _check_sklearn_availability():
    # type: () -> None

    if not _available:
        raise ImportError(
            'scikit-learn is not available. Please install scikit-learn to use '
            'this feature. scikit-learn can be installed by executing `$ pip '
            'install scikit-learn`. For further information, please refer to '
            'the installation guide of scikit-learn. (The actual import error '
            'is as follows: ' + str(_import_error) + ')'
        )


class Objective:
    """Callable that implements objective function.

    Args:
        estimator:
            Object to use to fit the data. This is assumed to implement the
            scikit-learn estimator interface. Either this needs to provide
            ``score``, or ``scoring`` must be passed.

        param_distributions:
            Dictionary where keys are parameters and values are distributions.
            Disributions are assumed to implement the optuna distribution
            interface.

        X:
            Training data.

        y:
            Target variable.

        cv:
            Cross-validation strategy. Possible inputs for cv are:

            - integer to specify the number of folds in a CV splitter,
            - a CV splitter,
            - an iterable yielding (train, test) splits as arrays of indices.

            For integer, if :obj:`estimator` is a classifier and :obj:`y` is
            either binary or multiclass,
            ``sklearn.model_selection.StratifiedKFold`` is used. otherwise,
            ``sklearn.model_selection.KFold`` is used.

        error_score:
            Value to assign to the score if an error occurs in fitting. If
            'raise', the error is raised. If numeric,
            ``sklearn.exception.FitFailedWarning`` is raised. This does not
            affect the refit step, which will always raise the error.

        fit_params:
            Parameters passed to ``fit`` one the estimator.

        groups:
            Group labels for the samples used while splitting the dataset into
            train/test set.

        max_iter:
            Maximum nember of epochs. This is only used if the underlying
            estimator supports ``partial_fit``.

        return_train_score:
            If :obj:`True`, training scores will be included. Computing
            training scores is used to get insights on how different
            hyperparameter settings impact the overfitting/underfitting
            trade-off. However computing training scores can be
            computationally expensive and is not strictly required to select
            the hyperparameters that yield the best generalization
            performance.

        scoring:
            Scorer function.
    """

    def __init__(
        self,
        estimator,
        param_distributions,
        X,
        y=None,
        cv=5,
        error_score=np.nan,
        fit_params=None,
        groups=None,
        max_iter=1000,
        return_train_score=False,
        scoring=None
    ):
        # type: (...) -> None

        self.X = X
        self.y = y
        self.cv = cv
        self.error_score = error_score
        self.estimator = estimator
        self.fit_params = fit_params
        self.groups = groups
        self.max_iter = max_iter
        self.param_distributions = param_distributions
        self.return_train_score = return_train_score
        self.scoring = scoring

    def __call__(self, trial):
        # type: (trial_module.Trial) -> float

        estimator = clone(self.estimator)
        params = self._get_params(trial)

        estimator.set_params(**params)

        if hasattr(estimator, 'partial_fit'):
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
                scoring=self.scoring
            )

        self._store_scores(trial, scores)

        return - trial.user_attrs['mean_test_score']

    def _cross_validate_with_pruning(self, trial, estimator):
        # type: (trial_module.Trial, BaseEstimator) -> Dict[str, np.ndarray]

        if is_classifier(estimator):
            partial_fit_params = self.fit_params.copy()
            classes = np.unique(self.y)

            partial_fit_params.setdefault('classes', classes)

        else:
            partial_fit_params = self.fit_params

        n_splits = self.cv.get_n_splits(self.X, self.y, groups=self.groups)
        estimators = [clone(estimator) for _ in range(n_splits)]
        scores = {
            'fit_time': np.zeros(n_splits),
            'score_time': np.zeros(n_splits),
            'test_score': np.empty(n_splits)
        }

        if self.return_train_score:
            scores['train_score'] = np.empty(n_splits)

        for step in range(self.max_iter):
            for i, (train, test) in enumerate(
                self.cv.split(self.X, self.y, groups=self.groups)
            ):
                out = self._partial_fit_and_score(
                    estimators[i],
                    train,
                    test,
                    partial_fit_params
                )

                if self.return_train_score:
                    scores['train_score'][i] = out.pop(0)

                scores['test_score'][i] = out[0]
                scores['fit_time'][i] += out[1]
                scores['score_time'][i] += out[2]

            intermediate_value = - np.nanmean(scores['test_score'])

            trial.report(intermediate_value, step=step)

            if trial.should_prune(step):
                self._store_scores(trial, scores)

                raise structs.TrialPruned(
                    'trial was pruned at iteration {}'.format(step)
                )

        return scores

    def _get_params(self, trial):
        # type: (trial_module.Trial) -> Dict[str, Any]

        return {
            name: trial._suggest(
                name, distribution
            ) for name, distribution in self.param_distributions.items()
        }

    def _partial_fit_and_score(
        self,
        estimator,
        train,
        test,
        partial_fit_params
    ):
        # type: (...) -> List[float]

        X_train, y_train = _safe_split(estimator, self.X, self.y, train)
        X_test, y_test = _safe_split(
            estimator,
            self.X,
            self.y,
            test,
            train_indices=train
        )

        start_time = perf_counter()

        try:
            estimator.partial_fit(X_train, y_train, **partial_fit_params)

        except Exception as e:
            if self.error_score == 'raise':
                raise e

            elif isinstance(self.error_score, Number):
                fit_time = perf_counter() - start_time
                test_score = self.error_score
                score_time = 0.0

                if self.return_train_score:
                    train_score = self.error_score

            else:
                raise ValueError("error_score must be 'raise' or numeric.")

        else:
            fit_time = perf_counter() - start_time
            test_score = self.scoring(estimator, X_test, y_test)
            score_time = perf_counter() - fit_time - start_time

            if self.return_train_score:
                train_score = self.scoring(estimator, X_train, y_train)

        ret = [test_score, fit_time, score_time]

        if self.return_train_score:
            ret.insert(0, train_score)

        return ret

    def _store_scores(self, trial, scores):
        # type: (trial_module.Trial, Dict[str, float]) -> None

        for name, array in scores.items():
            if name in ['test_score', 'train_score']:
                for i, score in enumerate(array):
                    trial.set_user_attr('split{}_{}'.format(i, name), score)

            trial.set_user_attr('mean_{}'.format(name), np.average(array))
            trial.set_user_attr('std_{}'.format(name), np.std(array))


class TPESearchCV(BaseEstimator, MetaEstimatorMixin):
    """Hyperparameter search with cross-validation.

    Args:
        estimator:
            Object to use to fit the data. This is assumed to implement the
            scikit-learn estimator interface. Either this needs to provide
            ``score``, or ``scoring`` must be passed.

        param_distributions:
            Dictionary where keys are parameters and values are distributions.
            Disributions are assumed to implement the optuna distribution
            interface.

        cv:
            Cross-validation strategy. Possible inputs for cv are:

            - integer to specify the number of folds in a CV splitter,
            - a CV splitter,
            - an iterable yielding (train, test) splits as arrays of indices.

            For integer, if :obj:`estimator` is a classifier and :obj:`y` is
            either binary or multiclass,
            ``sklearn.model_selection.StratifiedKFold`` is used. otherwise,
            ``sklearn.model_selection.KFold`` is used.

        error_score:
            Value to assign to the score if an error occurs in fitting. If
            'raise', the error is raised. If numeric,
            ``sklearn.exception.FitFailedWarning`` is raised. This does not
            affect the refit step, which will always raise the error.

        load_if_exists:
            If :obj:`True`, the existing study is used in the case where a
            study named :obj:`study_name` already exists in the
            :obj:`storage`.

        max_iter:
            Maximum nember of epochs. This is only used if the underlying
            estimator supports ``partial_fit``.

        n_jobs:
            Number of parallel jobs. :obj:`-1` means using all processors.

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

        return_train_score:
            If :obj:`True`, training scores will be included. Computing
            training scores is used to get insights on how different
            hyperparameter settings impact the overfitting/underfitting
            trade-off. However computing training scores can be
            computationally expensive and is not strictly required to select
            the hyperparameters that yield the best generalization
            performance.

        pruner:
            Pruner that decides early stopping of unpromising trials. If
            :obj:`None`, :class:`~optuna.sampler.MedianPruner` is used as the
            default.

        refit:
            If :obj:`True`, refit the estimator with the best found
            hyperparameters. The refitted estimator is made available at the
            ``best_estimator_`` attribute and permits using ``predict``
            directly.

        scoring:
            String or callable to evaluate the predictions on the test data.
            If :obj:`None`, ``score`` on the estimator is used.

        storage:
            Database URL. If :obj:`None`, in-memory storage is used, and the
            :class:`~optuna.study.Study` will not be persistent.

        study_name:
            name of the :class:`~optuna.study.Study`. If :obj:`None`, a unique
            name is generated automatically.

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

        n_splits_:
            Number of cross-validation splits.

        refit_time_:
            Time for refitting the best estimator. This is present only if
            ``refit`` is set to :obj:`True`.

        scorer_:
            Scorer function.

        study_:
            Study corresponds to the optimization task.

    Examples:
        >>> import optuna
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.svm import SVC
        >>> clf = SVC(gamma='auto')
        >>> param_distributions = {
        ...     'C': optuna.distributions.LogUniformDistribution(1e-10, 1e+10)
        ... }
        >>> tpe_search = optuna.integration.TPESearchCV(
        ...     clf,
        ...     param_distributions
        ... )
        >>> X, y = load_iris(return_X_y=True)
        >>> tpe_search.fit(X, y) # doctest: +ELLIPSIS
        TPESearchCV(...)
        >>> y_pred = tpe_search.predict(X)
    """

    @property
    def _estimator_type(self):
        # type: () -> str

        return self.estimator._estimator_type

    @property
    def _sampler(self):
        # type: () -> samplers.TPESampler

        random_state = check_random_state(self.random_state)
        seed = random_state.randint(0, np.iinfo(np.int32).max)

        return samplers.TPESampler(seed=seed)

    @property
    def best_index_(self):
        # type: () -> int
        """Index which corresponds to the best candidate parameter setting.
        """

        self._check_is_fitted()

        df = self.trials_dataframe()

        return df['value'].idxmin()

    @property
    def best_params_(self):
        # type: () -> Dict[str, Any]
        """Parameters of the best trial in the :class:`~optuna.study.Study`.
        """

        self._check_is_fitted()

        return self.study_.best_params

    @property
    def best_score_(self):
        # type: () -> float
        """Mean cross-validated score of the best estimator.
        """

        self._check_is_fitted()

        return - self.best_value_

    @property
    def best_trial_(self):
        # type: () -> structs.FrozenTrial
        """Best trial in the :class:`~optuna.study.Study`.
        """

        self._check_is_fitted()

        return self.study_.best_trial

    @property
    def best_value_(self):
        # type: () -> float
        """Best objective value in the :class:`~optuna.study.Study`.
        """

        self._check_is_fitted()

        return self.study_.best_value

    @property
    def classes_(self):
        # type: () -> np.ndarray
        """Class labels.
        """

        self._check_is_fitted()

        return self.best_estimator_.classes_

    @property
    def n_trials_(self):
        # type: () -> int
        """Actual number of trials.
        """

        self._check_is_fitted()

        return len(self.study_.trials)

    @property
    def decision_function(self):
        # type: () -> Callable[..., np.ndarray]
        """Call ``decision_function`` on the estimator with the best found
        parameters. This is available only if the underlying estimator
        supports ``decision_function`` and ``refit`` is set to :obj:`True`.
        """

        self._check_is_fitted()

        return self.best_estimator_.decision_function

    @property
    def inverse_transform(self):
        # type: () -> Callable[..., np.ndarray]
        """Call ``inverse_transform`` on the estimator with the best found
        parameters. This is available only if the underlying estimator
        supports ``inverse_transform`` and ``refit`` is set to :obj:`True`.
        """

        self._check_is_fitted()

        return self.best_estimator_.inverse_transform

    @property
    def predict(self):
        # type: () -> Callable[..., np.ndarray]
        """Call ``predict`` on the estimator with the best found parameters.
        This is available only if the underlying estimator supports
        ``predict`` and ``refit`` is set to :obj:`True`.
        """

        self._check_is_fitted()

        return self.best_estimator_.predict

    @property
    def predict_log_proba(self):
        # type: () -> Callable[..., np.ndarray]
        """Call ``predict_log_proba`` on the estimator with the best found
        parameters. This is available only if the underlying estimator
        supports ``predict_log_proba`` and ``refit`` is set to :obj:`True`.
        """

        self._check_is_fitted()

        return self.best_estimator_.predict_log_proba

    @property
    def predict_proba(self):
        # type: () -> Callable[..., np.ndarray]
        """Call ``predict_proba`` on the estimator with the best found
        parameters. This is available only if the underlying estimator
        supports ``predict_proba`` and ``refit`` is set to :obj:`True`.
        """

        self._check_is_fitted()

        return self.best_estimator_.predict_proba

    @property
    def score_samples(self):
        # type: () -> Callable[..., np.ndarray]
        """Call ``score_samples`` on the estimator with the best found
        parameters. This is available only if the underlying estimator
        supports ``score_samples`` and ``refit`` is set to :obj:`True`.
        """

        self._check_is_fitted()

        return self.best_estimator_.score_samples

    @property
    def transform(self):
        # type: () -> Callable[..., np.ndarray]
        """Call ``transform`` on the estimator with the best found parameters.
        This is available only if the underlying estimator supports
        ``transform`` and ``refit`` is set to :obj:`True`.
        """

        self._check_is_fitted()

        return self.best_estimator_.transform

    @property
    def trials_dataframe(self):
        # type: () -> Callable[..., pd.DataFrame]
        """Call ``trials_dataframe`` on the :class:`~optuna.study.Study`.
        """

        self._check_is_fitted()

        return self.study_.trials_dataframe

    def __init__(
        self,
        estimator,
        param_distributions,
        cv=5,
        error_score=np.nan,
        load_if_exists=False,
        max_iter=1000,
        n_jobs=1,
        n_trials=10,
        pruner=None,
        random_state=None,
        refit=True,
        return_train_score=False,
        scoring=None,
        storage=None,
        study_name=None,
        timeout=None,
        verbose=0
    ):
        # type: (...) -> None

        _check_sklearn_availability()

        self.cv = cv
        self.error_score = error_score
        self.estimator = estimator
        self.load_if_exists = load_if_exists
        self.max_iter = max_iter
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.param_distributions = param_distributions
        self.pruner = pruner
        self.random_state = random_state
        self.refit = refit
        self.return_train_score = return_train_score
        self.scoring = scoring
        self.storage = storage
        self.study_name = study_name
        self.timeout = timeout
        self.verbose = verbose

    def _check_is_fitted(self):
        # type: () -> None

        attributes = ['n_splits_', 'scorer_', 'study_']

        if self.refit:
            attributes += ['best_estimator_', 'refit_time_']

        check_is_fitted(self, attributes)

    def _check_params(self):
        # type: () -> None

        if not hasattr(self.estimator, 'fit'):
            raise ValueError(
                'estimator must be a scikit-learn estimator'
            )

        if type(self.param_distributions) is not dict:
            raise ValueError('param_distributions must be a dictionary')

        for name, distribution in self.param_distributions.items():
            if not isinstance(distribution, distributions.BaseDistribution):
                raise ValueError(
                    'value of {} must be a optuna distribution'.format(name)
                )

        if self.max_iter <= 0:
            raise ValueError(
                'max_iter must be > 0, got {}'.format(self.max_iter)
            )

    def _refit(self, X, y=None, **fit_params):
        # type: (np.ndarray, np.ndarray, Any) -> 'TPESearchCV'

        self.best_estimator_ = clone(self.estimator)

        self.best_estimator_.set_params(**self.study_.best_params)

        start_time = perf_counter()

        self.best_estimator_.fit(X, y, **fit_params)

        self.refit_time_ = perf_counter() - start_time

        return self

    def _set_verbosity(self):
        # type: () -> None

        if self.verbose > 1:
            logging.set_verbosity(DEBUG)
        elif self.verbose > 0:
            logging.set_verbosity(INFO)
        else:
            logging.set_verbosity(WARNING)

    def fit(self, X, y=None, groups=None, **fit_params):
        # type: (np.ndarray, np.ndarray, np.ndarray, Any) -> 'TPESearchCV'
        """Run fit with all sets of parameters.

        Args:
            X:
                Training data.

            y:
                Target variable.

            groups:
                Group labels for the samples used while splitting the dataset
                into train/test set.

            **fit_params:
                Parameters passed to ``fit`` on the estimator.

        Returns:
            self:
                Return self.
        """

        self._check_params()
        self._set_verbosity()

        classifier = is_classifier(self.estimator)
        cv = check_cv(self.cv, y, classifier)

        self.n_splits_ = cv.get_n_splits(X, y, groups=groups)
        self.scorer_ = check_scoring(self.estimator, scoring=self.scoring)
        self.study_ = study.create_study(
            load_if_exists=self.load_if_exists,
            pruner=self.pruner,
            sampler=self._sampler,
            storage=self.storage,
            study_name=self.study_name
        )

        objective = Objective(
            self.estimator,
            self.param_distributions,
            X,
            y,
            cv=cv,
            error_score=self.error_score,
            fit_params=fit_params,
            groups=groups,
            max_iter=self.max_iter,
            return_train_score=self.return_train_score,
            scoring=self.scorer_
        )

        self.study_.optimize(
            objective,
            n_jobs=self.n_jobs,
            n_trials=self.n_trials,
            timeout=self.timeout
        )

        if self.refit:
            self._refit(X, y, **fit_params)

        return self

    def score(self, X, y=None):
        # type: (np.ndarray, np.ndarray) -> float
        """Return the score on the given data.

        Args:
            X:
                Data.

            y:
                Target variable.

        Returns:
            score:
                Scaler score.
        """

        return self.scorer_(self.best_estimator_, X, y)
