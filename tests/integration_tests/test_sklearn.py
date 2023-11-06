from __future__ import annotations

from unittest.mock import MagicMock
from unittest.mock import patch
import warnings

import numpy as np
import pytest
import scipy as sp
from sklearn.datasets import make_blobs
from sklearn.datasets import make_regression
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KernelDensity
from sklearn.tree import DecisionTreeRegressor

from optuna import distributions
from optuna import integration
from optuna.samplers import BruteForceSampler
from optuna.study import create_study
from optuna.terminator.erroreval import _CROSS_VALIDATION_SCORES_KEY


pytestmark = pytest.mark.integration


def test_is_arraylike() -> None:
    assert integration.sklearn._is_arraylike([])
    assert integration.sklearn._is_arraylike(np.zeros(5))
    assert not integration.sklearn._is_arraylike(1)


def test_num_samples() -> None:
    x1 = np.random.random((10, 10))
    x2 = [1, 2, 3]
    assert integration.sklearn._num_samples(x1) == 10
    assert integration.sklearn._num_samples(x2) == 3


def test_make_indexable() -> None:
    x1 = np.random.random((10, 10))
    x2 = sp.sparse.coo_matrix(x1)
    x3 = [1, 2, 3]

    assert hasattr(integration.sklearn._make_indexable(x1), "__getitem__")
    assert hasattr(integration.sklearn._make_indexable(x2), "__getitem__")
    assert hasattr(integration.sklearn._make_indexable(x3), "__getitem__")
    assert integration.sklearn._make_indexable(None) is None


@pytest.mark.parametrize("enable_pruning", [True, False])
@pytest.mark.parametrize("fit_params", ["", "coef_init"])
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_optuna_search(enable_pruning: bool, fit_params: str) -> None:
    X, y = make_blobs(n_samples=10)
    est = SGDClassifier(max_iter=5, tol=1e-03)
    param_dist = {"alpha": distributions.FloatDistribution(1e-04, 1e03, log=True)}
    optuna_search = integration.OptunaSearchCV(
        est,
        param_dist,
        cv=3,
        enable_pruning=enable_pruning,
        error_score="raise",
        max_iter=5,
        random_state=0,
        return_train_score=True,
    )

    with pytest.raises(NotFittedError):
        optuna_search._check_is_fitted()

    if fit_params == "coef_init" and not enable_pruning:
        optuna_search.fit(X, y, coef_init=np.ones((3, 2), dtype=np.float64))
    else:
        optuna_search.fit(X, y)

    optuna_search.trials_dataframe()
    optuna_search.decision_function(X)
    optuna_search.predict(X)
    optuna_search.score(X, y)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_optuna_search_properties() -> None:
    X, y = make_blobs(n_samples=10)
    est = LogisticRegression(tol=1e-03)
    param_dist = {"C": distributions.FloatDistribution(1e-04, 1e03, log=True)}

    optuna_search = integration.OptunaSearchCV(
        est, param_dist, cv=3, error_score="raise", random_state=0, return_train_score=True
    )
    optuna_search.fit(X, y)
    optuna_search.set_user_attr("dataset", "blobs")

    assert optuna_search._estimator_type == "classifier"
    assert isinstance(optuna_search.best_index_, int)
    assert isinstance(optuna_search.best_params_, dict)
    assert isinstance(optuna_search.cv_results_, dict)
    for cv_result_list_ in optuna_search.cv_results_.values():
        assert len(cv_result_list_) == optuna_search.n_trials_
    assert optuna_search.best_score_ is not None
    assert optuna_search.best_trial_ is not None
    assert np.allclose(optuna_search.classes_, np.array([0, 1, 2]))
    assert optuna_search.n_trials_ == 10
    assert optuna_search.user_attrs_ == {"dataset": "blobs"}
    assert type(optuna_search.predict_log_proba(X)) == np.ndarray
    assert type(optuna_search.predict_proba(X)) == np.ndarray


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_optuna_search_score_samples() -> None:
    X, y = make_blobs(n_samples=10)
    est = KernelDensity()
    optuna_search = integration.OptunaSearchCV(
        est, {}, cv=3, error_score="raise", random_state=0, return_train_score=True
    )
    optuna_search.fit(X)
    assert optuna_search.score_samples(X) is not None


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_optuna_search_transforms() -> None:
    X, y = make_blobs(n_samples=10)
    est = PCA()
    optuna_search = integration.OptunaSearchCV(
        est, {}, cv=3, error_score="raise", random_state=0, return_train_score=True
    )
    optuna_search.fit(X)
    assert type(optuna_search.transform(X)) == np.ndarray
    assert type(optuna_search.inverse_transform(X)) == np.ndarray


def test_optuna_search_invalid_estimator() -> None:
    X, y = make_blobs(n_samples=10)
    est = "not an estimator"
    optuna_search = integration.OptunaSearchCV(
        est, {}, cv=3, error_score="raise", random_state=0, return_train_score=True
    )

    with pytest.raises(ValueError, match="estimator must be a scikit-learn estimator."):
        optuna_search.fit(X)


def test_optuna_search_pruning_without_partial_fit() -> None:
    X, y = make_blobs(n_samples=10)
    est = KernelDensity()
    param_dist = {}  # type: ignore
    optuna_search = integration.OptunaSearchCV(
        est,
        param_dist,
        cv=3,
        enable_pruning=True,
        error_score="raise",
        random_state=0,
        return_train_score=True,
    )

    with pytest.raises(ValueError, match="estimator must support partial_fit."):
        optuna_search.fit(X)


def test_optuna_search_negative_max_iter() -> None:
    X, y = make_blobs(n_samples=10)
    est = KernelDensity()
    param_dist = {}  # type: ignore
    optuna_search = integration.OptunaSearchCV(
        est,
        param_dist,
        cv=3,
        max_iter=-1,
        error_score="raise",
        random_state=0,
        return_train_score=True,
    )

    with pytest.raises(ValueError, match="max_iter must be > 0"):
        optuna_search.fit(X)


def test_optuna_search_tuple_instead_of_distribution() -> None:
    X, y = make_blobs(n_samples=10)
    est = KernelDensity()
    param_dist = {"kernel": ("gaussian", "linear")}
    optuna_search = integration.OptunaSearchCV(
        est,
        param_dist,  # type: ignore
        cv=3,
        error_score="raise",
        random_state=0,
        return_train_score=True,
    )

    with pytest.raises(ValueError, match="must be a optuna distribution."):
        optuna_search.fit(X)


def test_optuna_search_study_with_minimize() -> None:
    X, y = make_blobs(n_samples=10)
    est = KernelDensity()
    study = create_study(direction="minimize")
    optuna_search = integration.OptunaSearchCV(
        est, {}, cv=3, error_score="raise", random_state=0, return_train_score=True, study=study
    )

    with pytest.raises(ValueError, match="direction of study must be 'maximize'."):
        optuna_search.fit(X)


@pytest.mark.parametrize("verbose", [1, 2])
def test_optuna_search_verbosity(verbose: int) -> None:
    X, y = make_blobs(n_samples=10)
    est = KernelDensity()
    param_dist = {}  # type: ignore
    optuna_search = integration.OptunaSearchCV(
        est,
        param_dist,
        cv=3,
        error_score="raise",
        random_state=0,
        return_train_score=True,
        verbose=verbose,
    )
    optuna_search.fit(X)


def test_optuna_search_subsample() -> None:
    X, y = make_blobs(n_samples=10)
    est = KernelDensity()
    param_dist = {}  # type: ignore
    optuna_search = integration.OptunaSearchCV(
        est,
        param_dist,
        cv=3,
        error_score="raise",
        random_state=0,
        return_train_score=True,
        subsample=5,
    )
    optuna_search.fit(X)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_objective_y_None() -> None:
    X, y = make_blobs(n_samples=10)
    est = SGDClassifier(max_iter=5, tol=1e-03)
    param_dist = {}  # type: ignore
    optuna_search = integration.OptunaSearchCV(
        est,
        param_dist,
        cv=3,
        enable_pruning=True,
        error_score="raise",
        random_state=0,
        return_train_score=True,
    )

    with pytest.raises(ValueError):
        optuna_search.fit(X)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_objective_error_score_nan() -> None:
    X, y = make_blobs(n_samples=10)
    est = SGDClassifier(max_iter=5, tol=1e-03)
    param_dist = {}  # type: ignore
    optuna_search = integration.OptunaSearchCV(
        est,
        param_dist,
        cv=3,
        enable_pruning=True,
        max_iter=5,
        error_score=np.nan,
        random_state=0,
        return_train_score=True,
    )

    with pytest.raises(
        ValueError,
        match="This SGDClassifier estimator requires y to be passed, but the target y is None.",
    ):
        optuna_search.fit(X)

    for trial in optuna_search.study_.get_trials():
        assert np.all(np.isnan(list(trial.intermediate_values.values())))

        # "_score" stores every score value for train and test validation holds.
        for name, value in trial.user_attrs.items():
            if name.endswith("_score"):
                assert np.isnan(value)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_objective_error_score_invalid() -> None:
    X, y = make_blobs(n_samples=10)
    est = SGDClassifier(max_iter=5, tol=1e-03)
    param_dist = {}  # type: ignore
    optuna_search = integration.OptunaSearchCV(
        est,
        param_dist,
        cv=3,
        enable_pruning=True,
        max_iter=5,
        error_score="invalid error score",
        random_state=0,
        return_train_score=True,
    )

    with pytest.raises(ValueError, match="error_score must be 'raise' or numeric."):
        optuna_search.fit(X)


# This test checks whether OptunaSearchCV completes the study without halting, even if some trials
# fails due to misconfiguration.
@pytest.mark.parametrize(
    "param_dist,all_params",
    [
        ({"max_depth": distributions.IntDistribution(0, 1)}, [0, 1]),
        ({"max_depth": distributions.IntDistribution(0, 0)}, [0]),
    ],
)
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_no_halt_with_error(
    param_dist: dict[str, distributions.BaseDistribution], all_params: list[int]
) -> None:
    X, y = make_regression(n_samples=100, n_features=10)
    estimator = DecisionTreeRegressor()
    study = create_study(sampler=BruteForceSampler(), direction="maximize")

    # DecisionTreeRegressor raises ValueError when max_depth==0.
    optuna_search = integration.OptunaSearchCV(
        estimator,
        param_dist,
        study=study,
    )
    optuna_search.fit(X, y)
    all_suggested_values = [t.params["max_depth"] for t in study.trials]
    assert len(all_suggested_values) == len(all_params)
    for a in all_params:
        assert a in all_suggested_values


# TODO(himkt): Remove this method with the deletion of deprecated distributions.
# https://github.com/optuna/optuna/issues/2941
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_optuna_search_convert_deprecated_distribution() -> None:
    param_dist = {
        "ud": distributions.UniformDistribution(low=0, high=10),
        "dud": distributions.DiscreteUniformDistribution(low=0, high=10, q=2),
        "lud": distributions.LogUniformDistribution(low=1, high=10),
        "id": distributions.IntUniformDistribution(low=0, high=10),
        "idd": distributions.IntUniformDistribution(low=0, high=10, step=2),
        "ild": distributions.IntLogUniformDistribution(low=1, high=10),
    }

    expected_param_dist = {
        "ud": distributions.FloatDistribution(low=0, high=10, log=False, step=None),
        "dud": distributions.FloatDistribution(low=0, high=10, log=False, step=2),
        "lud": distributions.FloatDistribution(low=1, high=10, log=True, step=None),
        "id": distributions.IntDistribution(low=0, high=10, log=False, step=1),
        "idd": distributions.IntDistribution(low=0, high=10, log=False, step=2),
        "ild": distributions.IntDistribution(low=1, high=10, log=True, step=1),
    }

    with pytest.raises(ValueError):
        optuna_search = integration.OptunaSearchCV(
            KernelDensity(),
            param_dist,
        )

    # It confirms that ask doesn't convert non-deprecated distributions.
    optuna_search = integration.OptunaSearchCV(
        KernelDensity(),
        expected_param_dist,
    )

    assert optuna_search.param_distributions == expected_param_dist


def test_callbacks() -> None:
    callbacks = []

    for _ in range(2):
        callback = MagicMock()
        callback.__call__ = MagicMock(return_value=None)  # type: ignore
        callbacks.append(callback)

    n_trials = 5
    X, y = make_blobs(n_samples=10)
    est = SGDClassifier(max_iter=5, tol=1e-03)
    param_dist = {"alpha": distributions.FloatDistribution(1e-04, 1e03, log=True)}
    optuna_search = integration.OptunaSearchCV(
        est,
        param_dist,
        cv=3,
        enable_pruning=True,
        max_iter=5,
        n_trials=n_trials,
        error_score=np.nan,
        callbacks=callbacks,  # type: ignore
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        optuna_search.fit(X, y)

    for callback in callbacks:
        for trial in optuna_search.trials_:
            callback.assert_any_call(optuna_search.study_, trial)
        assert callback.call_count == n_trials


@pytest.mark.filterwarnings("ignore::UserWarning")
@patch("optuna.integration.sklearn.cross_validate")
def test_terminator_cv_score_reporting(mock: MagicMock) -> None:
    scores = {
        "fit_time": np.array([2.01, 1.78, 3.22]),
        "score_time": np.array([0.33, 0.35, 0.48]),
        "test_score": np.array([0.04, 0.80, 0.70]),
    }
    mock.return_value = scores

    X, _ = make_blobs(n_samples=10)
    est = PCA()
    optuna_search = integration.OptunaSearchCV(est, {}, cv=3, error_score="raise", random_state=0)
    optuna_search.fit(X)

    for trial in optuna_search.study_.trials:
        assert (trial.system_attrs[_CROSS_VALIDATION_SCORES_KEY] == scores["test_score"]).all()
