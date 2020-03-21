import pytest
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KernelDensity

from optuna import distributions
from optuna import integration
from optuna.study import create_study

import numpy as np
import scipy as sp


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
def test_optuna_search(enable_pruning, fit_params):
    # type: (bool, str) -> None

    X, y = make_blobs(n_samples=10)
    est = SGDClassifier(max_iter=5, tol=1e-03)
    param_dist = {"alpha": distributions.LogUniformDistribution(1e-04, 1e03)}
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
def test_optuna_search_properties():
    # type: () -> None

    X, y = make_blobs(n_samples=10)
    est = LogisticRegression(max_iter=5, tol=1e-03)
    param_dist = {"C": distributions.LogUniformDistribution(1e-04, 1e03)}

    optuna_search = integration.OptunaSearchCV(
        est, param_dist, cv=3, error_score="raise", random_state=0, return_train_score=True
    )
    optuna_search.fit(X, y)
    optuna_search.set_user_attr("dataset", "blobs")

    assert optuna_search._estimator_type == "classifier"
    assert type(optuna_search.best_index_) == int
    assert type(optuna_search.best_params_) == dict
    assert optuna_search.best_score_ is not None
    assert optuna_search.best_trial_ is not None
    assert np.allclose(optuna_search.classes_, np.array([0, 1, 2]))
    assert optuna_search.n_trials_ == 10
    assert optuna_search.user_attrs_ == {"dataset": "blobs"}
    assert type(optuna_search.predict_log_proba(X)) == np.ndarray
    assert type(optuna_search.predict_proba(X)) == np.ndarray


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_optuna_search_score_samples():
    # type: () -> None

    X, y = make_blobs(n_samples=10)
    est = KernelDensity()
    optuna_search = integration.OptunaSearchCV(
        est, {}, cv=3, error_score="raise", random_state=0, return_train_score=True
    )
    optuna_search.fit(X)
    assert optuna_search.score_samples(X) is not None


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_optuna_search_transforms():
    # type: () -> None

    X, y = make_blobs(n_samples=10)
    est = PCA()
    optuna_search = integration.OptunaSearchCV(
        est, {}, cv=3, error_score="raise", random_state=0, return_train_score=True
    )
    optuna_search.fit(X)
    assert type(optuna_search.transform(X)) == np.ndarray
    assert type(optuna_search.inverse_transform(X)) == np.ndarray


def test_optuna_search_invalid_estimator():
    # type: () -> None

    X, y = make_blobs(n_samples=10)
    est = "not an estimator"
    optuna_search = integration.OptunaSearchCV(
        est, {}, cv=3, error_score="raise", random_state=0, return_train_score=True
    )

    with pytest.raises(ValueError, match="estimator must be a scikit-learn estimator."):
        optuna_search.fit(X)


def test_optuna_search_invalid_param_dist():
    # type: () -> None

    X, y = make_blobs(n_samples=10)
    est = KernelDensity()
    param_dist = ["kernel", distributions.CategoricalDistribution(("gaussian", "linear"))]
    optuna_search = integration.OptunaSearchCV(
        est,
        param_dist,  # type: ignore
        cv=3,
        error_score="raise",
        random_state=0,
        return_train_score=True,
    )

    with pytest.raises(ValueError, match="param_distributions must be a dictionary."):
        optuna_search.fit(X)


def test_optuna_search_pruning_without_partial_fit():
    # type: () -> None

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


def test_optuna_search_negative_max_iter():
    # type: () -> None

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


def test_optuna_search_tuple_instead_of_distribution():
    # type: () -> None

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


def test_optuna_search_study_with_minimize():
    # type: () -> None

    X, y = make_blobs(n_samples=10)
    est = KernelDensity()
    study = create_study(direction="minimize")
    optuna_search = integration.OptunaSearchCV(
        est, {}, cv=3, error_score="raise", random_state=0, return_train_score=True, study=study
    )

    with pytest.raises(ValueError, match="direction of study must be 'maximize'."):
        optuna_search.fit(X)


@pytest.mark.parametrize("verbose", [1, 2])
def test_optuna_search_verbosity(verbose):
    # type: (int) -> None

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


def test_optuna_search_subsample():
    # type: () -> None

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
def test_objective_y_None():
    # type: () -> None

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

    with pytest.raises(ValueError, match="y cannot be None"):
        optuna_search.fit(X)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_objective_error_score_nan():
    # type: () -> None

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

    with pytest.raises(ValueError, match="y cannot be None"):
        optuna_search.fit(X)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_objective_error_score_invalid():
    # type: () -> None

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
