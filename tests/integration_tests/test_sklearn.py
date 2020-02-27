import pytest
from sklearn.datasets import make_blobs
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import SGDClassifier

from optuna import distributions
from optuna import integration

import numpy as np


def test_is_arraylike() -> None:

    assert integration.sklearn._is_arraylike([])
    assert integration.sklearn._is_arraylike(np.zeros(5))
    assert not integration.sklearn._is_arraylike(1)


def test__num_samples():
    x1 = np.random.random((10, 10))
    x2 = [1, 2, 3]
    assert integration.sklearn._num_samples(x1) == 10
    assert integration.sklearn._num_samples(x2) == 3


@pytest.mark.parametrize('enable_pruning', [True, False])
@pytest.mark.parametrize('fit_params', ['', 'coef_init'])
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_optuna_search(enable_pruning, fit_params):
    # type: (bool, str) -> None

    X, y = make_blobs(n_samples=10)
    est = SGDClassifier(max_iter=5, tol=1e-03)
    param_dist = {'alpha': distributions.LogUniformDistribution(1e-04, 1e+03)}
    optuna_search = integration.OptunaSearchCV(
        est,
        param_dist,
        cv=3,
        enable_pruning=enable_pruning,
        error_score='raise',
        max_iter=5,
        random_state=0,
        return_train_score=True
    )

    with pytest.raises(NotFittedError):
        optuna_search._check_is_fitted()

    if fit_params == 'coef_init' and not enable_pruning:
        optuna_search.fit(X, y, coef_init=np.ones((3, 2), dtype=np.float64))
    else:
        optuna_search.fit(X, y)

    optuna_search.trials_dataframe()
    optuna_search.decision_function(X)
    optuna_search.predict(X)
    optuna_search.score(X, y)
