from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import lightgbm as lgb
import numpy as np
import pytest
from sklearn.datasets import load_boston
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from optuna import study as study_module
from optuna.integration._lightgbm_tuner.sklearn import check_fit_params
from optuna.integration._lightgbm_tuner.sklearn import check_X
from optuna.integration._lightgbm_tuner.sklearn import LGBMClassifier
from optuna.integration._lightgbm_tuner.sklearn import LGBMRegressor


n_estimators = 10
random_state = 0
callback = lgb.reset_parameter(learning_rate=lambda iteration: 0.05 * (0.99 ** iteration))
early_stopping_rounds = 3


def log_likelihood(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y_pred = 1.0 / (1.0 + np.exp(-y_pred))

    return y_pred - y_true, y_pred * (1.0 - y_pred)


def zero_one_loss(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[str, float, bool]:
    return "zero_one_loss", np.mean(y_true != y_pred), False


def test_check_X() -> None:
    X, _ = load_boston(return_X_y=True)
    X = check_X(X)

    assert isinstance(X, np.ndarray)


def test_check_fit_params() -> None:
    X, y = load_boston(return_X_y=True)
    X, y, sample_weight = check_fit_params(X, y)

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(sample_weight, np.ndarray)


def test_ogbm_classifier() -> None:
    pytest.importorskip("sklearn", minversion="0.20.0")

    from sklearn.utils.estimator_checks import check_set_params

    clf = LGBMClassifier()
    name = clf.__class__.__name__

    check_set_params(name, clf)


def test_ogbm_regressor() -> None:
    pytest.importorskip("sklearn", minversion="0.20.0")

    from sklearn.utils.estimator_checks import check_set_params

    reg = LGBMRegressor()
    name = reg.__class__.__name__

    check_set_params(name, reg)


@pytest.mark.parametrize("early_stopping_rounds", [None, early_stopping_rounds])
def test_hasattr(early_stopping_rounds: int) -> None:
    X, y = load_breast_cancer(return_X_y=True)
    X, X_valid, y, y_valid = train_test_split(X, y, random_state=0)

    clf = LGBMClassifier(n_estimators=n_estimators)

    attrs = {
        "classes_": np.ndarray,
        "best_iteration_": (int, type(None)),
        "best_score_": dict,
        "booster_": lgb.Booster,
        "encoder_": LabelEncoder,
        "feature_importances_": np.ndarray,
        "n_classes_": int,
        "n_features_": int,
    }

    for attr in attrs:
        with pytest.raises(AttributeError):
            getattr(clf, attr)

    clf.fit(
        X,
        y,
        early_stopping_rounds=early_stopping_rounds,
        eval_set=[(X_valid, y_valid)],
    )

    for attr, klass in attrs.items():
        assert isinstance(getattr(clf, attr), klass)

    if early_stopping_rounds is None:
        assert clf.best_iteration_ is None
    else:
        assert clf.best_iteration_ > 0


@pytest.mark.parametrize(
    "boosting_type",
    [
        "dart",
        "gbdt",
        # "goss",
        # "rf",
    ],
)
@pytest.mark.parametrize("objective", [None, "binary", log_likelihood])
def test_fit_with_params(boosting_type: str, objective: Optional[Union[Callable, str]]) -> None:
    X, y = load_breast_cancer(return_X_y=True)
    X, X_valid, y, y_valid = train_test_split(X, y, random_state=0)

    clf = LGBMClassifier(
        boosting_type=boosting_type,
        n_estimators=n_estimators,
        objective=objective,
    )

    # See https://github.com/microsoft/LightGBM/issues/2328
    if boosting_type == "rf" and callable(objective):
        with pytest.raises(lgb.basic.LightGBMError):
            clf.fit(X, y, eval_set=[(X_valid, y_valid)])
    else:
        clf.fit(X, y, eval_set=[(X_valid, y_valid)])


def test_fit_with_invalid_study() -> None:
    X, y = load_breast_cancer(return_X_y=True)
    X, X_valid, y, y_valid = train_test_split(X, y, random_state=0)

    study = study_module.create_study(direction="maximize")
    clf = LGBMClassifier(study=study)

    with pytest.raises(ValueError):
        clf.fit(X, y, eval_set=[(X_valid, y_valid)])


@pytest.mark.parametrize("callbacks", [None, [callback]])
@pytest.mark.parametrize(
    "eval_metric",
    [
        None,
        "auc",
        # TODO(Kon): Fix #1351
        # zero_one_loss,
    ],
)
def test_fit_with_fit_params(
    callbacks: Optional[List[Callable]], eval_metric: Union[Callable, str]
) -> None:
    X, y = load_breast_cancer(return_X_y=True)
    X, X_valid, y, y_valid = train_test_split(X, y, random_state=0)

    clf = LGBMClassifier(n_estimators=n_estimators)

    clf.fit(
        X,
        y,
        callbacks=callbacks,
        eval_metric=eval_metric,
        eval_set=[(X_valid, y_valid)],
    )


def test_fit_with_unused_fit_params() -> None:
    X, y = load_breast_cancer(return_X_y=True)
    X, X_valid, y, y_valid = train_test_split(X, y, random_state=0)

    clf = LGBMClassifier(n_estimators=n_estimators)

    clf.fit(X, y, eval_set=[(X_valid, y_valid)], unknown_param=None)


@pytest.mark.parametrize("num_iteration", [None, 3])
def test_predict_with_predict_params(num_iteration: Optional[int]) -> None:
    X, y = load_breast_cancer(return_X_y=True)
    X, X_valid, y, y_valid = train_test_split(X, y, random_state=0)

    clf = LGBMClassifier(n_estimators=n_estimators)

    clf.fit(X, y, eval_set=[(X_valid, y_valid)])

    y_pred = clf.predict(X, num_iteration=num_iteration)

    assert isinstance(y_pred, np.ndarray)
    assert y.shape == y_pred.shape


def test_predict_with_unused_predict_params() -> None:
    X, y = load_breast_cancer(return_X_y=True)
    X, X_valid, y, y_valid = train_test_split(X, y, random_state=0)

    clf = LGBMClassifier(n_estimators=n_estimators)

    clf.fit(X, y, eval_set=[(X_valid, y_valid)])

    _ = clf.predict(X, unknown_param=None)


@pytest.mark.parametrize("n_jobs", [-1, 1])
def test_plot_importance(n_jobs: int) -> None:
    X, y = load_breast_cancer(return_X_y=True)
    X, X_valid, y, y_valid = train_test_split(X, y, random_state=0)

    clf = LGBMClassifier(
        n_estimators=n_estimators,
        n_jobs=n_jobs,
        refit=False,
    )

    clf.fit(X, y, eval_set=[(X_valid, y_valid)])

    lgb.plot_importance(clf)
