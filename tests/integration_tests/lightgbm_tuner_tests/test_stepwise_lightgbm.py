from typing import Any
from typing import Dict
from typing import Tuple

import lightgbm as lgb
import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from optuna import stepwise
import optuna.distributions
from optuna.integration.lightgbm import default_lgb_steps
from optuna.integration.lightgbm import StepwiseLightGBMTuner
from optuna.integration.lightgbm import StepwiseLightGBMTunerCV


SEED = 42
METRIC = "binary_logloss"
VALID_NAME = "valid_set"


def get_train_val() -> Tuple[lgb.Dataset, lgb.Dataset]:
    # cannnot be a fixture because lgbm keeps pointer references.
    X_train, X_test, y_train, y_test = train_test_split(
        *load_breast_cancer(return_X_y=True), test_size=0.1, random_state=SEED
    )
    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = train_set.create_valid(X_test, label=y_test, silent=True)
    return train_set, val_set


def get_train() -> lgb.Dataset:
    # cannnot be a fixture because lgbm keeps pointer references.
    X_train, y_train = load_breast_cancer(return_X_y=True)
    return lgb.Dataset(X_train, label=y_train)


@pytest.fixture
def train_kwargs() -> Dict[str, Any]:
    return {"num_boost_round": 10, "verbose_eval": False}


@pytest.fixture
def cv_kwargs() -> Dict[str, Any]:
    return {"num_boost_round": 10, "verbose_eval": False, "early_stopping_rounds": 1, "seed": SEED}


@pytest.fixture
def params() -> Dict[str, Any]:
    return {"objective": "binary", "metric": METRIC, "verbose": -1, "seed": SEED}


@pytest.fixture
def dummy_steps() -> stepwise.StepListType:
    dists = {"_NOT_EXIST_": optuna.distributions.IntUniformDistribution(1, 1)}
    steps = [("1", stepwise.Step(dists, n_trials=1))]
    return steps


@pytest.fixture
def default_train_score(params: Dict[str, Any], train_kwargs: Dict[str, Any]) -> float:
    train_set, val_set = get_train_val()
    booster = lgb.train(
        params, train_set, valid_sets=val_set, valid_names=VALID_NAME, **train_kwargs
    )
    return booster.best_score[VALID_NAME][METRIC]


@pytest.fixture
def default_cv_score(params: Dict[str, Any], cv_kwargs: Dict[str, Any]) -> float:
    train_set = get_train()
    cv_results = lgb.cv(params, train_set, **cv_kwargs)
    return cv_results[f"{METRIC}-mean"][-1]


def test_first_metric_only(params: Dict[str, Any]) -> None:
    params["first_metric_only"] = False
    train_set, val_set = get_train_val()
    with pytest.raises(ValueError, match="Optuna only handles a single metric"):
        StepwiseLightGBMTuner(params, train_set=train_set, valid_sets=val_set)


def _optimize_train(
    params: Dict[str, Any],
    train_kwargs: Dict[str, Any],
    steps: stepwise.StepListType,
    n_trials: int = 2,
) -> StepwiseLightGBMTuner:
    train_set, val_set = get_train_val()
    tuner = StepwiseLightGBMTuner(
        params, train_set=train_set, steps=steps, valid_sets=val_set, **train_kwargs
    )
    tuner.optimize(n_trials=n_trials)
    return tuner


def _check_optimize_train(
    params: Dict[str, Any],
    train_kwargs: Dict[str, Any],
    default_train_score: float,
    steps: stepwise.StepListType,
) -> None:
    tuner = _optimize_train(params, train_kwargs, steps, n_trials=100)
    assert (
        np.isclose(default_train_score, tuner.best_value)
        or tuner.best_value <= default_train_score
    )

    model = tuner.get_best_booster()
    assert model.best_score == tuner.best_value

    for name, value in tuner.best_params.items():
        assert model.params[name] == value


@pytest.mark.parametrize(
    "named_step",
    [named_step for named_step in default_lgb_steps()],
    ids=[step_name for step_name, _ in default_lgb_steps()],
)
def test_train_single_step(
    params: Dict[str, Any],
    train_kwargs: Dict[str, Any],
    default_train_score: float,
    named_step: Tuple[str, stepwise.StepType],
) -> None:
    _check_optimize_train(params, train_kwargs, default_train_score, [named_step])


def test_train_all_steps(
    params: Dict[str, Any],
    train_kwargs: Dict[str, Any],
    default_train_score: float,
) -> None:
    _check_optimize_train(params, train_kwargs, default_train_score, default_lgb_steps())


def _check_optimize_cv(
    params: Dict[str, Any],
    cv_kwargs: Dict[str, Any],
    default_cv_score: float,
    steps: stepwise.StepListType,
) -> None:
    train_set = get_train()
    tuner = StepwiseLightGBMTunerCV(params, train_set=train_set, steps=steps, **cv_kwargs)
    tuner.optimize(n_trials=100)
    assert np.isclose(default_cv_score, tuner.best_value) or tuner.best_value <= default_cv_score

    if lgb.__version__ >= "3.0.0":
        model = tuner.get_best_booster()
        assert model.best_score == tuner.best_value

        for name, value in tuner.best_params.items():
            for booster in model.boosters:
                assert booster.params[name] == value


@pytest.mark.parametrize(
    "named_step",
    [named_step for named_step in default_lgb_steps()],
    ids=[step_name for step_name, _ in default_lgb_steps()],
)
def test_cv_single_step(
    params: Dict[str, Any],
    cv_kwargs: Dict[str, Any],
    default_cv_score: float,
    named_step: Tuple[str, stepwise.StepType],
) -> None:
    _check_optimize_cv(params, cv_kwargs, default_cv_score, [named_step])


def test_cv_all_steps(
    params: Dict[str, Any], cv_kwargs: Dict[str, Any], default_cv_score: float
) -> None:
    _check_optimize_cv(params, cv_kwargs, default_cv_score, default_lgb_steps())


def test_multiple_metrics(
    params: Dict[str, Any], train_kwargs: Dict[str, Any], dummy_steps: stepwise.StepListType
) -> None:
    expected = _optimize_train(params, train_kwargs, dummy_steps, n_trials=2).best_value

    params["metric"] = ["binary_logloss", "mape"]
    actual = _optimize_train(params, train_kwargs, dummy_steps, n_trials=2).best_value
    assert np.isclose(actual, expected)

    params["metric"] = ["mape", "binary_logloss"]
    actual = _optimize_train(params, train_kwargs, dummy_steps, n_trials=2).best_value
    assert not np.isclose(actual, expected)


def custom_obj(preds: np.ndarray, train_data: lgb.Dataset) -> Tuple[str, np.ndarray, bool]:
    y = train_data.get_label()
    error_sum = (preds - y).sum()
    return ("custom_obj", error_sum, False)


def test_train_feval(train_kwargs: Dict[str, Any]) -> None:
    params = {"objective": "binary", "metric": "None", "verbose": -1, "seed": SEED}
    train_kwargs["feval"] = custom_obj
    train_set, val_set = get_train_val()

    booster = lgb.train(
        params, train_set, valid_sets=val_set, valid_names=VALID_NAME, **train_kwargs
    )
    expected = booster.best_score[VALID_NAME]["custom_obj"]

    _check_optimize_train(
        params, train_kwargs, default_train_score=expected, steps=default_lgb_steps()
    )


def test_cv_feval(train_kwargs: Dict[str, Any]) -> None:
    params = {"objective": "binary", "metric": "None", "verbose": -1, "seed": SEED}
    train_kwargs["feval"] = custom_obj
    train_set = get_train()

    cv_results = lgb.cv(params, train_set, **train_kwargs)
    expected = cv_results["custom_obj-mean"][-1]

    _check_optimize_cv(params, train_kwargs, default_cv_score=expected, steps=default_lgb_steps())


def test_direction(params: Dict[str, Any], dummy_steps: stepwise.StepListType) -> None:
    params["metric"] = ["auc"]
    study = optuna.create_study(direction="minimize")
    with pytest.raises(ValueError, match="Study direction is inconsistent with the metric 'auc'"):
        train_set, val_set = get_train_val()
        tuner = StepwiseLightGBMTuner(
            params, train_set=train_set, steps=dummy_steps, study=study, valid_sets=val_set
        )
        tuner.optimize(n_trials=2)


def test_multiple_valid_sets(
    params: Dict[str, Any], train_kwargs: Dict[str, Any], dummy_steps: stepwise.StepListType
) -> None:
    train_set, val_set = get_train_val()
    tuner = StepwiseLightGBMTuner(
        params, train_set=train_set, valid_sets=val_set, steps=dummy_steps, **train_kwargs
    )
    tuner.optimize(n_trials=2)
    expected = tuner.best_value

    train_set, val_set = get_train_val()
    tuner = StepwiseLightGBMTuner(
        params,
        train_set=train_set,
        valid_sets=[val_set, train_set],
        steps=dummy_steps,
        **train_kwargs,
    )
    tuner.optimize(n_trials=2)
    actual = tuner.best_value

    assert np.isclose(actual, expected)


def test_valid_sets_contain_train(
    params: Dict[str, Any], train_kwargs: Dict[str, Any], dummy_steps: stepwise.StepListType
) -> None:
    X_train = np.zeros((80, 2))
    y_train = np.zeros(X_train.shape[0])
    X_test = np.ones((20, 2))
    y_test = np.ones(X_test.shape[0])

    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = train_set.create_valid(X_test, label=y_test, silent=True)

    train_kwargs = {
        "num_boost_round": 10,
        "early_stopping_rounds": 2,
        "verbose_eval": 1,
        "valid_sets": train_set,
    }
    tuner = StepwiseLightGBMTuner(params, train_set, steps=dummy_steps, **train_kwargs)
    tuner.optimize(n_trials=1)
    assert tuner.best_value == 0  # model was optimized on training set

    train_kwargs["valid_sets"] = [train_set, val_set, train_set]
    tuner = StepwiseLightGBMTuner(params, train_set, steps=dummy_steps, **train_kwargs)
    tuner.optimize(n_trials=1)

    assert tuner.best_value > 0  # model was NOT optimized on training set
