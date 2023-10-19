from typing import List

import pytest

from optuna.integration._lightgbm_tuner.alias import _handling_alias_metrics
from optuna.integration._lightgbm_tuner.alias import _handling_alias_parameters


pytestmark = pytest.mark.integration


def test__handling_alias_parameters() -> None:
    params = {"reg_alpha": 0.1}
    _handling_alias_parameters(params)
    assert "reg_alpha" not in params
    assert "lambda_l1" in params


def test_handling_alias_parameter_with_user_supplied_param() -> None:
    params = {
        "num_boost_round": 5,
        "early_stopping_rounds": 2,
        "eta": 0.5,
    }
    _handling_alias_parameters(params)

    assert "eta" not in params
    assert "learning_rate" in params
    assert params["learning_rate"] == 0.5


def test_handling_alias_parameter() -> None:
    params = {
        "num_boost_round": 5,
        "early_stopping_rounds": 2,
        "min_data": 0.2,
    }
    _handling_alias_parameters(params)
    assert "min_data" not in params
    assert "min_child_samples" in params
    assert params["min_child_samples"] == 0.2


def test_handling_alias_parameter_duplication() -> None:
    params = {
        "num_boost_round": 5,
        "early_stopping_rounds": 2,
        "min_data": 0.2,
        "min_child_samples": 0.3,
        "l1_regularization": 0.0,
        "l2_regularization": 0.0,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
    }
    _handling_alias_parameters(params)
    # Here are the main alias names
    assert "min_child_samples" in params
    assert "lambda_l1" in params
    assert "lambda_l2" in params

    # None of them are the main alias names
    assert "min_data" not in params
    assert "l1_regularization" not in params
    assert "l2_regularization" not in params
    assert "reg_alpha" not in params
    assert "reg_lambda" not in params


@pytest.mark.parametrize(
    "aliases, expect",
    [
        (
            [
                "ndcg",
                "lambdarank",
                "rank_xendcg",
                "xendcg",
                "xe_ndcg",
                "xe_ndcg_mart",
                "xendcg_mart",
            ],
            "ndcg",
        ),
        (["mean_average_precision", "map"], "map"),
        (["rmse", "l2_root", "root_mean_squared_error"], "rmse"),
        (["l1", "regression_l1", "mean_absolute_error", "mae"], "l1"),
        (["l2", "regression", "regression_l2", "mean_squared_error", "mse"], "l2"),
        (["auc"], "auc"),
        (["binary_logloss", "binary"], "binary_logloss"),
        (
            [
                "multi_logloss",
                "multiclass",
                "softmax",
                "multiclassova",
                "multiclass_ova",
                "ova",
                "ovr",
            ],
            "multi_logloss",
        ),
        (["cross_entropy", "xentropy"], "cross_entropy"),
        (["cross_entropy_lambda", "xentlambda"], "cross_entropy_lambda"),
        (["kullback_leibler", "kldiv"], "kullback_leibler"),
        (["mape", "mean_absolute_percentage_error"], "mape"),
        (["auc_mu"], "auc_mu"),
        (["custom", "none", "null", "na"], "custom"),
        ([], None),  # If "metric" not in lgbm_params.keys(): return None.
        ([["lambdarank"]], ["ndcg"]),
        (
            [["lambdarank", "mean_average_precision", "root_mean_squared_error"]],
            ["ndcg", "map", "rmse"],
        ),
    ],
)
def test_handling_alias_metrics(aliases: List[str], expect: str) -> None:
    if len(aliases) > 0:
        for alias in aliases:
            lgbm_params = {"metric": alias}
            _handling_alias_metrics(lgbm_params)
            assert lgbm_params["metric"] == expect
    else:
        lgbm_params = {}
        _handling_alias_metrics(lgbm_params)
        assert lgbm_params == {}


def test_handling_unexpected_alias_metrics() -> None:
    with pytest.raises(ValueError):
        _handling_alias_metrics({"metric": 1})
