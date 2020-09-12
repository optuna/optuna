from optuna.integration._lightgbm_tuner.alias import _handling_alias_metrics
from optuna.integration._lightgbm_tuner.alias import _handling_alias_parameters


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
    assert "min_data_in_leaf" in params
    assert params["min_data_in_leaf"] == 0.2


def test_handling_alias_metrics() -> None:

    for alias in [
        "lambdarank",
        "rank_xendcg",
        "xendcg",
        "xe_ndcg",
        "xe_ndcg_mart",
        "xendcg_mart",
        "ndcg",
    ]:
        lgbm_params = {"metric": alias}
        _handling_alias_metrics(lgbm_params)
        assert lgbm_params["metric"] == "ndcg"

    for alias in ["mean_average_precision", "map"]:
        lgbm_params = {"metric": alias}
        _handling_alias_metrics(lgbm_params)
        assert lgbm_params["metric"] == "map"

    for alias in ["regression", "regression_l2", "l2", "mean_squared_error", "mse"]:
        lgbm_params = {"metric": alias}
        _handling_alias_metrics(lgbm_params)
        assert lgbm_params["metric"] == "l2"

    lgbm_params = {}
    _handling_alias_metrics(lgbm_params)
    assert lgbm_params == {}

    lgbm_params = {"metric": "auc"}
    _handling_alias_metrics(lgbm_params)
    assert lgbm_params["metric"] == "auc"

    lgbm_params = {"metric": "rmse"}
    _handling_alias_metrics(lgbm_params)
    assert lgbm_params["metric"] == "rmse"

    for alias in ["regression_l1", "l1", "mean_absolute_error", "mae"]:
        lgbm_params = {"metric": alias}
        _handling_alias_metrics(lgbm_params)
        assert lgbm_params["metric"] == "l1"

    for alias in ["binary_logloss", "binary"]:
        lgbm_params = {"metric": alias}
        _handling_alias_metrics(lgbm_params)
        assert lgbm_params["metric"] == "binary_logloss"

    for alias in [
        "multi_logloss",
        "multiclass",
        "softmax",
        "multiclassova",
        "multiclass_ova",
        "ova",
        "ovr",
    ]:
        lgbm_params = {"metric": alias}
        _handling_alias_metrics(lgbm_params)
        assert lgbm_params["metric"] == "multi_logloss"

    for alias in ["cross_entropy", "xentropy"]:
        lgbm_params = {"metric": alias}
        _handling_alias_metrics(lgbm_params)
        assert lgbm_params["metric"] == "cross_entropy"

    for alias in ["cross_entropy_lambda", "xentlambda"]:
        lgbm_params = {"metric": alias}
        _handling_alias_metrics(lgbm_params)
        assert lgbm_params["metric"] == "cross_entropy_lambda"

    for alias in ["kullback_leibler", "kldiv"]:
        lgbm_params = {"metric": alias}
        _handling_alias_metrics(lgbm_params)
        assert lgbm_params["metric"] == "kullback_leibler"

    for alias in ["mape", "mean_absolute_percentage_error"]:
        lgbm_params = {"metric": alias}
        _handling_alias_metrics(lgbm_params)
        assert lgbm_params["metric"] == "mape"

    for alias in ["auc_mu"]:
        lgbm_params = {"metric": alias}
        _handling_alias_metrics(lgbm_params)
        assert lgbm_params["metric"] == "auc_mu"

    for alias in ["none", "null", "custom", "na"]:
        lgbm_params = {"metric": alias}
        _handling_alias_metrics(lgbm_params)
        assert lgbm_params["metric"] == "custom"
