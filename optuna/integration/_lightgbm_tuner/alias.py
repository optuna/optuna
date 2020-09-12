from typing import Any
from typing import Dict
from typing import List  # NOQA


_ALIAS_GROUP_LIST = [
    {"param_name": "bagging_fraction", "alias_names": ["sub_row", "subsample", "bagging"]},
    {"param_name": "learning_rate", "alias_names": ["shrinkage_rate", "eta"]},
    {
        "param_name": "min_data_in_leaf",
        "alias_names": ["min_data_per_leaf", "min_data", "min_child_samples"],
    },
    {
        "param_name": "min_sum_hessian_in_leaf",
        "alias_names": [
            "min_sum_hessian_per_leaf",
            "min_sum_hessian",
            "min_hessian",
            "min_child_weight",
        ],
    },
    {"param_name": "bagging_freq", "alias_names": ["subsample_freq"]},
    {"param_name": "feature_fraction", "alias_names": ["sub_feature", "colsample_bytree"]},
    {"param_name": "lambda_l1", "alias_names": ["reg_alpha"]},
    {"param_name": "lambda_l2", "alias_names": ["reg_lambda", "lambda"]},
    {"param_name": "min_gain_to_split", "alias_names": ["min_split_gain"]},
]  # type: List[Dict[str, Any]]


def _handling_alias_parameters(lgbm_params: Dict[str, Any]) -> None:
    """Handling alias parameters."""

    for alias_group in _ALIAS_GROUP_LIST:
        param_name = alias_group["param_name"]
        alias_names = alias_group["alias_names"]

        for alias_name in alias_names:
            if alias_name in lgbm_params:
                lgbm_params[param_name] = lgbm_params[alias_name]
                del lgbm_params[alias_name]


_ALIAS_METRIC_LIST = [
    {
        "metric_name": "ndcg",
        "alias_names": [
            "lambdarank",
            "rank_xendcg",
            "xendcg",
            "xe_ndcg",
            "xe_ndcg_mart",
            "xendcg_mart",
        ],
    },
    {"metric_name": "map", "alias_names": ["mean_average_precision"]},
    {
        "metric_name": "l2",
        "alias_names": ["regression", "regression_l2", "l2", "mean_squared_error", "mse"],
    },
    {
        "metric_name": "l1",
        "alias_names": ["regression_l1", "l1", "mean_absolute_error", "mae"],
    },
    {
        "metric_name": "binary_logloss",
        "alias_names": ["binary_logloss", "binary"],
    },
    {
        "metric_name": "multi_logloss",
        "alias_names": [
            "multi_logloss",
            "multiclass",
            "softmax",
            "multiclassova",
            "multiclass_ova",
            "ova",
            "ovr",
        ],
    },
    {
        "metric_name": "cross_entropy",
        "alias_names": ["cross_entropy", "xentropy"],
    },
    {
        "metric_name": "cross_entropy_lambda",
        "alias_names": ["cross_entropy_lambda", "xentlambda"],
    },
    {
        "metric_name": "kullback_leibler",
        "alias_names": ["kullback_leibler", "kldiv"],
    },
    {
        "metric_name": "mape",
        "alias_names": ["mape", "mean_absolute_percentage_error"],
    },
    {
        "metric_name": "auc_mu",
        "alias_names": ["auc_mu"],
    },
    {
        "metric_name": "custom",
        "alias_names": ["none", "null", "custom", "na"],
    },
]  # type: List[Dict[str, Any]]


def _handling_alias_metrics(lgbm_params: Dict[str, Any]) -> None:
    """Handling alias metrics."""

    if "metric" not in lgbm_params.keys():
        return

    for metric in _ALIAS_METRIC_LIST:
        metric_name = metric["metric_name"]
        alias_names = metric["alias_names"]

        for alias_name in alias_names:
            if lgbm_params["metric"] == alias_name:
                lgbm_params["metric"] = metric_name
                break
