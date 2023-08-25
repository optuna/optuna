from typing import Any
from typing import Dict
from typing import List  # NOQA


_ALIAS_GROUP_LIST: List[Dict[str, Any]] = [
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
]


def _handling_alias_parameters(lgbm_params: Dict[str, Any]) -> None:
    """Handling alias parameters."""

    for alias_group in _ALIAS_GROUP_LIST:
        param_name = alias_group["param_name"]
        alias_names = alias_group["alias_names"]

        for alias_name in alias_names:
            if alias_name in lgbm_params:
                lgbm_params[param_name] = lgbm_params[alias_name]
                del lgbm_params[alias_name]


_ALIAS_METRIC_LIST: List[Dict[str, Any]] = [
    # The list `alias_names` do not include the `metric_name` itself.
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
        "alias_names": ["regression", "regression_l2", "mean_squared_error", "mse"],
    },
    {
        "metric_name": "l1",
        "alias_names": ["regression_l1", "mean_absolute_error", "mae"],
    },
    {
        "metric_name": "binary_logloss",
        "alias_names": ["binary"],
    },
    {
        "metric_name": "multi_logloss",
        "alias_names": [
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
        "alias_names": ["xentropy"],
    },
    {
        "metric_name": "cross_entropy_lambda",
        "alias_names": ["xentlambda"],
    },
    {
        "metric_name": "kullback_leibler",
        "alias_names": ["kldiv"],
    },
    {
        "metric_name": "mape",
        "alias_names": ["mean_absolute_percentage_error"],
    },
    {
        "metric_name": "custom",
        "alias_names": ["none", "null", "na"],
    },
    {
        "metric_name": "rmse",
        "alias_names": ["l2_root", "root_mean_squared_error"],
    },
]

_ALIAS_METRIC_MAP: Dict[str, str] = {
    alias_name: canonical_metric["metric_name"]
    for canonical_metric in _ALIAS_METRIC_LIST
    for alias_name in canonical_metric["alias_names"]
}


def _handling_alias_metrics(lgbm_params: Dict[str, Any]) -> None:
    """Handling alias metrics."""
    if "metric" not in lgbm_params.keys():
        return

    if isinstance(lgbm_params["metric"], str):
        lgbm_params["metric"] = (
            _ALIAS_METRIC_MAP.get(lgbm_params["metric"]) or lgbm_params["metric"]
        )
        return

    if not isinstance(lgbm_params["metric"], list):
        raise ValueError(
            "The `metric` parameter is expected to be a list or a string, but got "
            f"{type(lgbm_params['metric'])}."
        )

    canonical_metrics = []
    for metric in lgbm_params["metric"]:
        canonical_metrics.append(_ALIAS_METRIC_MAP.get(metric) or metric)
    lgbm_params["metric"] = canonical_metrics
