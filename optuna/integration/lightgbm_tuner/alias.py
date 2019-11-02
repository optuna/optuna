from optuna import type_checking

if type_checking.TYPE_CHECKING:
    from typing import Any  # NOQA
    from typing import Dict  # NOQA
    from typing import List  # NOQA


ALIAS_GROUP_LIST = [
    {
        'param_name': 'bagging_fraction',
        'alias_names': ['sub_row', 'subsample', 'bagging'],
    },
    {
        'param_name': 'learning_rate',
        'alias_names': ['shrinkage_rate', 'eta'],
    },
    {
        'param_name': 'min_data_in_leaf',
        'alias_names': ['min_data_per_leaf', 'min_data', 'min_child_samples'],
    },
    {
        'param_name': 'min_sum_hessian_in_leaf',
        'alias_names': ['min_sum_hessian_per_leaf', 'min_sum_hessian',
                        'min_hessian', 'min_child_weight'],
    },
    {
        'param_name': 'bagging_freq',
        'alias_names': ['subsample_freq'],
    },
    {
        'param_name': 'feature_fraction',
        'alias_names': ['sub_feature', 'colsample_bytree'],
    },
    {
        'param_name': 'lambda_l1',
        'alias_names': ['reg_alpha'],
    },
    {
        'param_name': 'lambda_l2',
        'alias_names': ['reg_lambda', 'lambda'],
    },
    {
        'param_name': 'min_gain_to_split',
        'alias_names': ['min_split_gain'],
    },
]  # type: List[Dict[str, Any]]


def _handling_alias_parameters(lgbm_params):
    # type: (Dict[str, Any]) -> None
    """Handling alias parameters."""

    for alias_group in ALIAS_GROUP_LIST:
        param_name = alias_group['param_name']
        alias_names = alias_group['alias_names']

        for alias_name in alias_names:
            if alias_name in lgbm_params:
                lgbm_params[param_name] = lgbm_params[alias_name]
                del lgbm_params[alias_name]
