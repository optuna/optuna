from optuna import _deprecated
from optuna._imports import _INTEGRATION_IMPORT_ERROR_TEMPLATE
from optuna._warnings import optuna_warn


try:
    from optuna_integration.xgboost import XGBoostPruningCallback
except ModuleNotFoundError:
    raise ModuleNotFoundError(_INTEGRATION_IMPORT_ERROR_TEMPLATE.format("xgboost"))

msg = _deprecated._DEPRECATION_WARNING_TEMPLATE.format(
    name="`optuna.integration.xgboost`", d_ver="4.9.0", r_ver="6.0.0"
)
optuna_warn(f"{msg} Use `optuna_integration.xgboost` instead.", FutureWarning)


__all__ = ["XGBoostPruningCallback"]
