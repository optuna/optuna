from optuna import _deprecated
from optuna._imports import _INTEGRATION_IMPORT_ERROR_TEMPLATE
from optuna._warnings import optuna_warn


try:
    from optuna_integration.tensorflow import TensorFlowPruningHook
except ModuleNotFoundError:
    raise ModuleNotFoundError(_INTEGRATION_IMPORT_ERROR_TEMPLATE.format("tensorflow"))

msg = _deprecated._DEPRECATION_WARNING_TEMPLATE.format(
    name="`optuna.integration.tensorflow`", d_ver="4.9.0", r_ver="6.0.0"
)
optuna_warn(f"{msg} Use `optuna_integration.tensorflow` instead.", FutureWarning)


__all__ = ["TensorFlowPruningHook"]
