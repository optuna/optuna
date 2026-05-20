from optuna import _deprecated
from optuna._imports import _INTEGRATION_IMPORT_ERROR_TEMPLATE
from optuna._warnings import optuna_warn


try:
    from optuna_integration.fastaiv2 import FastAIPruningCallback
    from optuna_integration.fastaiv2 import FastAIV2PruningCallback
except ModuleNotFoundError:
    raise ModuleNotFoundError(_INTEGRATION_IMPORT_ERROR_TEMPLATE.format("fastaiv2"))

msg = _deprecated._DEPRECATION_WARNING_TEMPLATE.format(
    name="`optuna.integration.fastaiv2`", d_ver="4.9.0", r_ver="6.0.0"
)
optuna_warn(f"{msg} Use `optuna_integration.fastaiv2` instead.", FutureWarning)


__all__ = ["FastAIV2PruningCallback", "FastAIPruningCallback"]
