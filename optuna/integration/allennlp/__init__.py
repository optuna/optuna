from optuna import _deprecated
from optuna._imports import _INTEGRATION_IMPORT_ERROR_TEMPLATE
from optuna._warnings import optuna_warn


try:
    from optuna_integration.allennlp._dump_best_config import dump_best_config
    from optuna_integration.allennlp._executor import AllenNLPExecutor
    from optuna_integration.allennlp._pruner import AllenNLPPruningCallback
except ModuleNotFoundError:
    raise ModuleNotFoundError(_INTEGRATION_IMPORT_ERROR_TEMPLATE.format("allennlp"))

msg = _deprecated._DEPRECATION_WARNING_TEMPLATE.format(
    name="`optuna.integration.allennlp`", d_ver="4.9.0", r_ver="6.0.0"
)
optuna_warn(f"{msg} Use `optuna_integration.allennlp` instead.", FutureWarning)


__all__ = ["dump_best_config", "AllenNLPExecutor", "AllenNLPPruningCallback"]
