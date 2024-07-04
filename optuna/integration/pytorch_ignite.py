from optuna._imports import _INTEGRATION_IMPORT_ERROR_TEMPLATE


try:
    from optuna_integration.pytorch_ignite import PyTorchIgnitePruningHandler
except ModuleNotFoundError:
    raise ModuleNotFoundError(_INTEGRATION_IMPORT_ERROR_TEMPLATE.format("pytorch_ignite"))


__all__ = ["PyTorchIgnitePruningHandler"]
