import os
import sys
from types import ModuleType
from typing import Any
from typing import TYPE_CHECKING

from optuna import _deprecated
from optuna._imports import _INTEGRATION_IMPORT_ERROR_TEMPLATE
from optuna._warnings import optuna_warn


try:
    import optuna_integration.lightgbm as lgb
except ModuleNotFoundError:
    raise ModuleNotFoundError(_INTEGRATION_IMPORT_ERROR_TEMPLATE.format("lightgbm"))

msg = _deprecated._DEPRECATION_WARNING_TEMPLATE.format(
    name="`optuna.integration.lightgbm`", d_ver="4.9.0", r_ver="6.0.0"
)
optuna_warn(f"{msg} Use `optuna_integration.lightgbm` instead.", FutureWarning)


if TYPE_CHECKING:
    # These modules are from optuna-integration.
    from optuna.integration.lightgbm_tuner import LightGBMPruningCallback
    from optuna.integration.lightgbm_tuner import LightGBMTuner
    from optuna.integration.lightgbm_tuner import LightGBMTunerCV
    from optuna.integration.lightgbm_tuner import train


__all__ = [
    "LightGBMPruningCallback",
    "LightGBMTuner",
    "LightGBMTunerCV",
    "train",
]


class _LightGBMModule(ModuleType):
    """Module class that implements `optuna.integration.lightgbm` package."""

    __all__ = __all__
    __file__ = globals()["__file__"]
    __path__ = [os.path.dirname(__file__)]

    def __getattr__(self, name: str) -> Any:
        return lgb.__dict__[name]


sys.modules[__name__] = _LightGBMModule(__name__)
