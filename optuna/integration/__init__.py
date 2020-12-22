import os
import sys
from types import ModuleType
from typing import Any
from typing import TYPE_CHECKING


_import_structure = {
    "allennlp": ["AllenNLPExecutor", "AllenNLPPruningCallback"],
    "botorch": ["BoTorchSampler"],
    "catalyst": ["CatalystPruningCallback"],
    "chainer": ["ChainerPruningExtension"],
    "chainermn": ["ChainerMNStudy"],
    "cma": ["CmaEsSampler", "PyCmaSampler"],
    "mlflow": ["MLflowCallback"],
    "keras": ["KerasPruningCallback"],
    "lightgbm": ["LightGBMPruningCallback", "LightGBMTuner", "LightGBMTunerCV"],
    "pytorch_ignite": ["PyTorchIgnitePruningHandler"],
    "pytorch_lightning": ["PyTorchLightningPruningCallback"],
    "sklearn": ["OptunaSearchCV"],
    "skorch": ["SkorchPruningCallback"],
    "mxnet": ["MXNetPruningCallback"],
    "skopt": ["SkoptSampler"],
    "tensorboard": ["TensorBoardCallback"],
    "tensorflow": ["TensorFlowPruningHook"],
    "tfkeras": ["TFKerasPruningCallback"],
    "xgboost": ["XGBoostPruningCallback"],
    "fastaiv1": ["FastAIV1PruningCallback"],
    "fastaiv2": ["FastAIV2PruningCallback", "FastAIPruningCallback"],
}


__all__ = list(_import_structure.keys()) + sum(_import_structure.values(), [])


if TYPE_CHECKING:
    from optuna.integration.allennlp import AllenNLPExecutor  # NOQA
    from optuna.integration.allennlp import AllenNLPPruningCallback  # NOQA
    from optuna.integration.botorch import BoTorchSampler  # NOQA
    from optuna.integration.catalyst import CatalystPruningCallback  # NOQA
    from optuna.integration.chainer import ChainerPruningExtension  # NOQA
    from optuna.integration.chainermn import ChainerMNStudy  # NOQA
    from optuna.integration.cma import CmaEsSampler  # NOQA
    from optuna.integration.cma import PyCmaSampler  # NOQA
    from optuna.integration.fastaiv1 import FastAIV1PruningCallback  # NOQA
    from optuna.integration.fastaiv2 import FastAIPruningCallback  # NOQA
    from optuna.integration.fastaiv2 import FastAIV2PruningCallback  # NOQA
    from optuna.integration.keras import KerasPruningCallback  # NOQA
    from optuna.integration.lightgbm import LightGBMPruningCallback  # NOQA
    from optuna.integration.lightgbm import LightGBMTuner  # NOQA
    from optuna.integration.lightgbm import LightGBMTunerCV  # NOQA
    from optuna.integration.mlflow import MLflowCallback  # NOQA
    from optuna.integration.mxnet import MXNetPruningCallback  # NOQA
    from optuna.integration.pytorch_ignite import PyTorchIgnitePruningHandler  # NOQA
    from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback  # NOQA
    from optuna.integration.sklearn import OptunaSearchCV  # NOQA
    from optuna.integration.skopt import SkoptSampler  # NOQA
    from optuna.integration.skorch import SkorchPruningCallback  # NOQA
    from optuna.integration.tensorboard import TensorBoardCallback  # NOQA
    from optuna.integration.tensorflow import TensorFlowPruningHook  # NOQA
    from optuna.integration.tfkeras import TFKerasPruningCallback  # NOQA
    from optuna.integration.xgboost import XGBoostPruningCallback  # NOQA
else:

    class _IntegrationModule(ModuleType):
        """Module class that implements `optuna.integration` package.

        This class applies lazy import under `optuna.integration`, where submodules are imported
        when they are actually accessed. Otherwise, `import optuna` becomes much slower because it
        imports all submodules and their dependencies (e.g., chainer, keras, lightgbm) all at once.
        """

        __file__ = globals()["__file__"]
        __path__ = [os.path.dirname(__file__)]

        _modules = set(_import_structure.keys())
        _class_to_module = {}
        for key, values in _import_structure.items():
            for value in values:
                _class_to_module[value] = key

        def __getattr__(self, name: str) -> Any:

            if name in self._modules:
                value = self._get_module(name)
            elif name in self._class_to_module.keys():
                module = self._get_module(self._class_to_module[name])
                value = getattr(module, name)
            else:
                raise AttributeError("module {} has no attribute {}".format(self.__name__, name))

            setattr(self, name, value)
            return value

        def _get_module(self, module_name: str) -> ModuleType:

            import importlib

            return importlib.import_module("." + module_name, self.__name__)

    sys.modules[__name__] = _IntegrationModule(__name__)
