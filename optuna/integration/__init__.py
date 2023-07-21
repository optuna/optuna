import os
import sys
from types import ModuleType
from typing import Any
from typing import TYPE_CHECKING


_import_structure = {
    "allennlp": ["AllenNLPExecutor", "AllenNLPPruningCallback"],
    "botorch": ["BoTorchSampler"],
    "catalyst": ["CatalystPruningCallback"],
    "catboost": ["CatBoostPruningCallback"],
    "chainer": ["ChainerPruningExtension"],
    "chainermn": ["ChainerMNStudy"],
    "cma": ["CmaEsSampler", "PyCmaSampler"],
    "dask": ["DaskStorage"],
    "mlflow": ["MLflowCallback"],
    "wandb": ["WeightsAndBiasesCallback"],
    "keras": ["KerasPruningCallback"],
    "lightgbm": ["LightGBMPruningCallback", "LightGBMTuner", "LightGBMTunerCV"],
    "pytorch_distributed": ["TorchDistributedTrial"],
    "pytorch_ignite": ["PyTorchIgnitePruningHandler"],
    "pytorch_lightning": ["PyTorchLightningPruningCallback"],
    "sklearn": ["OptunaSearchCV"],
    "shap": ["ShapleyImportanceEvaluator"],
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


if TYPE_CHECKING:
    from optuna.integration.allennlp import AllenNLPExecutor
    from optuna.integration.allennlp import AllenNLPPruningCallback
    from optuna.integration.botorch import BoTorchSampler
    from optuna.integration.catalyst import CatalystPruningCallback
    from optuna.integration.catboost import CatBoostPruningCallback
    from optuna.integration.chainer import ChainerPruningExtension
    from optuna.integration.chainermn import ChainerMNStudy
    from optuna.integration.cma import CmaEsSampler
    from optuna.integration.cma import PyCmaSampler
    from optuna.integration.dask import DaskStorage
    from optuna.integration.fastaiv1 import FastAIV1PruningCallback
    from optuna.integration.fastaiv2 import FastAIPruningCallback
    from optuna.integration.fastaiv2 import FastAIV2PruningCallback
    from optuna.integration.keras import KerasPruningCallback
    from optuna.integration.lightgbm import LightGBMPruningCallback
    from optuna.integration.lightgbm import LightGBMTuner
    from optuna.integration.lightgbm import LightGBMTunerCV
    from optuna.integration.mlflow import MLflowCallback
    from optuna.integration.mxnet import MXNetPruningCallback
    from optuna.integration.pytorch_distributed import TorchDistributedTrial
    from optuna.integration.pytorch_ignite import PyTorchIgnitePruningHandler
    from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback
    from optuna.integration.shap import ShapleyImportanceEvaluator
    from optuna.integration.sklearn import OptunaSearchCV
    from optuna.integration.skopt import SkoptSampler
    from optuna.integration.skorch import SkorchPruningCallback
    from optuna.integration.tensorboard import TensorBoardCallback
    from optuna.integration.tensorflow import TensorFlowPruningHook
    from optuna.integration.tfkeras import TFKerasPruningCallback
    from optuna.integration.wandb import WeightsAndBiasesCallback
    from optuna.integration.xgboost import XGBoostPruningCallback
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

            try:
                return importlib.import_module("." + module_name, self.__name__)
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    "Optuna's integration modules for third-party libraries have started "
                    "migrating from Optuna itself to a package called `optuna-integration`. "
                    "The module you are trying to use has already been migrated to "
                    "`optuna-integration`. Please install the package by running "
                    "`pip install optuna-integration`."
                )

    sys.modules[__name__] = _IntegrationModule(__name__)

__all__ = [
    "AllenNLPExecutor",
    "AllenNLPPruningCallback",
    "BoTorchSampler",
    "CatalystPruningCallback",
    "CatBoostPruningCallback",
    "ChainerPruningExtension",
    "ChainerMNStudy",
    "CmaEsSampler",
    "PyCmaSampler",
    "DaskStorage",
    "MLflowCallback",
    "WeightsAndBiasesCallback",
    "KerasPruningCallback",
    "LightGBMPruningCallback",
    "LightGBMTuner",
    "LightGBMTunerCV",
    "TorchDistributedTrial",
    "PyTorchIgnitePruningHandler",
    "PyTorchLightningPruningCallback",
    "OptunaSearchCV",
    "ShapleyImportanceEvaluator",
    "SkorchPruningCallback",
    "MXNetPruningCallback",
    "SkoptSampler",
    "TensorBoardCallback",
    "TensorFlowPruningHook",
    "TFKerasPruningCallback",
    "XGBoostPruningCallback",
    "FastAIV1PruningCallback",
    "FastAIV2PruningCallback",
    "FastAIPruningCallback",
]
