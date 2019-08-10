import os
import sys
from types import ModuleType
from typing import Any  # NOQA

from optuna.types import TYPE_CHECKING


_import_structure = {
    'chainer': ['ChainerPruningExtension'],
    'chainermn': ['ChainerMNStudy'],
    'cma': ['CmaEsSampler'],
    'keras': ['KerasPruningCallback'],
    'lightgbm': ['LightGBMPruningCallback'],
    'mxnet': ['MXNetPruningCallback'],
    'skopt': ['SkoptSampler'],
    'tensorflow': ['TensorFlowPruningHook'],
    'xgboost': ['XGBoostPruningCallback'],
}


__all__ = list(_import_structure.keys()) + sum(_import_structure.values(), [])


if sys.version_info[0] == 2 or TYPE_CHECKING:
    from optuna.integration.chainer import ChainerPruningExtension  # NOQA
    from optuna.integration.chainermn import ChainerMNStudy  # NOQA
    from optuna.integration.cma import CmaEsSampler  # NOQA
    from optuna.integration.keras import KerasPruningCallback  # NOQA
    from optuna.integration.lightgbm import LightGBMPruningCallback  # NOQA
    from optuna.integration.mxnet import MXNetPruningCallback  # NOQA
    from optuna.integration.skopt import SkoptSampler  # NOQA
    from optuna.integration.tensorflow import TensorFlowPruningHook  # NOQA
    from optuna.integration.xgboost import XGBoostPruningCallback  # NOQA
else:
    class _IntegrationModule(ModuleType):
        """Module class that implements `optuna.integration` package.

        This class applies lazy import under `optuna.integration`, where submodules are imported
        when they are actually accessed. Otherwise, `import optuna` becomes much slower because it
        imports all submodules and their dependencies (e.g., chainer, keras, lightgbm) all at once.
        """

        __file__ = globals()['__file__']
        __path__ = [os.path.dirname(__file__)]

        _modules = set(_import_structure.keys())
        _class_to_module = {}
        for key, values in _import_structure.items():
            for value in values:
                _class_to_module[value] = key

        def __getattr__(self, name):
            # type: (str) -> Any

            if name in self._modules:
                value = self._get_module(name)
            elif name in self._class_to_module.keys():
                module = self._get_module(self._class_to_module[name])
                value = getattr(module, name)
            else:
                raise AttributeError('module {} has no attribute {}'.format(self.__name__, name))

            setattr(self, name, value)
            return value

        def _get_module(self, module_name):
            # type: (str) -> ModuleType

            import importlib
            return importlib.import_module('.' + module_name, self.__name__)

    sys.modules[__name__] = _IntegrationModule(__name__)
