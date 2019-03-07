import os
import sys
from types import ModuleType
from typing import Any  # NOQA


__all__ = [
    'chainer',
    'chainermn'
    'keras',
    'lightgbm',
    'tensorflow',
    'xgboost',
    'ChainerMNStudy'
    'ChainerPruningExtension',
    'KerasPruningCallback',
    'LightGBMPruningCallback',
    'TensorFlowPruningHook',
    'XGBoostPruningCallback',
]


class _IntegrationModule(ModuleType):

    __path__ = [os.path.dirname(__file__)]

    _modules = {
        'chainer',
        'chainermn',
        'keras',
        'lightgbm',
        'tensorflow',
        'xgboost',
    }

    _class_to_module = {
        'ChainerMNStudy': 'chainermn',
        'ChainerPruningExtension': 'chainer',
        'KerasPruningCallback': 'keras',
        'LightGBMPruningCallback': 'lightgbm',
        'TensorFlowPruningHook': 'tensorflow',
        'XGBoostPruningCallback': 'xgboost',
    }

    def __getattr__(self, name):
        # type: (str) -> Any

        if name in self._modules:
            return self._get_module(name)
        elif name in self._class_to_module.keys():
            module = self._get_module(self._class_to_module[name])
            return getattr(module, name)

        raise AttributeError('module {} has no attribute {}'.format(self.__name__, name))

    if sys.version_info[0] == 2:
        def _get_module(self, module_name):
            # type: (str) -> ModuleType

            import imp
            file, path_name, description = imp.find_module(module_name, self.__path__)
            return imp.load_module(module_name, file, path_name, description)
    else:
        def _get_module(self, module_name):
            # type: (str) -> ModuleType

            import importlib
            return importlib.import_module('.' + module_name, self.__name__)


sys.modules[__name__] = _IntegrationModule(__name__)
