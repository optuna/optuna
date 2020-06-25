import importlib
import types
from typing import Any

from optuna import distributions  # NOQA
from optuna import exceptions  # NOQA
from optuna import importance  # NOQA
from optuna import integration  # NOQA
from optuna import logging  # NOQA
from optuna import multi_objective  # NOQA
from optuna import pruners  # NOQA
from optuna import samplers  # NOQA
from optuna import storages  # NOQA
from optuna import study  # NOQA
from optuna import trial  # NOQA
from optuna import version  # NOQA
from optuna import visualization  # NOQA

from optuna.exceptions import TrialPruned  # NOQA
from optuna.study import create_study  # NOQA
from optuna.study import delete_study  # NOQA
from optuna.study import get_all_study_summaries  # NOQA
from optuna.study import load_study  # NOQA
from optuna.study import Study  # NOQA
from optuna.trial import Trial  # NOQA
from optuna.version import __version__  # NOQA
from optuna.type_checking import TYPE_CHECKING  # NOQA


class _LazyImport(types.ModuleType):
    """Module wrapper for lazy import.

    This class wraps specified module and lazily import it when they are actually accessed.
    Otherwise, `import optuna` becomes slower because it imports all submodules and
    their dependencies (e.g., bokeh) all at once.
    Within this project's usage, importlib override this module's attribute on the first
    access and the imported submodule is directly accessed from the second access.

    Args:
        name: Name of module to apply lazy import.
    """

    def __init__(self, name: str) -> None:
        super(_LazyImport, self).__init__(name)
        self._name = name

    def _load(self) -> types.ModuleType:
        module = importlib.import_module(self._name)
        self.__dict__.update(module.__dict__)
        return module

    def __getattr__(self, item: str) -> Any:
        return getattr(self._load(), item)


if TYPE_CHECKING:
    from optuna import dashboard  # NOQA
else:
    dashboard = _LazyImport("optuna.dashboard")

structs = _LazyImport("optuna.structs")
