import importlib
import types
from typing import Any
from typing import TYPE_CHECKING

from optuna import distributions
from optuna import exceptions
from optuna import importance
from optuna import integration
from optuna import logging
from optuna import multi_objective
from optuna import pruners
from optuna import samplers
from optuna import storages
from optuna import study
from optuna import trial
from optuna import version
from optuna import visualization
from optuna.exceptions import TrialPruned
from optuna.study import create_study
from optuna.study import delete_study
from optuna.study import get_all_study_summaries
from optuna.study import load_study
from optuna.study import Study
from optuna.trial import create_trial
from optuna.trial import Trial
from optuna.version import __version__


__all__ = [
    "Study",
    "TYPE_CHECKING",
    "Trial",
    "TrialPruned",
    "__version__",
    "create_study",
    "create_trial",
    "delete_study",
    "distributions",
    "exceptions",
    "get_all_study_summaries",
    "importance",
    "integration",
    "load_study",
    "logging",
    "multi_objective",
    "pruners",
    "samplers",
    "storages",
    "study",
    "trial",
    "version",
    "visualization",
]


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
type_checking = _LazyImport("optuna.type_checking")
