from optuna.logging import get_logger
from optuna.visualization.matplotlib._matplotlib_imports import _imports

if _imports.is_successful():
    from optuna.visualization.matplotlib._matplotlib_imports import Axes

_logger = get_logger(__name__)


def plot_slice() -> Axes:
    raise NotImplementedError("To be implemented soon.")
