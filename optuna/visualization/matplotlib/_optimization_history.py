from optuna.logging import get_logger
from optuna.visualization.matplotlib._matplotlib_imports import _imports

_logger = get_logger(__name__)


if _imports.is_successful():
    from optuna.visualization.matplotlib._matplotlib_imports import Axes

    
def plot_optimization_history() -> Axes:
    raise NotImplementedError("To be implemented soon.")
