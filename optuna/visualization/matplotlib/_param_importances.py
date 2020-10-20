from optuna._experimental import experimental
from optuna.visualization.matplotlib._matplotlib_imports import _imports


if _imports.is_successful():
    from optuna.visualization.matplotlib._matplotlib_imports import Axes


@experimental("2.2.0")
def plot_param_importances() -> Axes:
    raise NotImplementedError("To be implemented soon.")
