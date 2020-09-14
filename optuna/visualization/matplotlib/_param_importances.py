from optuna.visualization.matplotlib._matplotlib_imports import _imports


if _imports.is_successful():
    from optuna.visualization.matplotlib._matplotlib_imports import Axes


def plot_param_importances() -> Axes:
    raise NotImplementedError("To be implemented soon.")
