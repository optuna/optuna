from __future__ import annotations

from collections.abc import Callable

from matplotlib.axes._axes import Axes
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pytest

import optuna
from optuna.study.study import ObjectiveFuncType
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_rank
from optuna.visualization import plot_slice
from optuna.visualization import plot_timeline
from optuna.visualization.matplotlib import (
    plot_optimization_history as matplotlib_plot_optimization_history,
)
from optuna.visualization.matplotlib import (
    plot_parallel_coordinate as matplotlib_plot_parallel_coordinate,
)
from optuna.visualization.matplotlib import (
    plot_param_importances as matplotlib_plot_param_importances,
)
from optuna.visualization.matplotlib import plot_contour as matplotlib_plot_contour
from optuna.visualization.matplotlib import plot_edf as matplotlib_plot_edf
from optuna.visualization.matplotlib import plot_rank as matplotlib_plot_rank
from optuna.visualization.matplotlib import plot_slice as matplotlib_plot_slice
from optuna.visualization.matplotlib import plot_timeline as matplotlib_plot_timeline


parametrize_visualization_functions_for_single_objective = pytest.mark.parametrize(
    "plot_func",
    [
        plot_optimization_history,
        plot_edf,
        plot_contour,
        plot_parallel_coordinate,
        plot_rank,
        plot_slice,
        plot_timeline,
        plot_param_importances,
        matplotlib_plot_optimization_history,
        matplotlib_plot_edf,
        matplotlib_plot_contour,
        matplotlib_plot_parallel_coordinate,
        matplotlib_plot_rank,
        matplotlib_plot_slice,
        matplotlib_plot_timeline,
        matplotlib_plot_param_importances,
    ],
)


def objective_single_dynamic_with_categorical(trial: optuna.Trial) -> float:
    category = trial.suggest_categorical("category", ["foo", "bar"])
    if category == "foo":
        return (trial.suggest_float("x1", 0, 10) - 2) ** 2
    else:
        return -((trial.suggest_float("x2", -10, 0) + 5) ** 2)


def objective_single_none_categorical(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -100, 100)
    trial.suggest_categorical("y", ["foo", None])
    return x**2


parametrize_single_objective_functions = pytest.mark.parametrize(
    "objective_func",
    [
        objective_single_dynamic_with_categorical,
        objective_single_none_categorical,
    ],
)


@parametrize_visualization_functions_for_single_objective
@parametrize_single_objective_functions
def test_visualizations_with_single_objectives(
    plot_func: Callable[[optuna.study.Study], go.Figure | Axes], objective_func: ObjectiveFuncType
) -> None:
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler())
    study.optimize(objective_func, n_trials=20)

    fig = plot_func(study)  # Must not raise an exception here.
    if isinstance(fig, Axes):
        plt.close()
