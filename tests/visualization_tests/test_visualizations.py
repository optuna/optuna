from __future__ import annotations

from typing import Callable

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

from .parametrize_objectives import parametrize_single_objective_functions


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


@parametrize_visualization_functions_for_single_objective
@parametrize_single_objective_functions
def test_visualization_with_dynamic_search_space(
    plot_func: Callable[[optuna.study.Study], None], objective_func: ObjectiveFuncType
):
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler())
    study.optimize(objective_func, n_trials=50)
    plot_func(study)  # Must not raise an exception here.
