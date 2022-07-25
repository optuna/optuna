import pytest

from optuna.visualization._optimization_history import _OptimizationHistoryInfo
from optuna.visualization._optimization_history import _ValuesInfo
from optuna.visualization.matplotlib._optimization_history import _get_optimization_history_plot


@pytest.mark.parametrize("target_name", ["Objective Value", "Target Name"])
def test_get_optimization_history_plot(target_name: str) -> None:
    # Empty info.
    figure = _get_optimization_history_plot([], target_name=target_name)
    assert figure.get_ylabel() == target_name

    # Info with error bar.
    info_list = [
        _OptimizationHistoryInfo(
            [0, 1, 2],
            _ValuesInfo([1.0, 2.0, 0.0], [0.0, 0.0, 0.0], "Dummy"),
            _ValuesInfo([1.0, 1.0, 1.0], [0.0, 0.0, 0.0], "Best Value"),
        )
    ]
    figure = _get_optimization_history_plot(info_list, target_name)
    assert figure.get_ylabel() == target_name
