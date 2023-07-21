from __future__ import annotations

from io import BytesIO

import pytest

from optuna.visualization._optimization_history import _OptimizationHistoryInfo
from optuna.visualization.matplotlib._matplotlib_imports import plt
from optuna.visualization.matplotlib._optimization_history import _get_optimization_history_plot
from tests.visualization_tests.test_optimization_history import optimization_history_info_lists


@pytest.mark.parametrize("target_name", ["Objective Value", "Target Name"])
@pytest.mark.parametrize("info_list", optimization_history_info_lists)
def test_get_optimization_history_plot(
    target_name: str, info_list: list[_OptimizationHistoryInfo]
) -> None:
    figure = _get_optimization_history_plot(info_list, target_name=target_name)
    assert figure.get_ylabel() == target_name
    expected_legends = []
    for info in info_list:
        expected_legends.append(info.values_info.label_name)
        if info.best_values_info is not None:
            expected_legends.append(info.best_values_info.label_name)
    legends = [legend.get_text() for legend in figure.legend().get_texts()]
    assert sorted(legends) == sorted(expected_legends)
    plt.savefig(BytesIO())
    plt.close()
