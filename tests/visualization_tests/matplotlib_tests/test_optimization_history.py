from typing import List

import pytest

from optuna.visualization._optimization_history import _OptimizationHistoryInfo
from optuna.visualization.matplotlib._optimization_history import _get_optimization_history_plot
from tests.visualization_tests.test_optimization_history import optimization_history_info_lists


@pytest.mark.parametrize("target_name", ["Objective Value", "Target Name"])
@pytest.mark.parametrize("info_list", optimization_history_info_lists)
def test_get_optimization_history_plot(
    target_name: str, info_list: List[_OptimizationHistoryInfo]
) -> None:
    figure = _get_optimization_history_plot(info_list, target_name=target_name)
    assert figure.get_ylabel() == target_name
