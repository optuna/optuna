from typing import List

import pytest

from optuna.visualization._optimization_history import _OptimizationHistoryInfo
from optuna.visualization._optimization_history import _ValuesInfo
from optuna.visualization.matplotlib._optimization_history import _get_optimization_history_plot


@pytest.mark.parametrize("target_name", ["Objective Value", "Target Name"])
@pytest.mark.parametrize(
    "info_list",
    [
        [],  # Empty info.
        [  # Vanilla.
            _OptimizationHistoryInfo(
                [0, 1, 2],
                _ValuesInfo([1.0, 2.0, 0.0], None, "Dummy"),
                None,
            )
        ],
        [  # With best values.
            _OptimizationHistoryInfo(
                [0, 1, 2],
                _ValuesInfo([1.0, 2.0, 0.0], None, "Dummy"),
                _ValuesInfo([1.0, 1.0, 1.0], None, "Best Value"),
            )
        ],
        [  # With error bar.
            _OptimizationHistoryInfo(
                [0, 1, 2],
                _ValuesInfo([1.0, 2.0, 0.0], [1.0, 2.0, 0.0], "Dummy"),
                None,
            )
        ],
        [  # With best values and error bar.
            _OptimizationHistoryInfo(
                [0, 1, 2],
                _ValuesInfo([1.0, 2.0, 0.0], [1.0, 2.0, 0.0], "Dummy"),
                _ValuesInfo([1.0, 1.0, 1.0], [1.0, 2.0, 0.0], "Best Value"),
            )
        ],
    ],
)
def test_get_optimization_history_plot(
    target_name: str, info_list: List[_OptimizationHistoryInfo]
) -> None:
    figure = _get_optimization_history_plot(info_list, target_name=target_name)
    assert figure.get_ylabel() == target_name
