from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest

from optuna.study import create_study
from optuna.testing.objectives import fail_objective
from optuna.trial import Trial
from optuna.visualization._optimization_history import _get_optimization_history_info_list
from optuna.visualization._optimization_history import _get_optimization_history_plot
from optuna.visualization._optimization_history import _OptimizationHistoryInfo
from optuna.visualization._optimization_history import _ValuesInfo


def test_target_is_none_and_study_is_multi_obj() -> None:
    study = create_study(directions=["minimize", "minimize"])
    with pytest.raises(ValueError):
        _get_optimization_history_info_list(
            study, target=None, target_name="Objective Value", error_bar=False
        )


@pytest.mark.parametrize("direction", ["minimize", "maximize"])
@pytest.mark.parametrize("error_bar", [False, True])
def test_warn_default_target_name_with_customized_target(direction: str, error_bar: bool) -> None:
    # Single study.
    study = create_study(direction=direction)
    with pytest.warns(UserWarning):
        _get_optimization_history_info_list(
            study, target=lambda t: t.number, target_name="Objective Value", error_bar=error_bar
        )

    # Multiple studies.
    studies = [create_study(direction=direction) for _ in range(10)]
    with pytest.warns(UserWarning):
        _get_optimization_history_info_list(
            studies, target=lambda t: t.number, target_name="Objective Value", error_bar=error_bar
        )


@pytest.mark.parametrize("direction", ["minimize", "maximize"])
@pytest.mark.parametrize("error_bar", [False, True])
def test_info_with_no_trials(direction: str, error_bar: bool) -> None:
    # Single study.
    study = create_study(direction=direction)
    info_list = _get_optimization_history_info_list(
        study, target=None, target_name="Objective Value", error_bar=error_bar
    )
    assert info_list == []

    # Multiple studies.
    studies = [create_study(direction=direction) for _ in range(10)]
    info_list = _get_optimization_history_info_list(
        studies, target=None, target_name="Objective Value", error_bar=error_bar
    )
    assert info_list == []


@pytest.mark.parametrize("direction", ["minimize", "maximize"])
@pytest.mark.parametrize("error_bar", [False, True])
def test_ignore_failed_trials(direction: str, error_bar: bool) -> None:
    # Single study.
    study = create_study(direction=direction)
    study.optimize(fail_objective, n_trials=1, catch=(ValueError,))
    info_list = _get_optimization_history_info_list(
        study, target=None, target_name="Objective Value", error_bar=error_bar
    )
    assert info_list == []

    # Multiple studies.
    studies = [create_study(direction=direction) for _ in range(10)]
    for study in studies:
        study.optimize(fail_objective, n_trials=1, catch=(ValueError,))
    info_list = _get_optimization_history_info_list(
        studies, target=None, target_name="Objective Value", error_bar=error_bar
    )
    assert info_list == []


@pytest.mark.parametrize("direction", ["minimize", "maximize"])
def test_get_optimization_history_info_list(direction: str) -> None:
    target_name = "Target Name"

    def objective(trial: Trial) -> float:
        if trial.number == 0:
            return 1.0
        elif trial.number == 1:
            return 2.0
        elif trial.number == 2:
            return 0.0
        return 0.0

    # Test with a trial.
    study = create_study(direction=direction)
    study.optimize(objective, n_trials=3)
    info_list = _get_optimization_history_info_list(
        study, target=None, target_name=target_name, error_bar=False
    )

    best_values = [1.0, 1.0, 0.0] if direction == "minimize" else [1.0, 2.0, 2.0]
    assert info_list == [
        _OptimizationHistoryInfo(
            [0, 1, 2],
            _ValuesInfo([1.0, 2.0, 0.0], None, target_name),
            _ValuesInfo(best_values, None, "Best Value"),
        )
    ]

    # Test customized target.
    info_list = _get_optimization_history_info_list(
        study, target=lambda t: t.number, target_name=target_name, error_bar=False
    )
    assert info_list == [
        _OptimizationHistoryInfo(
            [0, 1, 2],
            _ValuesInfo([0.0, 1.0, 2.0], None, target_name),
            None,
        )
    ]


@pytest.mark.parametrize("direction", ["minimize", "maximize"])
def test_get_optimization_history_info_list_with_multiple_studies(direction: str) -> None:
    n_studies = 10
    base_values = [1.0, 2.0, 0.0]
    base_best_values = [1.0, 1.0, 0.0] if direction == "minimize" else [1.0, 2.0, 2.0]
    target_name = "Target Name"

    # Test with trials.
    studies = [create_study(direction=direction) for _ in range(n_studies)]
    for i, study in enumerate(studies):
        study.optimize(lambda t: base_values[t.number] + i, n_trials=3)
    info_list = _get_optimization_history_info_list(
        studies, target=None, target_name=target_name, error_bar=False
    )

    for i, info in enumerate(info_list):
        values_i = [v + i for v in base_values]
        best_values_i = [v + i for v in base_best_values]
        assert info == _OptimizationHistoryInfo(
            [0, 1, 2],
            _ValuesInfo(values_i, None, f"{target_name} of {studies[i].study_name}"),
            _ValuesInfo(best_values_i, None, f"Best Value of {studies[i].study_name}"),
        )

    # Test customized target.
    info_list = _get_optimization_history_info_list(
        studies, target=lambda t: t.number, target_name=target_name, error_bar=False
    )
    for i, info in enumerate(info_list):
        assert info == _OptimizationHistoryInfo(
            [0, 1, 2],
            _ValuesInfo([0.0, 1.0, 2.0], None, f"{target_name} of {studies[i].study_name}"),
            None,
        )


@pytest.mark.parametrize("direction", ["minimize", "maximize"])
def test_get_optimization_history_info_list_with_error_bar(direction: str) -> None:
    n_studies = 10
    target_name = "Target Name"

    def objective(trial: Trial) -> float:
        if trial.number == 0:
            return 1.0
        elif trial.number == 1:
            return 2.0
        elif trial.number == 2:
            return 0.0
        return 0.0

    # Test with trials.
    studies = [create_study(direction=direction) for _ in range(n_studies)]
    for study in studies:
        study.optimize(objective, n_trials=3)
    info_list = _get_optimization_history_info_list(
        study, target=None, target_name=target_name, error_bar=True
    )

    best_values = [1.0, 1.0, 0.0] if direction == "minimize" else [1.0, 2.0, 2.0]
    assert info_list == [
        _OptimizationHistoryInfo(
            [0, 1, 2],
            _ValuesInfo([1.0, 2.0, 0.0], [0.0, 0.0, 0.0], target_name),
            _ValuesInfo(best_values, [0.0, 0.0, 0.0], "Best Value"),
        )
    ]

    # Test customized target.
    info_list = _get_optimization_history_info_list(
        study, target=lambda t: t.number, target_name=target_name, error_bar=True
    )
    assert info_list == [
        _OptimizationHistoryInfo(
            [0, 1, 2],
            _ValuesInfo([0.0, 1.0, 2.0], [0.0, 0.0, 0.0], target_name),
            None,
        )
    ]


@pytest.mark.parametrize("direction", ["minimize", "maximize"])
def test_error_bar_in_optimization_history(direction: str) -> None:
    def objective(trial: Trial) -> float:
        return trial.suggest_float("x", 0, 1)

    studies = [create_study(direction=direction) for _ in range(3)]
    suggested_params = [0.1, 0.3, 0.2]
    for x, study in zip(suggested_params, studies):
        study.enqueue_trial({"x": x})
        study.optimize(objective, n_trials=1)
    info_list = _get_optimization_history_info_list(
        studies, target=None, target_name="Objective Value", error_bar=True
    )
    mean = np.mean(suggested_params).item()
    std = np.std(suggested_params).item()
    assert info_list == [
        _OptimizationHistoryInfo(
            [0],
            _ValuesInfo([mean], [std], "Objective Value"),
            _ValuesInfo([mean], [std], "Best Value"),
        )
    ]


optimization_history_info_lists = [
    [],  # Empty info.
    [  # Vanilla.
        _OptimizationHistoryInfo(
            [0, 1, 2],
            _ValuesInfo([1.0, 2.0, 0.0], None, "Dummy"),
            None,
        )
    ],
    [  # Multiple.
        _OptimizationHistoryInfo(
            [0, 1, 2],
            _ValuesInfo([1.0, 2.0, 0.0], None, "Dummy"),
            None,
        ),
        _OptimizationHistoryInfo(
            [0, 1, 2],
            _ValuesInfo([1.0, 1.0, 1.0], None, "Dummy"),
            None,
        ),
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
]


@pytest.mark.parametrize("target_name", ["Objective Value", "Target Name"])
@pytest.mark.parametrize("info_list", optimization_history_info_lists)
def test_get_optimization_history_plot(
    target_name: str, info_list: list[_OptimizationHistoryInfo]
) -> None:
    figure = _get_optimization_history_plot(info_list, target_name=target_name)
    assert figure.layout.yaxis.title.text == target_name
    expected_legends = []
    for info in info_list:
        expected_legends.append(info.values_info.label_name)
        if info.best_values_info is not None:
            expected_legends.append(info.best_values_info.label_name)
    legends = [scatter.name for scatter in figure.data if scatter.name is not None]
    assert sorted(legends) == sorted(expected_legends)
    figure.write_image(BytesIO())
