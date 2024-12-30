from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence
from io import BytesIO
from typing import Any

import pytest

from optuna.study import create_study
from optuna.testing.objectives import fail_objective
from optuna.trial import FrozenTrial
from optuna.trial import Trial
import optuna.visualization._intermediate_values
from optuna.visualization._intermediate_values import _get_intermediate_plot_info
from optuna.visualization._intermediate_values import _IntermediatePlotInfo
from optuna.visualization._intermediate_values import _TrialInfo
from optuna.visualization._plotly_imports import go
import optuna.visualization.matplotlib._intermediate_values
from optuna.visualization.matplotlib._matplotlib_imports import plt


def test_intermediate_plot_info() -> None:
    # Test with no trials.
    study = create_study(direction="minimize")

    assert _get_intermediate_plot_info(study) == _IntermediatePlotInfo(trial_infos=[])

    # Test with a trial with intermediate values.
    def objective(trial: Trial, report_intermediate_values: bool) -> float:
        if report_intermediate_values:
            trial.report(1.0, step=0)
            trial.report(2.0, step=1)
        return 0.0

    study = create_study()
    study.optimize(lambda t: objective(t, True), n_trials=1)

    assert _get_intermediate_plot_info(study) == _IntermediatePlotInfo(
        trial_infos=[
            _TrialInfo(
                trial_number=0, sorted_intermediate_values=[(0, 1.0), (1, 2.0)], feasible=True
            )
        ]
    )

    # Test a study with one trial with intermediate values and
    # one trial without intermediate values.
    # Expect the trial with no intermediate values to be ignored.
    study.optimize(lambda t: objective(t, False), n_trials=1)

    assert _get_intermediate_plot_info(study) == _IntermediatePlotInfo(
        trial_infos=[
            _TrialInfo(
                trial_number=0, sorted_intermediate_values=[(0, 1.0), (1, 2.0)], feasible=True
            )
        ]
    )

    # Test a study of only one trial that has no intermediate values.
    study = create_study()
    study.optimize(lambda t: objective(t, False), n_trials=1)
    assert _get_intermediate_plot_info(study) == _IntermediatePlotInfo(trial_infos=[])

    # Ignore failed trials.
    study = create_study()
    study.optimize(fail_objective, n_trials=1, catch=(ValueError,))
    assert _get_intermediate_plot_info(study) == _IntermediatePlotInfo(trial_infos=[])

    # Test a study with constraints
    def objective_with_constraints(trial: Trial) -> float:
        trial.set_user_attr("constraint", [trial.number % 2])

        trial.report(1.0, step=0)
        trial.report(2.0, step=1)
        return 0.0

    def constraints(trial: FrozenTrial) -> Sequence[float]:
        return trial.user_attrs["constraint"]

    study = create_study(sampler=optuna.samplers.NSGAIIISampler(constraints_func=constraints))
    study.optimize(objective_with_constraints, n_trials=2)
    assert _get_intermediate_plot_info(study) == _IntermediatePlotInfo(
        trial_infos=[
            _TrialInfo(
                trial_number=0, sorted_intermediate_values=[(0, 1.0), (1, 2.0)], feasible=True
            ),
            _TrialInfo(
                trial_number=1, sorted_intermediate_values=[(0, 1.0), (1, 2.0)], feasible=False
            ),
        ]
    )


@pytest.mark.parametrize(
    "plotter",
    [
        optuna.visualization._intermediate_values._get_intermediate_plot,
        optuna.visualization.matplotlib._intermediate_values._get_intermediate_plot,
    ],
)
@pytest.mark.parametrize(
    "info",
    [
        _IntermediatePlotInfo(trial_infos=[]),
        _IntermediatePlotInfo(
            trial_infos=[
                _TrialInfo(
                    trial_number=0, sorted_intermediate_values=[(0, 1.0), (1, 2.0)], feasible=True
                )
            ]
        ),
        _IntermediatePlotInfo(
            trial_infos=[
                _TrialInfo(
                    trial_number=0, sorted_intermediate_values=[(0, 1.0), (1, 2.0)], feasible=True
                ),
                _TrialInfo(
                    trial_number=1, sorted_intermediate_values=[(1, 2.0), (0, 1.0)], feasible=False
                ),
            ]
        ),
    ],
)
def test_plot_intermediate_values(
    plotter: Callable[[_IntermediatePlotInfo], Any], info: _IntermediatePlotInfo
) -> None:
    figure = plotter(info)
    if isinstance(figure, go.Figure):
        figure.write_image(BytesIO())
    else:
        plt.savefig(BytesIO())
        plt.close()
