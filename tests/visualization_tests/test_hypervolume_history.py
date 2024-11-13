from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from optuna.samplers import NSGAIISampler
from optuna.study import create_study
from optuna.trial import FrozenTrial
from optuna.trial import Trial
from optuna.visualization._hypervolume_history import _get_hypervolume_history_info
from optuna.visualization._hypervolume_history import _HypervolumeHistoryInfo


@pytest.mark.parametrize(
    "directions",
    [
        ["minimize", "minimize"],
        ["minimize", "maximize"],
        ["maximize", "minimize"],
        ["maximize", "maximize"],
    ],
)
def test_get_optimization_history_info(directions: str) -> None:
    signs = [1 if d == "minimize" else -1 for d in directions]

    def objective(trial: Trial) -> Sequence[float]:
        def impl(trial: Trial) -> Sequence[float]:
            if trial.number == 0:
                return 1.5, 1.5  # dominated by the reference_point
            elif trial.number == 1:
                return 0.75, 0.75
            elif trial.number == 2:
                return 0.5, 0.5  # dominates Trial #1
            elif trial.number == 3:
                return 0.5, 0.5  # dominates Trial #1
            elif trial.number == 4:
                return 0.75, 0.25  # incomparable
            return 0.0, 0.0  # dominates all

        values = impl(trial)
        return signs[0] * values[0], signs[1] * values[1]

    def constraints(trial: FrozenTrial) -> Sequence[float]:
        if trial.number == 2:
            return (1,)  # infeasible

        return (0,)  # feasible

    sampler = NSGAIISampler(constraints_func=constraints)
    study = create_study(directions=directions, sampler=sampler)
    study.optimize(objective, n_trials=6)

    reference_point = np.asarray(signs)
    info = _get_hypervolume_history_info(study, reference_point)
    assert info == _HypervolumeHistoryInfo(
        trial_numbers=[0, 1, 2, 3, 4, 5], values=[0.0, 0.0625, 0.0625, 0.25, 0.3125, 1.0]
    )
