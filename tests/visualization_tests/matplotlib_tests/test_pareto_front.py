from typing import Sequence
import pytest

import optuna
from optuna.trial import FrozenTrial
from optuna.visualization.matplotlib import plot_pareto_front


def test_constraints_func_experimental_warning() -> None:
    study = optuna.create_study(directions=["minimize", "minimize"])

    def constraints_func(t: FrozenTrial) -> Sequence[float]:
        return [1.0]

    with pytest.warns(optuna.exceptions.ExperimentalWarning):
        plot_pareto_front(
            study=study,
            constraints_func=constraints_func,
        )
