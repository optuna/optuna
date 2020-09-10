from typing import Tuple

import numpy as np

import optuna
from optuna.batch.multi_objective.trial import BatchMultiObjectiveTrial


def test_suggest() -> None:
    def objective(trial: BatchMultiObjectiveTrial) -> Tuple[np.ndarray, np.ndarray]:
        p0 = trial.suggest_float("p0", -10, 10)
        p1 = trial.suggest_uniform("p1", 3, 5)
        p2 = trial.suggest_loguniform("p2", 0.00001, 0.1)
        p3 = trial.suggest_discrete_uniform("p3", 100, 200, q=5)
        p4 = trial.suggest_int("p4", -20, -15)
        p5 = trial.suggest_categorical("p5", [7, 1, 100]).astype(np.int64)
        p6 = trial.suggest_float("p6", -10, 10, step=1.0)
        p7 = trial.suggest_int("p7", 1, 7, log=True)
        return (p0 + p1 + p2, p3 + p4 + p5 + p6 + p7)

    study = optuna.batch.multi_objective.create_study(["maximize", "maximize"])
    study.optimize(objective, n_batches=3, batch_size=2)


def test_report() -> None:
    batch_size = 3

    def objective(trial: BatchMultiObjectiveTrial) -> np.ndarray:
        if trial._trials[0].number == 0:
            trial.report([np.ones(batch_size), 2 * np.ones(batch_size)], 1)
            trial.report([10 * np.ones(batch_size), 20 * np.ones(batch_size)], 2)
        return 100 * np.ones(batch_size), 200 * np.ones(batch_size)

    study = optuna.batch.multi_objective.create_study(
        ["minimize", "minimize"]
    )
    study.optimize(objective, n_batches=2, batch_size=batch_size)

    for i in range(batch_size):
        trial = study.trials[i]
        assert trial.intermediate_values == {1: (1, 2), 2: (10, 20)}
        assert trial.values == (100, 200)
        assert trial.last_step == 2

    for i in range(batch_size, 2 * batch_size):
        trial = study.trials[i]
        assert trial.intermediate_values == {}
        assert trial.values == (100, 200)
        assert trial.last_step is None
