from typing import TypeVar

import optuna


_T = TypeVar("_T")


def objective(trial: optuna.Trial) -> float:
    return trial.suggest_float("x", 0, 1)


def op(t: _T) -> _T:
    return t


tt = op(optuna.samplers.QMCSampler)
print(type(tt))



# sampler = optuna.samplers.QMCSampler()
# study = optuna.create_study()
