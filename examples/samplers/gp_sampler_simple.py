import math

import optuna
from optuna.samplers import GPSampler


def objective(trial):
    x = trial.suggest_float('x', -1., 1.)
    y = trial.suggest_float('y', 1e-2, 10.)
    return x ** 2 + math.log(y, 10)


if __name__ == '__main__':
    n_trials = 50
    sampler = GPSampler(
        optimizer_kwargs={'maxiter': 100, 'n_samples_for_anchor': 10, 'n_anchor': 1},
    )
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
