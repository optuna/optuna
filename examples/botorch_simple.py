from botorch.test_functions import Hartmann
import torch

import optuna
from optuna.trial import Trial
from optuna.samplers import RandomSampler


# Hartmann minimization objective function.
obj = Hartmann(dim=6)


def objective(trial: Trial) -> float:
    xs = [trial.suggest_float(f"x{i}", 0, 1) for i in range(obj.dim)]
    return obj(torch.tensor(xs)).item()


# def objective(trial: Trial) -> float:
#     x0 = trial.suggest_float("x0", 0, 2)
#     x1 = trial.suggest_float("x1", 1, 2, log=True)
#     x2 = trial.suggest_float("x2", 0, 1, step=0.2)
#     x3 = trial.suggest_int("x3", 1, 3)
#     x4 = trial.suggest_int("x4", 1, 10, log=True)
#     x5 = trial.suggest_int("x5", 0, 10, step=2)
#     x6 = float(trial.suggest_categorical("x6", [2, 4, 8]))
#
#     return x0 + x1 + x2 + x3 + x4 + x5 + x6


if __name__ == "__main__":
    sampler = optuna.integration.BoTorchSampler(
        n_startup_trials=10,
        independent_sampler=RandomSampler(),
    )
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=32)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # optuna.visualization.plot_slice(study).show()
    optuna.visualization.plot_optimization_history(study).show()
