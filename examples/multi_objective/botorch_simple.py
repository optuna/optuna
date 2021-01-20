from botorch.settings import suppress_botorch_warnings
from botorch.settings import validate_input_scaling

import optuna


def objective(trial):
    # Binh and Korn function with constraints.
    x = trial.suggest_float("x", -15, 30)
    y = trial.suggest_float("y", -15, 30)

    # Constraints which are considered feasible if less than or equal to zero.
    # The feasible region is basically the intersection of a circle centered at (x=5, y=0)
    # and the complement to a circle centered at (x=8, y=-3).
    c0 = (x - 5) ** 2 + y ** 2 - 25
    c1 = -((x - 8) ** 2) - (y + 3) ** 2 + 7.7

    # Store the constraints as user attributes so that they can be restored after optimization.
    trial.set_user_attr("constraint", (c0, c1))

    v0 = 4 * x ** 2 + 4 * y ** 2
    v1 = (x - 5) ** 2 + (y - 5) ** 2

    return v0, v1


def constraints(trial):
    return trial.user_attrs["constraint"]


if __name__ == "__main__":
    # Show warnings from BoTorch such as unnormalized input data warnings.
    suppress_botorch_warnings(False)
    validate_input_scaling(True)

    sampler = optuna.integration.BoTorchSampler(
        constraints_func=constraints,
        n_startup_trials=10,
    )
    study = optuna.create_study(
        directions=["minimize", "minimize"],
        sampler=sampler,
    )
    study.optimize(objective, n_trials=32, timeout=600)

    print("Number of finished trials: ", len(study.trials))

    print("Pareto front:")

    trials = sorted(study.best_trials, key=lambda t: t.values)

    for trial in trials:
        print("  Trial#{}".format(trial.number))
        print(
            "    Values: Values={}, Constraint={}".format(
                trial.values, trial.user_attrs["constraint"][0]
            )
        )
        print("    Params: {}".format(trial.params))
