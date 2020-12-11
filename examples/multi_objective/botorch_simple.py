from botorch.settings import suppress_botorch_warnings
from botorch.settings import validate_input_scaling
from botorch.test_functions.multi_objective import C2DTLZ2
import torch

import optuna


# C2-DTLZ2 minimization objective function with a single constraint.
# It is negated to be a maximization problem since BoTorch otherwise assumes maximization.
_OBJECTIVE = C2DTLZ2(dim=3, num_objectives=2, negate=True)


def objective(trial):
    xs = torch.tensor([trial.suggest_float(f"x{i}", 0, 1) for i in range(_OBJECTIVE.dim)])
    values = _OBJECTIVE(xs)

    constraint = _OBJECTIVE.evaluate_slack(xs.unsqueeze(dim=0))[0]
    trial.set_user_attr("constraint", constraint.tolist())

    return values.tolist()


def constraints(study, trial):
    return trial.user_attrs["constraint"]


if __name__ == "__main__":
    # Show warnings from BoTorch such as unnormalized input data warnings.
    suppress_botorch_warnings(False)
    validate_input_scaling(True)

    sampler = optuna.integration.BoTorchSampler(
        constraints_func=constraints,
        n_startup_trials=10,
    )
    study = optuna.multi_objective.create_study(
        directions=["maximize"] * _OBJECTIVE.num_objectives,
        sampler=sampler,
    )
    study.optimize(objective, n_trials=32)

    print("Number of finished trials: ", len(study.trials))

    print("Pareto front:")

    trials = {str(trial.values): trial for trial in study.get_pareto_front_trials()}
    trials = list(trials.values())
    trials.sort(key=lambda t: t.values)

    for trial in trials:
        print("  Trial#{}".format(trial.number))
        print(
            "    Values: Values={}, Constraint={}".format(
                trial.values, trial.user_attrs["constraint"][0]
            )
        )
        print("    Params: {}".format(trial.params))
