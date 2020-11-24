from typing import Sequence

from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.test_functions.multi_objective import C2DTLZ2
from botorch.utils.multi_objective.box_decomposition import NondominatedPartitioning
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
import torch

import optuna
from optuna.study import Study
from optuna.trial import Trial
from optuna.trial import FrozenTrial


# C2-DTLZ2 minimization objective function with a single constraint.
# It is negated to be a maximization problem since BoTorch otherwise assumes maximization.
_OBJECTIVE = C2DTLZ2(dim=3, num_objectives=2, negate=True)


def objective(trial: Trial) -> Sequence[float]:
    # xs = torch.tensor([trial.suggest_float(f"x{i}", 0, 1) for i in range(_OBJECTIVE.dim)])
    # values = _OBJECTIVE(xs)
    # constraint = _OBJECTIVE.evaluate_slack(xs.unsqueeze(dim=0))[0]
    # return values.tolist() + constraint.tolist()
    x = trial.suggest_float("x", 0, 10)
    y = trial.suggest_float("y", 0, 10)
    return x, y


def constraints(study: Study, trial: FrozenTrial) -> Sequence[float]:
    # Constraints are considered feasible if negative.
    x = trial.params["x"]
    y = trial.params["y"]
    return (x + y - 10,)


def optimize_func(
    train_x: torch.Tensor, train_obj: torch.Tensor, train_con: torch.Tensor, bounds: torch.Tensor
) -> torch.Tensor:
    # We can always assume maximization in `optimize_func`.

    # Initialize and fit GP.
    train_y = torch.cat([train_obj, train_con], dim=-1)
    model = SingleTaskGP(train_x, train_y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)

    # Optimize qEHVI acquisition function.

    # TODO(hvy): Use tighter reference point to reduce the size of `Y` by taking the intersection
    # of `is_feas` and points in `train_obj` better than the reference point.
    n_outcomes = train_obj.size(-1)
    is_feas = (train_con <= 0).all(dim=-1)
    partitioning = NondominatedPartitioning(num_outcomes=n_outcomes, Y=train_obj[is_feas])

    acqf_constraints = []
    n_contraints = train_con.size(1)
    for i in range(1, n_contraints + 1):
        acqf_constraints.append(lambda Z, i=i: Z[..., -i])

    ref_point = train_obj.amin(dim=0) - 1e-6
    ref_point = ref_point.tolist()

    acqf = qExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point,
        partitioning=partitioning,
        sampler=SobolQMCNormalSampler(num_samples=128),
        objective=IdentityMCMultiOutputObjective(outcomes=list(range(n_outcomes))),
        constraints=acqf_constraints,
    )

    candidates, _ = optimize_acqf(
        acq_function=acqf,
        bounds=bounds,
        q=1,
        num_restarts=20,
        raw_samples=1024,
        options={"batch_limit": 5, "maxiter": 200, "nonnegative": True},
        sequential=True,
    )

    return candidates


if __name__ == "__main__":
    sampler = optuna.integration.BoTorchSampler(
        optimize_func=optimize_func,
        constraints=constraints,
        n_startup_trials=10,
    )
    study = optuna.multi_objective.create_study(
        directions=["maximize", "maximize"],
        sampler=sampler,
    )
    study.optimize(objective, n_trials=64)

    print("Number of finished trials: ", len(study.trials))

    print("Pareto front:")

    trials = {str(trial.values): trial for trial in study.get_pareto_front_trials()}
    trials = list(trials.values())
    trials.sort(key=lambda t: t.values)

    for trial in trials:
        print("  Trial#{}".format(trial.number))
        print("    Values: Values={}, Constraint={}".format(trial.values[:-1], trial.values[-1]))
        print("    Params: {}".format(trial.params))

    fig = optuna.multi_objective.visualization.plot_pareto_front(
        study, include_dominated_trials=True
    )
    fig.show()
