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
from optuna.trial import Trial


# C2-DTLZ2 minimization objective function with a single constraint.
# It is negated to be a maximization problem since BoTorch otherwise assumes maximization.
_OBJECTIVE = C2DTLZ2(dim=3, num_objectives=2, negate=True)


def objective(trial: Trial) -> Sequence[float]:
    xs = torch.tensor([trial.suggest_float(f"x{i}", 0, 1) for i in range(_OBJECTIVE.dim)])
    values = _OBJECTIVE(xs)
    constraint = -_OBJECTIVE.evaluate_slack(xs.unsqueeze(dim=0))[0]
    return values.tolist() + constraint.tolist()


def optimize_func(
    train_x: torch.Tensor, train_obj_and_con: torch.Tensor, bounds: torch.Tensor
) -> torch.Tensor:
    # We can always assume maximization in `optimize_func`.

    train_con = train_obj_and_con[:, -1]  # Given that the last column is a constraint.
    train_con = train_con.unsqueeze(dim=-1)
    train_obj = train_obj_and_con[:, :-1]

    is_feas = (train_con <= 0).all(dim=-1)

    # Debug.
    hv = Hypervolume(ref_point=_OBJECTIVE.ref_point)  # `Hypervolume` assumes maximization.
    feas_train_obj = train_obj[is_feas]
    pareto_mask = is_non_dominated(feas_train_obj)
    pareto_obj = feas_train_obj[pareto_mask]
    print("Pareto front size:", pareto_obj.numel(), "hypervolume:", hv.compute(pareto_obj))

    # Initialize and fit GP.
    model = SingleTaskGP(train_x, train_obj_and_con)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)

    # Optimize acquisition function.
    better_than_ref = (train_obj > _OBJECTIVE.ref_point).all(dim=-1)
    partitioning = NondominatedPartitioning(
        num_outcomes=_OBJECTIVE.num_objectives,
        Y=train_obj[better_than_ref & is_feas],
    )

    sampler = SobolQMCNormalSampler(num_samples=128)
    acqf = qExpectedHypervolumeImprovement(
        model=model,
        ref_point=_OBJECTIVE.ref_point.tolist(),
        partitioning=partitioning,
        sampler=sampler,
        objective=IdentityMCMultiOutputObjective(outcomes=[0, 1]),
        constraints=[lambda Z: Z[..., -1]],  # Given that the last column is a constraint.
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
        n_startup_trials=10,
    )
    study = optuna.multi_objective.create_study(
        directions=["maximize", "maximize", "maximize"],
        sampler=sampler,
    )
    study.optimize(objective, n_trials=30)

    print("Number of finished trials: ", len(study.trials))

    print("Pareto front:")

    trials = {str(trial.values): trial for trial in study.get_pareto_front_trials()}
    trials = list(trials.values())
    trials.sort(key=lambda t: t.values)

    for trial in trials:
        print("  Trial#{}".format(trial.number))
        print("    Values: Values={}, Constraint={}".format(trial.values[:-1], trial.values[-1]))
        print("    Params: {}".format(trial.params))
