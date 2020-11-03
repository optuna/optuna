import botorch
from botorch.acquisition import UpperConfidenceBound
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.test_functions import Hartmann
from gpytorch.mlls import ExactMarginalLogLikelihood
import torch

import optuna
from optuna.samplers import RandomSampler


# Hartmann minimization objective function.
obj = Hartmann(dim=3)


def objective(trial):
    xs = [trial.suggest_float(f"x{i}", 0.00001, 1, log=True) for i in range(obj.dim)]
    #xs = [trial.suggest_float(f"x{i}", 1., 1.) for i in range(obj.dim)]
    return obj(torch.tensor(xs)).item()

'''
def optimize_func(
    train_x: torch.Tensor, train_obj: torch.Tensor, bounds: torch.Tensor
) -> torch.Tensor:
    # Input shapes are as follows.
    #
    # `train_x.shape`: (n_trials, n_params)
    # `train_obj.shape`: (n_trials, 1), 1 is the output dimension
    # `bounds.shape`: (n_params, 2), 2 is for low and high.
    assert all(isinstance(t, torch.Tensor) for t in [train_x, train_obj, bounds])
    assert train_x.ndim == 2
    assert train_x.shape[1] == obj.dim
    assert train_obj.ndim == 2
    assert train_obj.shape[0] == train_x.shape[0]
    assert bounds.shape == (2, obj.dim)

    print(bounds)

    # Initialize and fit GP.
    model = SingleTaskGP(train_x, train_obj)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)

    # Optimize acquisition function.
    acqf = UpperConfidenceBound(model, beta=0.1)
    candidates, _ = optimize_acqf(acqf, bounds=bounds, q=1, num_restarts=5, raw_samples=40)

    return candidates
'''



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

    #optuna.visualization.plot_optimization_history(study).show()
    optuna.visualization.plot_slice(study).show()
