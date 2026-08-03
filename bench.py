import optunahub
import torch  # noqa: F401  # import first to reduce the overhead later.

import optuna
from optuna._gp import prior


bbob = optunahub.load_module("benchmarks/bbob")


def run_gp_sampler(trial: optuna.Trial) -> float:
    function_id = trial.suggest_int("function_id", 1, 24)
    dimension = trial.suggest_categorical("dimension", [2, 5, 10])
    seed = trial.suggest_int("seed", 0, 9)
    problem = bbob.Problem(function_id=function_id, dimension=dimension)
    prior_type = trial.suggest_categorical("prior_type", ["optuna", "hvarfner"])
    sampler = optuna.samplers.GPSampler(seed=seed)
    sampler._kernel_type = trial.suggest_categorical("kernel_type", ["rbf", "matern"])
    if prior_type == "hvarfner":
        sampler._log_prior = prior.hvarfner_log_prior
        sampler._kernel_param_bounds = prior.HVARFNER_KERNEL_PARAM_BOUNDS
    elif prior_type != "optuna":
        assert False
    # prior_type == "optuna" needs no change; the defaults are
    # prior.default_log_prior + prior.DEFAULT_KERNEL_PARAM_BOUNDS.
    study = optuna.create_study(sampler=sampler)
    study.optimize(problem, n_trials=200)
    trials = study.get_trials(deepcopy=False)
    trial.set_user_attr("values", [t.value for t in trials])
    return 0.0  # placeholder


journal_file = optuna.storages.journal.JournalFileBackend("prior-benchmark.log")
storage = optuna.storages.JournalStorage(journal_file)
sampler = optuna.samplers.BruteForceSampler()
study = optuna.create_study(sampler=sampler, storage=storage, load_if_exists=True)
study.optimize(run_gp_sampler)
