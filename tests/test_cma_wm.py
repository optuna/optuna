import optuna


def objective(trial):
    x = trial.suggest_float("x", 0, 10000)
    y = trial.suggest_int("y", 0, 100)
    return (x - 5) ** 2 + (y - 5) ** 2


if __name__ == "__main__":
    cmaes_sampler = optuna.samplers.CmaEsSampler(seed=42)
    cmaes_sampler_wm = optuna.samplers.CmaEsSampler(with_margin=True, seed=42)

    study_wm = optuna.create_study(sampler=cmaes_sampler_wm)
    study = optuna.create_study(sampler=cmaes_sampler)

    N = 1000

    study.optimize(objective, n_trials=N)
    study_wm.optimize(objective, n_trials=N)

    print("cmaes_sampler:")
    print(
        f"Best value: {study.best_value} (params: {study.best_params}, Trial number is {study.best_trial.number})"
    )
    print("cmaes_sampler_with_margin:")
    print(
        f"Best value: {study_wm.best_value} (params: {study_wm.best_params}, Trial number is {study_wm.best_trial.number})"
    )
    print(optuna.__version__)
    assert optuna.__version__ == "3.2.0.dev"