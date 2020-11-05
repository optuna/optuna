from optuna.samplers._tpe.sampler import default_gamma
import optuna
import optuna.samplers._stepwise


def objective(trial):
    x = trial.suggest_float("x", 0, 100)
    y = trial.suggest_float("y", 0, 100)
    z = trial.suggest_float("z", 0, 100)
    return (x - 20) ** 2 + (y - 80) ** 2 + (z - 50) ** 2


def search_x(params):
    return {"x": optuna.distributions.UniformDistribution(0, 100)}


def search_y(params):
    return {"y": optuna.distributions.UniformDistribution(0, 100)}


def search_z(params):
    return {"z": optuna.distributions.UniformDistribution(0, 100)}

def search_x_y(params):
    return {"x": optuna.distributions.UniformDistribution(0, 100), "y": optuna.distributions.UniformDistribution(0, 100)}


steps = [
    optuna.samplers._stepwise.Step(search_x_y, optuna.samplers.CmaEsSampler(), n_trials=25),
    optuna.samplers._stepwise.Step(search_x, optuna.samplers.TPESampler(), n_trials=50),
    optuna.samplers._stepwise.Step(search_y, optuna.samplers.RandomSampler(), n_trials=25),
    optuna.samplers._stepwise.Step(search_z, optuna.samplers.TPESampler(), n_trials=25),
    optuna.samplers._stepwise.Step(search_x_y, optuna.samplers.CmaEsSampler(), n_trials=25),
]

sampler = optuna.samplers._stepwise.StepwiseSampler(steps=steps, default_params={"x": 0, "y": 0, "z": 0})
study = optuna.create_study(sampler=sampler)
study.optimize(objective)

print(study.best_trial)
print(study.trials_dataframe(["number", "value", "params"]))
