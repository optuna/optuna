from __future__ import annotations

from typing import cast

import optuna
from optuna.samplers import BaseSampler
from optuna.samplers import CmaEsSampler
from optuna.samplers import NSGAIISampler
from optuna.samplers import RandomSampler
from optuna.samplers import TPESampler
from optuna.testing.storages import StorageSupplier


def parse_args(args: str) -> list[int | str]:
    ret: list[int | str] = []
    for arg in map(lambda s: s.strip(), args.split(",")):
        try:
            ret.append(int(arg))
        except ValueError:
            ret.append(arg)
    return ret


SAMPLER_MODES = [
    "random",
    "tpe",
    "cmaes",
]


def create_sampler(sampler_mode: str) -> BaseSampler:
    if sampler_mode == "random":
        return RandomSampler()
    elif sampler_mode == "tpe":
        return TPESampler()
    elif sampler_mode == "cmaes":
        return CmaEsSampler()
    elif sampler_mode == "nsgaii":
        return NSGAIISampler()
    else:
        assert False


class OptimizeSuite:
    def objective(self, trial: optuna.Trial) -> float:
        x = trial.suggest_float("x", -100, 100)
        y = trial.suggest_int("y", -100, 100)
        return x**2 + y**2

    def multi_objective(self, trial: optuna.Trial) -> tuple[float, float]:
        x = trial.suggest_float("x", -100, 100)
        y = trial.suggest_int("y", -100, 100)
        return (x**2 + y**2, (x - 2) ** 2 + (y - 2) ** 2)

    def optimize(
        self, storage_mode: str, sampler_mode: str, n_trials: int, objective_type: str
    ) -> None:
        with StorageSupplier(storage_mode) as storage:
            sampler = create_sampler(sampler_mode)
            if objective_type == "single":
                directions = ["minimize"]
            elif objective_type == "multi":
                directions = ["minimize", "minimize"]
            else:
                assert False
            study = optuna.create_study(storage=storage, sampler=sampler, directions=directions)
            if objective_type == "single":
                study.optimize(self.objective, n_trials=n_trials)
            elif objective_type == "multi":
                study.optimize(self.multi_objective, n_trials=n_trials)
            else:
                assert False

    def time_optimize(self, args: str) -> None:
        storage_mode, sampler_mode, n_trials, objective_type = parse_args(args)
        storage_mode = cast(str, storage_mode)
        sampler_mode = cast(str, sampler_mode)
        n_trials = cast(int, n_trials)
        objective_type = cast(str, objective_type)
        self.optimize(storage_mode, sampler_mode, n_trials, objective_type)

    params = (
        "inmemory, random, 1000, single",
        "inmemory, random, 10000, single",
        "inmemory, tpe, 1000, single",
        "inmemory, cmaes, 1000, single",
        "sqlite, random, 1000, single",
        "sqlite, tpe, 1000, single",
        "sqlite, cmaes, 1000, single",
        "journal, random, 1000, single",
        "journal, tpe, 1000, single",
        "journal, cmaes, 1000, single",
        "inmemory, tpe, 1000, multi",
        "inmemory, nsgaii, 1000, multi",
        "sqlite, tpe, 1000, multi",
        "sqlite, nsgaii, 1000, multi",
        "journal, tpe, 1000, multi",
        "journal, nsgaii, 1000, multi",
    )
    param_names = ["storage, sampler, n_trials, objective_type"]
    timeout = 600
