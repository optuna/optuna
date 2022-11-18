from typing import cast
from typing import List
from typing import Union

import optuna
from optuna.samplers import BaseSampler
from optuna.samplers import CmaEsSampler
from optuna.samplers import RandomSampler
from optuna.samplers import TPESampler
from optuna.testing.storages import StorageSupplier


def parse_args(args: str) -> List[Union[int, str]]:
    ret: List[Union[int, str]] = []
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
    else:
        assert False


class OptimizeSuite:
    def objective(self, trial: optuna.Trial) -> float:
        x = trial.suggest_float("x", -100, 100)
        y = trial.suggest_int("y", -100, 100)
        return x**2 + y**2

    def optimize(self, storage_mode: str, sampler_mode: str, n_trials: int) -> None:
        with StorageSupplier(storage_mode) as storage:
            sampler = create_sampler(sampler_mode)
            study = optuna.create_study(storage=storage, sampler=sampler)
            study.optimize(self.objective, n_trials=n_trials)

    def time_optimize(self, args: str) -> None:
        storage_mode, sampler_mode, n_trials = parse_args(args)
        storage_mode = cast(str, storage_mode)
        sampler_mode = cast(str, sampler_mode)
        n_trials = cast(int, n_trials)
        self.optimize(storage_mode, sampler_mode, n_trials)

    params = (
        "inmemory, random, 1000",
        "inmemory, random, 10000",
        "inmemory, tpe, 1000",
        "inmemory, cmaes, 1000",
        "sqlite, random, 1000",
        "cached_sqlite, random, 1000",
    )
    param_names = ["storage, sampler, n_trials"]
    timeout = 600
