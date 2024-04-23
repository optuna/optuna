from optuna.samplers import nsgaii
from optuna.samplers._base import BaseSampler
from optuna.samplers._brute_force import BruteForceSampler
from optuna.samplers._cmaes import CmaEsSampler
from optuna.samplers._gp.sampler import GPSampler
from optuna.samplers._grid import GridSampler
from optuna.samplers._nsgaiii._sampler import NSGAIIISampler
from optuna.samplers._partial_fixed import PartialFixedSampler
from optuna.samplers._qmc import QMCSampler
from optuna.samplers._random import RandomSampler
from optuna.samplers._tpe.sampler import TPESampler
from optuna.samplers.nsgaii._sampler import NSGAIISampler


__all__ = [
    "BaseSampler",
    "BruteForceSampler",
    "CmaEsSampler",
    "GridSampler",
    "NSGAIISampler",
    "NSGAIIISampler",
    "PartialFixedSampler",
    "QMCSampler",
    "RandomSampler",
    "TPESampler",
    "GPSampler",
    "nsgaii",
]


class MOTPESampler:
    def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        # TODO(nabenabe0928): Come up with any ways to remove this file.
        # NOTE(nabenabe0928): Discuss when to remove this class.
        raise ImportError(
            "`MOTPESampler` was removed at v4.0.0. "
            "Please use `TPESampler` instead or downgrade your Optuna version."
        )
