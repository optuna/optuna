from optuna.samplers._gp.acquisition.base import BaseAcquisitionFunction
from optuna.samplers._gp.acquisition.ei import EI


def acquisition_selector(acquisition: str) -> BaseAcquisitionFunction:
    """Selector module for acquisition functions."""

    if acquisition == 'EI':
        return EI()
    else:
        raise ValueError("The acquisition function {} is not supported.".format(acquisition))
