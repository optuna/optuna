from typing import List

from optuna.distributions import LogUniformDistribution
from optuna.trial import FrozenTrial
from optuna.visualization.matplotlib import _matplotlib_imports

__all__ = ["is_available"]


def is_available() -> bool:
    """Returns whether visualization with `matplotlib` is available or not.

    .. note::

        :mod:`~optuna.visualization` module depends on Matplotlib version 3.0.0 or higher. If a
        supported version of Matplotlib isn't installed in your environment, this function will
        return :obj:`False`. In such a case, please execute ``$ pip install -U matplotlib>=3.0.0``
        to install Matplotlib.

    Returns:
        :obj:`True` if visualization with `matplotlib` is available, :obj:`False` otherwise.
    """

    return _matplotlib_imports._imports.is_successful()


def _is_log_scale(trials: List[FrozenTrial], param: str) -> bool:

    return any(
        isinstance(t.distributions[param], LogUniformDistribution)
        for t in trials
        if param in t.params
    )
