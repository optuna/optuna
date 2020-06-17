from typing import List

from optuna.distributions import LogUniformDistribution
from optuna.trial import FrozenTrial
from optuna.visualization import _plotly_imports

__all__ = ["is_available"]


def is_available() -> bool:
    """Returns whether visualization is available or not.

    .. note::

        :mod:`~optuna.visualization` module depends on plotly version 4.0.0 or higher. If a
        supported version of plotly isn't installed in your environment, this function will return
        :obj:`False`. In such case, please execute ``$ pip install -U plotly>=4.0.0`` to install
        plotly.

    Returns:
        :obj:`True` if visualization is available, :obj:`False` otherwise.
    """

    return _plotly_imports._imports.is_successful()


def _is_log_scale(trials: List[FrozenTrial], param: str) -> bool:

    return any(
        isinstance(t.distributions[param], LogUniformDistribution)
        for t in trials
        if param in t.params
    )
