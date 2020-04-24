from optuna.distributions import LogUniformDistribution
from optuna import type_checking
from optuna.visualization import plotly_imports

if type_checking.TYPE_CHECKING:
    from typing import List  # NOQA

    from optuna.trial import FrozenTrial  # NOQA


__all__ = ["is_available"]


def is_available():
    # type: () -> bool
    """Returns whether visualization is available or not.

    .. note::

        :mod:`~optuna.visualization` module depends on plotly version 4.0.0 or higher. If a
        supported version of plotly isn't installed in your environment, this function will return
        :obj:`False`. In such case, please execute ``$ pip install -U plotly>=4.0.0`` to install
        plotly.

    Returns:
        :obj:`True` if visualization is available, :obj:`False` otherwise.
    """

    return plotly_imports._available


def _check_plotly_availability():
    # type: () -> None

    if not is_available():
        raise ImportError(
            "Plotly is not available. Please install plotly to use this feature. "
            "Plotly can be installed by executing `$ pip install plotly`. "
            "For further information, please refer to the installation guide of plotly. "
            "(The actual import error is as follows: " + str(plotly_imports._import_error) + ")"
        )

    from distutils.version import StrictVersion
    from plotly import __version__ as plotly_version

    if StrictVersion(plotly_version) < StrictVersion("4.0.0"):
        raise ImportError(
            "Your version of Plotly is " + plotly_version + " . "
            "Please install plotly version 4.0.0 or higher. "
            "Plotly can be installed by executing `$ pip install -U plotly>=4.0.0`. "
            "For further information, please refer to the installation guide of plotly. "
        )


def _is_log_scale(trials, param):
    # type: (List[FrozenTrial], str) -> bool

    return any(
        isinstance(t.distributions[param], LogUniformDistribution)
        for t in trials
        if param in t.params
    )
