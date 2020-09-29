import itertools
from typing import List
from typing import Sequence
from typing import Union

import numpy as np

from optuna._experimental import experimental
from optuna.logging import get_logger
from optuna.study import Study
from optuna.trial import TrialState
from optuna.visualization.matplotlib._matplotlib_imports import _imports


if _imports.is_successful():
    from optuna.visualization.matplotlib._matplotlib_imports import Axes
    from optuna.visualization.matplotlib._matplotlib_imports import plt

_logger = get_logger(__name__)


@experimental("2.2.0")
def plot_edf(study: Union[Study, Sequence[Study]]) -> "Axes":
    """Plot the objective value EDF (empirical distribution function) of a study with Matplotlib.

    .. seealso::  optuna.visualization.plot_edf

    Args:
        study:
            A target :class:`~optuna.study.Study` object.
            You can pass multiple studies if you want to compare those EDFs.

    Returns:
        A :class:`matplotlib.axes.Axes` object.
    """

    _imports.check()

    if isinstance(study, Study):
        studies = [study]
    else:
        studies = list(study)

    return _get_edf_plot(studies)


def _get_edf_plot(studies: List[Study]) -> "Axes":

    # Set up the graph style.
    plt.style.use("ggplot")  # Use ggplot style sheet for similar outputs to plotly.
    _, ax = plt.subplots()
    ax.set_title("Empirical Distribution Function Plot")
    ax.set_xlabel("Objective Value")
    ax.set_ylabel("Cumulative Probability")
    ax.set_ylim(0, 1)
    cmap = plt.get_cmap("tab20")  # Use tab20 colormap for multiple line plots.

    # Prepare data for plotting.
    if len(studies) == 0:
        _logger.warning("There are no studies.")
        return ax

    all_trials = list(
        itertools.chain.from_iterable(
            (
                trial
                for trial in study.get_trials(deepcopy=False)
                if trial.state == TrialState.COMPLETE
            )
            for study in studies
        )
    )

    if len(all_trials) == 0:
        _logger.warning("There are no complete trials.")
        return ax

    min_x_value = min(trial.value for trial in all_trials)
    max_x_value = max(trial.value for trial in all_trials)
    x_values = np.linspace(min_x_value, max_x_value, 100)

    # Draw multiple line plots.
    for i, study in enumerate(studies):
        values = np.asarray(
            [
                trial.value
                for trial in study.get_trials(deepcopy=False)
                if trial.state == TrialState.COMPLETE
            ]
        )

        y_values = np.sum(values[:, np.newaxis] <= x_values, axis=0) / values.size

        ax.plot(x_values, y_values, color=cmap(i), alpha=0.7, label=study.study_name)

    return ax
