import itertools
from typing import List

from matplotlib.figure import Figure
from matplotlib import pyplot as plt
import numpy as np

from optuna.logging import get_logger
from optuna.study import Study
from optuna.trial import TrialState

_logger = get_logger(__name__)


def _get_edf_plot(studies: List[Study]) -> Figure:
    """Plot the objective value EDF (empirical distribution function) of studies with matplotlib.

    Note that only the complete trials are considered when plotting the EDF.

    .. note::

        EDF is useful to analyze and improve search spaces.
        For instance, you can see a practical use case of EDF in the paper
        `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>`_.

    .. note::

        The plotted EDF assumes that the value of the objective function is in
        accordance with the uniform distribution over the objective space.

    Args:
        studies:
            A list of target :class:`~optuna.study.Study` objects.
            You can pass multiple studies if you want to compare those EDFs.

    Returns:
        A :class:`matplotlib.figure.Figure` object.
    """

    # Set up the graph style.
    plt.style.use("ggplot")  # Use ggplot style sheet for similar outputs to plotly.
    fig, ax = plt.subplots()
    ax.set_title("Empirical Distribution Function Plot")
    ax.set_xlabel("Objective Value")
    ax.set_ylabel("Cumulative Probability")
    ax.set_ylim(0, 1)
    cmap = plt.get_cmap("tab20")  # Use tab20 colormap for multiple line plots.

    # Prepare data for plotting.
    if len(studies) == 0:
        _logger.warning("There are no studies.")
        return fig

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
        return fig

    min_x_value = min(trial.value for trial in all_trials)
    max_x_value = max(trial.value for trial in all_trials)
    x_values = np.linspace(min_x_value, max_x_value, 100)

    # Draw multiple line plots.
    traces = []
    for i, study in enumerate(studies):
        values = np.asarray(
            [
                trial.value
                for trial in study.get_trials(deepcopy=False)
                if trial.state == TrialState.COMPLETE
            ]
        )

        y_values = np.sum(values[:, np.newaxis] <= x_values, axis=0) / values.size

        trace = ax.plot(x_values, y_values, color=cmap(i), alpha=0.7, label=study.study_name)
        traces.append(trace)

    ax.legend()

    return fig
