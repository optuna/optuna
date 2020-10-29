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

    .. seealso::
        Please refer to :func:`optuna.visualization.plot_edf` for an example,
        where this function can be replaced with it.

    Example:

        The following code snippet shows how to plot EDF.

        .. plot::

            import math

            import optuna


            def ackley(x, y):
                a = 20 * math.exp(-0.2 * math.sqrt(0.5 * (x ** 2 + y ** 2)))
                b = math.exp(0.5 * (math.cos(2 * math.pi * x) + math.cos(2 * math.pi * y)))
                return -a - b + math.e + 20


            def objective(trial, low, high):
                x = trial.suggest_float("x", low, high)
                y = trial.suggest_float("y", low, high)
                return ackley(x, y)


            sampler = optuna.samplers.RandomSampler(seed=10)

            # Widest search space.
            study0 = optuna.create_study(study_name="x=[0,5), y=[0,5)", sampler=sampler)
            study0.optimize(lambda t: objective(t, 0, 5), n_trials=500)

            # Narrower search space.
            study1 = optuna.create_study(study_name="x=[0,4), y=[0,4)", sampler=sampler)
            study1.optimize(lambda t: objective(t, 0, 4), n_trials=500)

            # Narrowest search space but it doesn't include the global optimum point.
            study2 = optuna.create_study(study_name="x=[1,3), y=[1,3)", sampler=sampler)
            study2.optimize(lambda t: objective(t, 1, 3), n_trials=500)

            optuna.visualization.matplotlib.plot_edf([study0, study1, study2])

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

    if any([s._n_ovbjectives > 1 for s in studies]):
        raise NotImplementedError("The EDF plot only supports the single-objective optimization.")

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
