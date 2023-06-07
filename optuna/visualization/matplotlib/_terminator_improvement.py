from __future__ import annotations

from optuna._experimental import experimental_func
from optuna.logging import get_logger
from optuna.study.study import Study
from optuna.terminator import BaseErrorEvaluator
from optuna.terminator import BaseImprovementEvaluator
from optuna.terminator.improvement.evaluator import DEFAULT_MIN_N_TRIALS
from optuna.visualization._terminator_improvement import _get_improvement_info
from optuna.visualization._terminator_improvement import _get_y_range
from optuna.visualization._terminator_improvement import _ImprovementInfo
from optuna.visualization.matplotlib._matplotlib_imports import _imports


if _imports.is_successful():
    from optuna.visualization.matplotlib._matplotlib_imports import Axes
    from optuna.visualization.matplotlib._matplotlib_imports import plt

_logger = get_logger(__name__)


PADDING_RATIO_Y = 0.05
ALPHA = 0.25


@experimental_func("3.2.0")
def plot_terminator_improvement(
    study: Study,
    plot_error: bool = False,
    improvement_evaluator: BaseImprovementEvaluator | None = None,
    error_evaluator: BaseErrorEvaluator | None = None,
    min_n_trials: int = DEFAULT_MIN_N_TRIALS,
) -> "Axes":
    """Plot the potentials for future objective improvement.

    This function visualizes the objective improvement potentials, evaluated
    with ``improvement_evaluator``.
    It helps to determine whether we should continue the optimization or not.
    You can also plot the error evaluated with
    ``error_evaluator`` if the ``plot_error`` argument is set to :obj:`True`.
    Note that this function may take some time to compute
    the improvement potentials.

    .. seealso::
        Please refer to :func:`optuna.visualization.plot_terminator_improvement`.

    Example:
        The following code snippet shows how to plot improvement potentials,
        together with cross-validation errors.

        .. plot::

            from lightgbm import LGBMClassifier
            from sklearn.datasets import load_wine
            from sklearn.model_selection import cross_val_score
            from sklearn.model_selection import KFold
            import optuna
            from optuna.terminator import report_cross_validation_scores
            from optuna.visualization.matplotlib import plot_terminator_improvement

            def objective(trial):
                X, y = load_wine(return_X_y=True)
                clf = LGBMClassifier(
                    reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                    reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                    num_leaves=trial.suggest_int("num_leaves", 2, 256),
                    colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
                    subsample=trial.suggest_float("subsample", 0.4, 1.0),
                    subsample_freq=trial.suggest_int("subsample_freq", 1, 7),
                    min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
                )
                scores = cross_val_score(clf, X, y, cv=KFold(n_splits=5, shuffle=True))
                report_cross_validation_scores(trial, scores)
                return scores.mean()

            study = optuna.create_study()
            study.optimize(objective, n_trials=30)

            plot_terminator_improvement(study, plot_error=True)

    Args:
        study:
            A :class:`~optuna.study.Study` object whose trials are plotted for their improvement.
        plot_error:
            A flag to show the error. If it is set to :obj:`True`, errors
            evaluated by ``error_evaluator`` are also plotted as line graph.
            Defaults to :obj:`False`.
        improvement_evaluator:
            An object that evaluates the improvement of the objective function.
            Default to :class:`~optuna.terminator.RegretBoundEvaluator`.
        error_evaluator:
            An object that evaluates the error inherent in the objective function.
            Default to :class:`~optuna.terminator.CrossValidationErrorEvaluator`.
        min_n_trials:
            The minimum number of trials before termination is considered.
            Terminator improvements for trials below this value are
            shown in a lighter color. Defaults to ``20``.

    Returns:
        A :class:`matplotlib.axes.Axes` object.
    """
    _imports.check()

    info = _get_improvement_info(study, plot_error, improvement_evaluator, error_evaluator)
    return _get_improvement_plot(info, min_n_trials)


def _get_improvement_plot(info: _ImprovementInfo, min_n_trials: int) -> "Axes":
    n_trials = len(info.trial_numbers)

    # Set up the graph style.
    plt.style.use("ggplot")  # Use ggplot style sheet for similar outputs to plotly.
    _, ax = plt.subplots()
    ax.set_title("Terminator Improvement Plot")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Terminator Improvement")
    cmap = plt.get_cmap("tab10")  # Use tab10 colormap for similar outputs to plotly.

    if n_trials == 0:
        _logger.warning("There are no complete trials.")
        return ax

    ax.plot(
        info.trial_numbers[: min_n_trials + 1],
        info.improvements[: min_n_trials + 1],
        marker="o",
        color=cmap(0),
        alpha=ALPHA,
        label="Terminator Improvement" if n_trials <= min_n_trials else None,
    )

    if n_trials > min_n_trials:
        ax.plot(
            info.trial_numbers[min_n_trials:],
            info.improvements[min_n_trials:],
            marker="o",
            color=cmap(0),
            label="Terminator Improvement",
        )

    if info.errors is not None:
        ax.plot(
            info.trial_numbers,
            info.errors,
            marker="o",
            color=cmap(3),
            label="Error",
        )
    ax.legend()
    ax.set_ylim(_get_y_range(info, min_n_trials))
    return ax
