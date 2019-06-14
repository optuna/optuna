from optuna.pruners.percentile import PercentilePruner


class MedianPruner(PercentilePruner):
    """Pruner using the median stopping rule.

    Prune if the trial's best intermediate result is worse than median of intermediate results of
    previous trials at the same step.

    Example:

        We minimize an objective function with the median stopping rule.

        .. code::

            >>> from optuna import create_study
            >>> from optuna.pruners import MedianPruner
            >>>
            >>> def objective(trial):
            >>>     ...
            >>>
            >>> study = create_study(pruner=MedianPruner())
            >>> study.optimize(objective)

    Args:
        n_startup_trials:
            Pruning is disabled until the given number of trials finish in the same study.
        n_warmup_steps:
            Pruning is disabled until the trial reaches the given number of step.
    """

    def __init__(self, n_startup_trials=5, n_warmup_steps=0):
        # type: (int, int) -> None

        super(MedianPruner, self).__init__(50.0, n_startup_trials, n_warmup_steps)
