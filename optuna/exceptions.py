class OptunaError(Exception):
    """Base class for Optuna specific errors."""

    pass


class TrialPruned(OptunaError):
    """Exception for pruned trials.

    This error tells a trainer that the current :class:`~optuna.trial.Trial` was pruned. It is
    supposed to be raised after :func:`optuna.trial.Trial.should_prune` as shown in the following
    example.

    Example:

        .. code::

            >>> def objective(trial):
            >>>     ...
            >>>     for step in range(n_train_iter):
            >>>         ...
            >>>         if trial.should_prune():
            >>>             raise TrailPruned()
    """

    pass


class CLIUsageError(OptunaError):
    """Exception for CLI.

    CLI raises this exception when it receives invalid configuration.
    """

    pass


class StorageInternalError(OptunaError):
    """Exception for storage operation.

    This error is raised when an operation failed in backend DB of storage.
    """

    pass


class DuplicatedStudyError(OptunaError):
    """Exception for a duplicated study name.

    This error is raised when a specified study name already exists in the storage.
    """

    pass
