class OptunaError(Exception):
    """Base class for Optuna specific errors."""

    pass


class TrialPruned(OptunaError):
    """Exception for pruned trials.

    This error tells a trainer that the current :class:`~optuna.trial.Trial` was pruned. It is
    supposed to be raised after :func:`optuna.trial.Trial.should_prune` as shown in the following
    example.

    Example:

        .. testsetup::

            import numpy as np
            from sklearn.model_selection import train_test_split

            np.random.seed(seed=0)
            X = np.random.randn(200).reshape(-1, 1)
            y = np.where(X[:, 0] < 0.5, 0, 1)
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
            classes = np.unique(y)

        .. testcode::

            import optuna
            from sklearn.linear_model import SGDClassifier

            def objective(trial):
                alpha = trial.suggest_uniform('alpha', 0.0, 1.0)
                clf = SGDClassifier(alpha=alpha)
                n_train_iter = 100

                for step in range(n_train_iter):
                    clf.partial_fit(X_train, y_train, classes=classes)

                    intermediate_value = clf.score(X_test, y_test)
                    trial.report(intermediate_value, step)

                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()

                return clf.score(X_test, y_test)

            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=20)
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


class ExperimentalWarning(Warning):
    """Experimental Warning class.

    This implementation exists here because the policy of `FutureWarning` has been changed
    since Python 3.7 was released. See the details in
    https://docs.python.org/3/library/warnings.html#warning-categories.
    """

    pass
