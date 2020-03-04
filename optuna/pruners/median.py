from optuna.pruners.percentile import PercentilePruner


class MedianPruner(PercentilePruner):
    """Pruner using the median stopping rule.

    Prune if the trial's best intermediate result is worse than median of intermediate results of
    previous trials at the same step.

    Example:

        We minimize an objective function with the median stopping rule.

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

            study = optuna.create_study(direction='maximize',
                                        pruner=optuna.pruners.MedianPruner(n_startup_trials=5,
                                                                           n_warmup_steps=30,
                                                                           interval_steps=10))
            study.optimize(objective, n_trials=20)

    Args:
        n_startup_trials:
            Pruning is disabled until the given number of trials finish in the same study.
        n_warmup_steps:
            Pruning is disabled until the trial reaches the given number of step.
        interval_steps:
            Interval in number of steps between the pruning checks, offset by the warmup steps.
            If no value has been reported at the time of a pruning check, that particular check
            will be postponed until a value is reported.
    """

    def __init__(self, n_startup_trials=5, n_warmup_steps=0, interval_steps=1):
        # type: (int, int, int) -> None

        super(MedianPruner, self).__init__(50.0, n_startup_trials, n_warmup_steps, interval_steps)
