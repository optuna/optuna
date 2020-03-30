from optuna.pruners import BasePruner
from optuna import type_checking

if type_checking.TYPE_CHECKING:
    from optuna import structs  # NOQA
    from optuna.study import Study  # NOQA


class NopPruner(BasePruner):
    """Pruner which never prunes trials.

    Example:

        .. testcode::

            import optuna
            import numpy as np
            from sklearn.datasets import load_iris
            from sklearn.linear_model import SGDClassifier
            from sklearn.model_selection import train_test_split

            X, y = load_iris(return_X_y=True)
            X_train, X_test, y_train, y_test = train_test_split(X, y)
            classes = np.unique(y)

            def objective(trial):
                alpha = trial.suggest_uniform('alpha', 0.0, 1.0)
                clf = SGDClassifier(alpha=alpha)
                n_train_iter = 100

                for step in range(n_train_iter):
                    clf.partial_fit(X_train, y_train, classes=classes)

                    intermediate_value = clf.score(X_test, y_test)
                    trial.report(intermediate_value, step)

                    if trial.should_prune():
                        assert False, "should_prune() should always return False with this pruner."
                        raise optuna.exceptions.TrialPruned()

                return clf.score(X_test, y_test)

            study = optuna.create_study(direction='maximize',
                                        pruner=optuna.pruners.NopPruner())
            study.optimize(objective, n_trials=20)
    """

    def prune(self, study, trial):
        # type: (Study, structs.FrozenTrial) -> bool

        return False
