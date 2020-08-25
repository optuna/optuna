import abc

import numpy as np


class BaseHypervolume(object, metaclass=abc.ABCMeta):
    """Base class for hypervolume calculators.

        .. note::
            This class is used for computing the hypervolumes of points in multi-objective space.
            Each coordinate of each point represents one value of the multi-objective function.

        .. note::
            We check that each objective is to be minimized. Transform objective values that are
            to be maximized before calling this class's `compute` method. For example,

            .. testcode::

                import time

                import numpy as np
                from sklearn.datasets import load_iris
                from sklearn.linear_model import SGDClassifier
                from sklearn.model_selection import train_test_split

                import optuna

                X, y = load_iris(return_X_y=True)
                X_train, X_valid, y_train, y_valid = train_test_split(X, y)
                classes = np.unique(y)
                n_train_iter = 100

                def objective(trial):
                    start = time.time()

                    alpha = trial.suggest_uniform('alpha', 0.0, 1.0)
                    clf = SGDClassifier(alpha=alpha)

                    for step in range(n_train_iter):
                        clf.partial_fit(X_train, y_train, classes=classes)

                    accuracy = clf.score(X_valid, y_valid)

                    elapsed_time = time.time() - start

                    return accuracy, elapsed_time

                study = optuna.multi_objective.create_study(["maximize", "minimize"])
                study.optimize(objective, n_trials=100)
                trials = study.get_pareto_front_trials()
                solution_sets = np.ndarray([t.values for t in trials])
                # Transform the objective to be maximized into that for minimization.
                solution_sets = np.ndarray([[-s[0], s[1]] for s in solution_sets])
                reference_point = 2 * np.max(solution_set, axis=0)
                hypervolume = WFG().compute(solution_sets, reference_point)
                print("Hypervolume of the Pareto solutions is {}.".format(hypervolume))
    """

    def compute(self, solution_set: np.ndarray, reference_point: np.ndarray) -> float:
        """Compute the hypervolume for the given solution set and reference point.

        .. note::
            We assume that all points in the solution set dominate or equal the reference point.
            In other words, for all points in the solution set and the coordinate `i`,
            `point[i] <= reference_point[i]`.

        Args:
            solution_set:
                The solution set which we want to compute the hypervolume.
            reference_point:
                The reference point to compute the hypervolume.
        """

        self._validate(solution_set, reference_point)
        return self._compute(solution_set, reference_point)

    @staticmethod
    def _validate(solution_set: np.ndarray, reference_point: np.ndarray) -> None:
        # Validates that all points in the solution set dominate or equal the reference point.
        if not (solution_set <= reference_point).all():
            raise ValueError(
                "All points must dominate or equal the reference point. "
                "That is, for all points in the solution_set and the coordinate `i`, "
                "`point[i] <= reference_point[i]`."
            )

    @abc.abstractmethod
    def _compute(self, solution_set: np.ndarray, reference_point: np.ndarray) -> float:
        raise NotImplementedError
