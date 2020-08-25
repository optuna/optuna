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
                def objective(trial):
                    ...
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
