import abc

import numpy as np


class BaseHypervolume(object, metaclass=abc.ABCMeta):
    """Base class for hypervolume calculators

        .. note::
            In Optuna, this class is used for computing the hypervolume of objective points.
            In other words, the coordinate of each point represents the one of the multi-objective
            function's values.

        .. note::
            We assume that the each objective function to be minimized. If you use some objectives
            to be maximized, please transform inputs to this class's `compute` method for the
            minimization problem. For example,

            .. code::
                def objective(trial):
                    ...
                    return accuracy, elapsed_time

                class YourSampler(BaseSampler):
                    ...
                    def sample_relative(study, trial, search_space):
                        ...
                        trials = study.get_trials()
                        solution_sets = np.ndarray([t.values for t in trials])

                        # Transform the objective to be maximized into that for minimization.
                        solution_sets = np.ndarray([[-s[0], s[1]] for s in solution_sets])

                        reference_point = 2 * np.max(solution_set, axis=0)
                        hypervolume = WFG().compute(solution_sets, reference_point)
                        ...
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
