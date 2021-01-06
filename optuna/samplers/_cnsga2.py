import copy
from typing import Callable
from typing import Optional
from typing import Sequence
import warnings

from optuna._multi_objective import _dominates
from optuna.samplers._nsga2 import NSGAIISampler
from optuna.study import Study
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


_CONSTRAINTS_KEY = "cnsga2:constraints"


class CNSGAIISampler(NSGAIISampler):
    """Multi-objective sampler using the constrained NSGA-II algorithm.

    This sampler extends :class:`~optuna.samplers.NSGAIISampler` to handle constraints
    based on the constrained-domination principle. In short, any feasible trial has a
    better nondomination rank than any infeasible trial.

    For further details of constrained NSGA-II, please refer to the following paper:

    - `A fast and elitist multiobjective genetic algorithm: NSGA-II
      <https://ieeexplore.ieee.org/document/996017>`_

    Args:
        population_size:
            Number of individuals (trials) in a generation.

        mutation_prob:
            Probability of mutating each parameter when creating a new individual.
            If :obj:`None` is specified, the value ``1.0 / len(parent_trial.params)`` is used
            where ``parent_trial`` is the parent trial of the target individual.

        crossover_prob:
            Probability that a crossover (parameters swapping between parents) will occur
            when creating a new individual.

        swapping_prob:
            Probability of swapping each parameter of the parents during crossover.

        seed:
            Seed for random number generator.

        constraints_func:
            A function that computes the objective constraints. It must take a
            :class:`~optuna.trial.FrozenTrial` and return the constraints. The return value must
            be a sequence of :obj:`float` s. A value strictly larger than 0 means that a
            constraints is violated. A value equal to or smaller than 0 is considered feasible.
    """

    def __init__(
        self,
        *,
        population_size: int = 50,
        mutation_prob: Optional[float] = None,
        crossover_prob: float = 0.9,
        swapping_prob: float = 0.5,
        seed: Optional[int] = None,
        constraints_func: Callable[[FrozenTrial], Sequence[float]],
    ) -> None:
        super().__init__(
            population_size=population_size,
            mutation_prob=mutation_prob,
            crossover_prob=crossover_prob,
            swapping_prob=swapping_prob,
            seed=seed,
        )
        self._constraints_func = constraints_func

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Optional[Sequence[float]],
    ) -> None:
        if self._constraints_func is not None:
            constraints = None
            _trial = copy.copy(trial)
            _trial.state = state
            _trial.values = values
            try:
                con = self._constraints_func(_trial)
                if not isinstance(con, (tuple, list)):
                    warnings.warn(
                        f"Constraints should be a sequence of floats but got {constraints}."
                    )
                constraints = tuple(con)
            except Exception:
                raise
            finally:
                assert constraints is None or isinstance(constraints, tuple)

                study._storage.set_trial_system_attr(
                    trial._trial_id,
                    _CONSTRAINTS_KEY,
                    constraints,
                )

    @classmethod
    def _dominates(
        cls, trial0: FrozenTrial, trial1: FrozenTrial, directions: Sequence[StudyDirection]
    ) -> bool:
        """Checks constrained-domination.

        A trial x is said to constrained-dominate a trial y, if any of the following conditions is
        true:
        1) Trial x is feasible and trial y is not.
        2) Trial x and y are both infeasible, but solution x has a smaller overall constraint
        violation.
        3) Trial x and y are feasible and trial x dominates trial y.
        """

        constraints0 = trial0.system_attrs[_CONSTRAINTS_KEY]
        constraints1 = trial1.system_attrs[_CONSTRAINTS_KEY]

        assert isinstance(constraints0, (list, tuple))
        assert isinstance(constraints1, (list, tuple))

        if len(constraints1) != len(constraints1):
            raise ValueError("Trials with different numbers of constraints cannot be compared.")

        if trial0.state != TrialState.COMPLETE:
            return False

        if trial1.state != TrialState.COMPLETE:
            return True

        if all(v <= 0 for v in constraints0) and all(v <= 0 for v in constraints1):
            # Both trials satisfy the constraints.
            return _dominates(trial0, trial1, directions)

        if all(v <= 0 for v in constraints0):
            # trial0 satisfies the constraints, but trial1 violates them.
            return True

        if all(v <= 0 for v in constraints1):
            # trial1 satisfies the constraints, but trial0 violates them.
            return False

        # Both trials violate the constraints.
        violation0 = sum(v for v in constraints0 if v > 0)
        violation1 = sum(v for v in constraints1 if v > 0)
        return violation0 < violation1
