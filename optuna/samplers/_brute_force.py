from __future__ import annotations

from dataclasses import dataclass
import decimal
import sys
from typing import Any
from typing import TYPE_CHECKING

import numpy as np

from optuna._experimental import experimental_class
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.samplers import BaseSampler
from optuna.samplers._lazy_random_state import LazyRandomState
from optuna.trial import create_trial
from optuna.trial import TrialState


if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence

    from optuna.distributions import BaseDistribution
    from optuna.study import Study
    from optuna.trial import FrozenTrial


_NaN = float("nan")  # NOTE: Define once to save runtime overhead of the variable definition.


# TODO(nabenabe): Simply use `slots=True` once Python 3.9 is dropped.
@dataclass(**({"slots": True} if sys.version_info >= (3, 10) else {}))
class _TreeNode:
    """
    This is a class to represent the tree of search space.

    A tree node has three states:
        1. Unexpanded. This is represented by children=None.
        2. Leaf. This is represented by children={} and param_name=None.
        3. Normal node. It has a param_name and non-empty children.

    NOTE(nabenabe): I tried representations by list and dict, but they did not really speed up.
    """

    param_name: str | None = None
    children: dict[float, "_TreeNode"] | None = None
    is_running: bool = False

    def expand(self, param_name: str | None, choices: Iterable[float]) -> dict[float, "_TreeNode"]:
        # If the node is unexpanded, expand it.
        # Otherwise, check if the node is compatible with the given search space.
        if self.children is None:
            # Expand the node
            self.param_name = param_name
            self.children = {value: _TreeNode() for value in choices}
            return self.children
        else:
            if self.param_name != param_name:
                raise ValueError(f"param_name mismatch: {self.param_name} != {param_name}")
            if self.children.keys() != set(choices):
                raise ValueError(
                    f"search_space mismatch: {set(self.children.keys())} != {set(choices)}"
                )
            return self.children

    def set_running(self) -> None:
        self.is_running = True

    def set_leaf(self) -> None:
        self.expand(None, [])

    def add_path(
        self, params_and_choices: Iterable[tuple[str, Iterable[float], float]]
    ) -> _TreeNode | None:
        # Add a path (i.e. a list of suggested parameters in one trial) to the tree.
        current_node = self
        for param_name, choices, value in params_and_choices:
            next_node = current_node.expand(param_name, choices).get(value)
            if next_node is None:
                return None
            current_node = next_node
        return current_node

    def count_unexpanded(self, exclude_running: bool) -> int:
        # Count the number of unexpanded nodes in the subtree.
        if self.children is None:
            return 0 if exclude_running and self.is_running else 1
        return sum(child.count_unexpanded(exclude_running) for child in self.children.values())

    def sample_child(self, rng: np.random.RandomState, exclude_running: bool) -> float:
        assert self.children is not None

        unexpanded_counts = np.array(
            [child.count_unexpanded(exclude_running) for child in self.children.values()],
            dtype=np.float64,
        )

        # Blend exact uniform sampling with flat uniform sampling
        # to prevent starvation of unexplored branches
        alpha = 0.5
        weights_orig = unexpanded_counts / unexpanded_counts.sum()
        weights_flat = np.where(unexpanded_counts > 0, 1.0, 0.0)
        weights_flat /= weights_flat.sum()

        weights = (1.0 - alpha) * weights_orig + alpha * weights_flat
        if any(
            not value.is_running and weights[i] > 0
            for i, value in enumerate(self.children.values())
        ):
            # Prioritize picking non-running and unexpanded nodes.
            for i, child in enumerate(self.children.values()):
                if child.is_running:
                    weights[i] = 0.0
        weights /= weights.sum()
        return rng.choice(list(self.children.keys()), p=weights)


def _get_non_waiting_trials_and_current_trial_index(
    study: Study, current_trial_number: int
) -> tuple[list[FrozenTrial], int]:
    # We directly query the storage to get trials here instead of `study.get_trials`,
    # since some pruners such as `HyperbandPruner` use the study transformed
    # to filter trials. See https://github.com/optuna/optuna/issues/2327 for details.
    states = (TrialState.COMPLETE, TrialState.PRUNED, TrialState.RUNNING, TrialState.FAIL)
    trials = study._storage.get_all_trials(study._study_id, deepcopy=False, states=states)
    for i in range(1, len(trials) + 1):
        # The current trial can be found at the later part for almost all cases.
        t = trials[-i]
        if t.number == current_trial_number:
            return trials, len(trials) - i
    assert False, "Should not reach"


@experimental_class("3.1.0")
class BruteForceSampler(BaseSampler):
    """Sampler using brute force.

    This sampler performs exhaustive search on the defined search space.

    Example:

        .. testcode::

            import optuna


            def objective(trial):
                c = trial.suggest_categorical("c", ["float", "int"])
                if c == "float":
                    return trial.suggest_float("x", 1, 3, step=0.5)
                elif c == "int":
                    a = trial.suggest_int("a", 1, 3)
                    b = trial.suggest_int("b", a, 3)
                    return a + b


            study = optuna.create_study(sampler=optuna.samplers.BruteForceSampler())
            study.optimize(objective)

    Note:
        The defined search space must be finite. Therefore, when using
        :class:`~optuna.distributions.FloatDistribution` or
        :func:`~optuna.trial.Trial.suggest_float`, ``step=None`` is not allowed.

    Note:
        The sampler may fail to try the entire search space in when the suggestion ranges or
        parameters are changed in the same :class:`~optuna.study.Study`.

    Args:
        seed:
            A seed to fix the order of trials as the search order randomly shuffled. Please note
            that it is not recommended using this option in distributed optimization settings since
            this option cannot ensure the order of trials and may increase the number of duplicate
            suggestions during distributed optimization.
        avoid_premature_stop:
            If :obj:`True`, the sampler performs a strict exhaustive search. Please note
            that enabling this option may increase the likelihood of duplicate sampling.
            When this option is not enabled (default), the sampler applies a looser criterion for
            determining when to stop the search, which may result in incomplete coverage of the
            search space. For more information, see https://github.com/optuna/optuna/issues/5780.
    """

    def __init__(self, seed: int | None = None, avoid_premature_stop: bool = False) -> None:
        self._rng = LazyRandomState(seed)
        self._avoid_premature_stop = avoid_premature_stop

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> dict[str, BaseDistribution]:
        return {}

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        return {}

    @staticmethod
    def _populate_tree(tree: _TreeNode, trials: list[FrozenTrial], params: dict[str, Any]) -> None:
        # Populate tree under given params from the given trials.
        for trial in trials:
            # NOTE(nabenabe): `nan` cannot be assigned as a param value.
            if params and not all(trial.params.get(p, _NaN) == v for p, v in params.items()):
                continue
            leaf = tree.add_path(
                (
                    (
                        param_name,
                        _enumerate_candidates(param_distribution),
                        param_distribution.to_internal_repr(trial.params[param_name]),
                    )
                    for param_name, param_distribution in trial.distributions.items()
                    if param_name not in params
                )
            )
            if leaf is not None:
                # The parameters are on the defined grid.
                if trial.state.is_finished():
                    leaf.set_leaf()
                else:
                    leaf.set_running()

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        exclude_running = not self._avoid_premature_stop
        trials, current_idx = _get_non_waiting_trials_and_current_trial_index(study, trial.number)
        trials.pop(current_idx)
        tree = _TreeNode()
        candidates = _enumerate_candidates(param_distribution)
        tree.expand(param_name, candidates)
        # Populating must happen after the initialization above to prevent `tree` from
        # being initialized as an empty graph, which is created with n_jobs > 1
        # where we get trials[i].params = {} for some i.
        self._populate_tree(tree, trials, trial.params)
        if tree.count_unexpanded(exclude_running) == 0:
            return param_distribution.to_external_repr(self._rng.rng.choice(candidates))
        else:
            return param_distribution.to_external_repr(
                tree.sample_child(self._rng.rng, exclude_running)
            )

    def after_trial(
        self, study: Study, trial: FrozenTrial, state: TrialState, values: Sequence[float] | None
    ) -> None:
        exclude_running = not self._avoid_premature_stop
        trials, current_idx = _get_non_waiting_trials_and_current_trial_index(study, trial.number)
        # Set current trial as complete.
        current_trial = create_trial(
            state=state, values=values, params=trial.params, distributions=trial.distributions
        )
        trials[current_idx] = current_trial
        tree = _TreeNode()
        self._populate_tree(tree, trials, {})
        if tree.count_unexpanded(exclude_running) == 0:
            study.stop()


def _enumerate_candidates(param_distribution: BaseDistribution) -> Sequence[float]:
    if isinstance(param_distribution, FloatDistribution):
        if param_distribution.step is None:
            raise ValueError(
                "FloatDistribution.step must be given for BruteForceSampler"
                " (otherwise, the search space will be infinite)."
            )
        low = decimal.Decimal(str(param_distribution.low))
        high = decimal.Decimal(str(param_distribution.high))
        step = decimal.Decimal(str(param_distribution.step))

        ret = []
        value = low
        while value <= high:
            ret.append(float(value))
            value += step

        return ret
    elif isinstance(param_distribution, IntDistribution):
        return list(
            range(param_distribution.low, param_distribution.high + 1, param_distribution.step)
        )
    elif isinstance(param_distribution, CategoricalDistribution):
        return list(range(len(param_distribution.choices)))  # Internal representations.
    else:
        raise ValueError(f"Unknown distribution {param_distribution}.")
