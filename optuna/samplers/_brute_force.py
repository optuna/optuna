from __future__ import annotations

from dataclasses import dataclass
import decimal
import math
from numbers import Real
import sys
from typing import Any
from typing import TYPE_CHECKING

import numpy as np

from optuna._experimental import experimental_class
from optuna.distributions import CategoricalChoiceType
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


# TODO(nabenabe): Simply use `slots=True` once Python 3.9 is dropped.
@dataclass(**({"slots": True} if sys.version_info >= (3, 10) else {}))
class _TreeNode:
    # A tree representing the search space for brute force sampling.
    # Each internal node corresponds to a parameter, and its children are keyed by the parameter's
    # candidate values (in internal representation). A path from the root to a terminal node
    # represents a complete ``params``.
    #
    # A node takes one of the following four states:
    #   1. Unexpanded (no trial tried the ``params``). ``children=None`` and ``is_running=False``.
    #   2. Running. ``children=None`` and ``is_running=True``.
    #   3. Leaf. ``children={}`` and ``param_name=None``.
    #   4. Internal. ``param_name`` is set and ``children`` is non-empty.
    # Leaf represents the last parameter of a finished ``params``, and internal means the node
    # does not represent a complete ``params``.
    # NOTE(nabenabe): I tried representations by list and dict, but they did not really speed up.

    param_name: str | None = None
    children: dict[float, "_TreeNode"] | None = None
    is_running: bool = False
    choices_fingerprint: tuple[int, float, float] | None = None

    def expand(self, param_name: str | None, choices: list[float]) -> None:
        # If the node is unexpanded, expand it.
        # Otherwise, check if the node is compatible with the given search space.
        choices_fingerprint = (len(choices), choices[0], choices[-1]) if choices else (0, 0, 0)
        if self.children is None:
            # Expand the node
            self.param_name = param_name
            self.choices_fingerprint = choices_fingerprint
            # TODO(nabenabe): We can lazily instantiate _TreeNode, and this is cleaner.
            self.children = {value: _TreeNode() for value in choices}
        else:
            if self.param_name != param_name:
                raise ValueError(f"param_name mismatch: {self.param_name} != {param_name}")
            if choices_fingerprint != self.choices_fingerprint:
                # NOTE(nabenabe): search space and children are sorted, and each distribution has
                # a uniform interval (FloatDistribution raises error for log=True and finite step),
                # so the first and last elements and length check are equivalent to
                # ``children.keys() != set(search_space)``.
                raise ValueError(
                    f"search_space mismatch: {set(self.children.keys())} != {set(choices)}"
                )

    def set_running(self) -> None:
        self.is_running = True

    def set_leaf(self) -> None:
        self.expand(None, [])

    def add_path(self, trial_path: Iterable[tuple[str, list[float], float]]) -> _TreeNode | None:
        # Add a path (i.e. a list of suggested parameters in one trial) to the tree.
        current_node = self
        for param_name, choices, value in trial_path:
            current_node.expand(param_name, choices)
            # TODO(nabenabe): This is a temporal fix until the lazy node is introduced.
            next_node = (current_node.children or {}).get(value)
            if next_node is None:
                return None
            current_node = next_node
        return current_node

    def is_any_expandable(self, exclude_running: bool) -> bool:
        if (children := self.children) is None:
            return not exclude_running or not self.is_running
        return any(child.is_any_expandable(exclude_running) for child in children.values())

    def count_unexpanded(self, exclude_running: bool) -> int:
        # Count the number of unexpanded nodes in the subtree.
        if (children := self.children) is None:
            return 0 if exclude_running and self.is_running else 1
        return sum(child.count_unexpanded(exclude_running) for child in children.values())

    def sample_child(self, rng: np.random.RandomState, exclude_running: bool) -> float:
        assert (children := self.children) is not None
        unexpanded_counts = np.array(
            [child.count_unexpanded(exclude_running) for child in children.values()], dtype=float
        )

        # Blend exact uniform sampling with flat uniform sampling
        # to prevent starvation of unexplored branches
        alpha = 0.5
        weights_orig = unexpanded_counts / unexpanded_counts.sum()
        weights_flat = np.where(unexpanded_counts > 0, 1.0, 0.0)
        weights_flat /= weights_flat.sum()

        weights = (1.0 - alpha) * weights_orig + alpha * weights_flat
        if any(
            not value.is_running and weights[i] > 0 for i, value in enumerate(children.values())
        ):
            # Prioritize picking non-running and unexpanded nodes.
            for i, child in enumerate(children.values()):
                if child.is_running:
                    weights[i] = 0.0
        weights /= weights.sum()
        return rng.choice(list(children.keys()), p=weights).item()


def _get_non_waiting_trials_and_current_trial_index(
    study: Study, current_trial_number: int
) -> tuple[list[FrozenTrial], int]:
    # We directly query the storage to get trials here instead of `study.get_trials`,
    # since some pruners such as `HyperbandPruner` use the study transformed
    # to filter trials. See https://github.com/optuna/optuna/issues/2327 for details.
    states = (TrialState.COMPLETE, TrialState.PRUNED, TrialState.RUNNING, TrialState.FAIL)
    trials = study._storage.get_all_trials(study._study_id, deepcopy=False, states=states)
    # `trials` is fetched by shallow copy, so pop() or element replacement are safe operations.
    for i in range(1, len(trials) + 1):
        # The current trial can be found at the later part for almost all cases.
        t = trials[-i]
        if t.number == current_trial_number:
            return trials, len(trials) - i
    assert False, "Should not reach"


@experimental_class("3.1.0")
class BruteForceSampler(BaseSampler):
    """Sampler that performs exhaustive search over the define-by-run search space.

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
        params_items = params.items()
        nonnan_params_items = {k: v for k, v in params_items if not _is_nan(v)}.items()
        nan_param_names = [k for k, v in params_items if _is_nan(v)]
        for trial in trials:
            trial_params = trial.params
            if params:
                if not (nonnan_params_items <= trial_params.items()):
                    continue
                if not all(_is_nan(trial_params.get(p)) for p in nan_param_names):
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
        if not tree.is_any_expandable(exclude_running):
            return param_distribution.to_external_repr(self._rng.rng.choice(candidates).item())
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
        trials[current_idx] = create_trial(
            state=state, values=values, params=trial.params, distributions=trial.distributions
        )
        tree = _TreeNode()
        self._populate_tree(tree, trials, {})
        if not tree.is_any_expandable(exclude_running):
            study.stop()


def _is_nan(v: CategoricalChoiceType) -> bool:
    return isinstance(v, Real) and math.isnan(float(v))


def _enumerate_candidates(param_distribution: BaseDistribution) -> list[float]:
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
