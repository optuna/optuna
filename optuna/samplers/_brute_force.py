from __future__ import annotations

from dataclasses import dataclass
import decimal
from functools import lru_cache
import math
from numbers import Real
import sys
from typing import Any
from typing import cast
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
    from collections.abc import Sequence

    from optuna.distributions import BaseDistribution
    from optuna.study import Study
    from optuna.trial import FrozenTrial

    ChoicesArgsType = tuple[int | float, int | float, int | float | None]  # low, high, step


@dataclass(frozen=True)
class _UnexpandedTreeNode:
    is_running: bool = False

    def is_any_expandable(self, exclude_running: bool) -> bool:
        return True

    def count_unexpanded(self, exclude_running: bool) -> int:
        return 1


_UNEXPANDED_NODE = _UnexpandedTreeNode()


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
    children: dict[float, _TreeNode | _UnexpandedTreeNode] | None = None
    is_running: bool = False
    choices_args: ChoicesArgsType | None = None

    def _validate_search_space_consistency(
        self, param_name: str | None, choices_args: ChoicesArgsType | None
    ) -> None:
        if self.param_name != param_name:
            raise ValueError(f"param_name mismatch: {self.param_name} != {param_name}")
        if choices_args != self.choices_args:
            assert self.children is not None and choices_args is not None
            choices_old = list(self.children)
            choices_new = _enumerate_candidates(*choices_args)
            raise ValueError(
                f"search_space mismatch in {param_name}: {choices_old} != {choices_new}"
            )

    def expand(self, param_name: str | None, choices_args: ChoicesArgsType) -> None:
        # If the node is unexpanded, expand it.
        # Otherwise, check if the node is compatible with the given search space.
        if self.children is None:
            # Expand the node
            self.param_name = param_name
            choices = _enumerate_candidates(*choices_args)
            self.children = {value: _UNEXPANDED_NODE for value in choices}
            self.choices_args = choices_args
        else:
            self._validate_search_space_consistency(param_name, choices_args)

    def set_running(self) -> None:
        self.is_running = True

    def set_leaf(self) -> None:
        if self.children is not None:
            self._validate_search_space_consistency(None, None)
        self.children = {}

    def add_path(self, trial_path: list[tuple[str, ChoicesArgsType, float]]) -> _TreeNode | None:
        # Add a path (i.e. a list of suggested parameters in one trial) to the tree.
        current_node = self
        for param_name, choices_args, value in trial_path:
            current_node.expand(param_name, choices_args)
            if not (children := current_node.children):  # children is empty or None.
                return None
            elif (next_node := children.get(value)) is None:
                return None
            elif next_node is _UNEXPANDED_NODE:
                next_node = _TreeNode()
                children[value] = next_node
            current_node = cast(_TreeNode, next_node)
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
        cat_internal_repr_cache: dict[str, dict[CategoricalChoiceType, float]] = {}
        params_items = params.items()
        nonnan_params_items = {k: v for k, v in params_items if not _is_nan(v)}.items()
        nan_param_names = [k for k, v in params_items if _is_nan(v)]

        def _get_trial_path(trial: FrozenTrial) -> list[tuple[str, ChoicesArgsType, float]]:
            trial_path: list[tuple[str, ChoicesArgsType, float]] = []
            trial_params = trial.params
            for name, dist in trial.distributions.items():
                if name in params:
                    continue
                if name not in cat_internal_repr_cache:
                    # NOTE(nabenabe): isinstance is too slow here, and the easiest hack to avoid it
                    # is to set an empty dict. cf. https://github.com/optuna/optuna/pull/6705
                    cat_internal_repr_cache[name] = {}
                    if isinstance(dist, CategoricalDistribution):
                        cat_internal_repr_cache[name] = {c: i for i, c in enumerate(dist.choices)}
                if cat_repr := cat_internal_repr_cache[name]:
                    if (value := cat_repr.get(param_val := trial_params[name])) is None:
                        value = dist.to_internal_repr(param_val)  # most likely param_val is nan.
                    dist = cast(CategoricalDistribution, dist)  # mypy redefinition.
                    trial_path.append((name, (0, len(dist.choices) - 1, 1), value))
                else:
                    dist = cast("IntDistribution | FloatDistribution", dist)  # mypy redefinition.
                    trial_path.append((name, (dist.low, dist.high, dist.step), trial_params[name]))
            return trial_path

        for trial in trials:
            if params:
                trial_params = trial.params
                if not (nonnan_params_items <= trial_params.items()):
                    continue
                if not all(_is_nan(trial_params.get(p)) for p in nan_param_names):
                    continue
            if (leaf := tree.add_path(_get_trial_path(trial))) is not None:
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
        if isinstance(param_distribution, CategoricalDistribution):
            c_args: ChoicesArgsType = (0, len(param_distribution.choices) - 1, 1)
        elif isinstance(param_distribution, (IntDistribution, FloatDistribution)):
            c_args = (param_distribution.low, param_distribution.high, param_distribution.step)
        else:
            assert False, "Should not reach."
        tree.expand(param_name, c_args)
        # Populating must happen after the initialization above to prevent `tree` from
        # being initialized as an empty graph, which is created with n_jobs > 1
        # where we get trials[i].params = {} for some i.
        self._populate_tree(tree, trials, trial.params)
        if not tree.is_any_expandable(exclude_running):
            choices = _enumerate_candidates(*c_args)
            return param_distribution.to_external_repr(self._rng.rng.choice(choices).item())
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


@lru_cache
def _enumerate_candidates(
    low: int | float, high: int | float, step: int | float | None
) -> tuple[float, ...]:
    if step is None:
        raise ValueError(
            "FloatDistribution.step must be given for BruteForceSampler"
            " (otherwise, the search space will be infinite)."
        )
    if isinstance(low, int) and isinstance(high, int) and isinstance(step, int):
        return tuple(range(low, high + 1, step))
    else:
        low_ = decimal.Decimal(str(low))
        high_ = decimal.Decimal(str(high))
        step_ = decimal.Decimal(str(step))
        ret = []
        while low_ <= high_:
            ret.append(float(low_))
            low_ += step_
        return tuple(ret)
