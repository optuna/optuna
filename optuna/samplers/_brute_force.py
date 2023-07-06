from __future__ import annotations

from dataclasses import dataclass
import decimal
from typing import Any
from typing import NoReturn
from typing import Sequence

import numpy as np

from optuna._experimental import experimental_class
from optuna.distributions import _categorical_choice_equal
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.samplers import BaseSampler
from optuna.study import Study
from optuna.trial import create_trial
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


@dataclass
class _TreeNode:
    # This is a class to represent the tree of search space.

    # A tree node has three states:
    # 1. Unexpanded. This is represented by param_name=None and children=None.
    # 2. Leaf. This is represented by children={} and param_name=None.
    # 3. Normal node. It has a param_name and non-empty children.

    param_name: str | None = None
    children: dict[Any, "_TreeNode"] | None = None

    def count_unexpanded(self) -> int:
        # Count the number of unexpanded nodes in the subtree.
        return (
            1
            if self.children is None
            else sum(child.count_unexpanded() for child in self.children.values())
        )

    def sample_child(self, rng: np.random.RandomState) -> Any:
        assert self.children is not None
        # Sample an unexpanded node in the subtree uniformly, and return the first
        # parameter value in the path to the node.
        # Equivalently, we sample the child node with weights proportional to the number
        # of unexpanded nodes in the subtree.
        weights = np.array(
            [child.count_unexpanded() for child in self.children.values()], dtype=np.float64
        )
        weights /= weights.sum()
        return rng.choice(list(self.children.keys()), p=weights)


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
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.RandomState(seed)

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> dict[str, BaseDistribution]:
        return {}

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        return {}

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        trials = study.get_trials(
            deepcopy=False,
            states=(
                TrialState.COMPLETE,
                TrialState.RUNNING,
            ),
        )
        trials = [t for t in trials if t.number != trial.number]
        trial_infos = _get_trial_infos(trials, trial.params)
        trial_infos.append(
            _TrialInfo(is_running=True, params={param_name: (param_distribution, None)})
        )
        tree = _build_tree(trial_infos)

        if tree.count_unexpanded() == 0:
            candidates = _enumerate_candidates(param_distribution)
            return param_distribution.to_external_repr(self._rng.choice(candidates))
        else:
            return param_distribution.to_external_repr(tree.sample_child(self._rng))

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Sequence[float] | None,
    ) -> None:
        trials = study.get_trials(
            deepcopy=False,
            states=(
                TrialState.COMPLETE,
                TrialState.RUNNING,
            ),
        )
        trials = [
            t
            if t.number != trial.number
            else create_trial(
                state=state,  # Set current trial as complete.
                values=values,
                params=trial.params,
                distributions=trial.distributions,
            )
            for t in trials
        ]
        trial_infos = _get_trial_infos(trials, {})
        tree = _build_tree(trial_infos)

        if tree.count_unexpanded() == 0:
            study.stop()


@dataclass
class _TrialInfo:
    is_running: bool
    params: dict[str, tuple[BaseDistribution, Any | None]]


def _build_tree(trial_infos: list[_TrialInfo]) -> _TreeNode:
    def _raise_change_dependency() -> NoReturn:
        raise ValueError(
            "Parameter dependency change detected. "
            + "Did you change the objective function during optimization?"
        )

    nonempty_trial_infos = [info for info in trial_infos if len(info.params) > 0]
    if len(trial_infos) == 0:
        return _TreeNode()  # Unexpanded node
    elif len(trial_infos) > 0 and len(nonempty_trial_infos) == 0:
        return _TreeNode(param_name=None, children={})  # Leaf node
    else:
        if any(info for info in trial_infos if len(info.params) == 0 and not info.is_running):
            _raise_change_dependency()

        pivot_candidates = [(p, d) for (p, (d, _)) in nonempty_trial_infos[0].params.items()]
        for param, distr in pivot_candidates:
            if all(
                (param in info.params and info.params[param][0] == distr)
                for info in nonempty_trial_infos
            ):
                break
        else:
            _raise_change_dependency()

        children_trial_infos = {
            value: [info for info in nonempty_trial_infos if info.params[param][1] == value]
            for value in _enumerate_candidates(distr)
        }

        for infos in children_trial_infos.values():
            for info in infos:
                del info.params[param]

        return _TreeNode(
            param_name=param,
            children={
                value: _build_tree(infos) for (value, infos) in children_trial_infos.items()
            },
        )


def _get_trial_infos(
    trials: list[FrozenTrial], current_params: dict[str, Any]
) -> list[_TrialInfo]:
    return [
        _TrialInfo(
            is_running=not trial.state.is_finished(),
            params={
                param: (
                    trial.distributions[param],
                    trial.distributions[param].to_internal_repr(trial.params[param]),
                )
                for param in trial.params
                if param not in current_params
            },
        )
        for trial in trials
        if all(
            param in trial.params and _categorical_choice_equal(trial.params[param], value)
            for (param, value) in current_params.items()
        )
    ]


def _enumerate_candidates(param_distribution: BaseDistribution) -> Sequence[Any]:
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
