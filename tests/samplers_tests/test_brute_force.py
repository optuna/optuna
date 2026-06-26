from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest

import optuna
from optuna import samplers
from optuna.samplers._brute_force import _enumerate_candidates
from optuna.samplers._brute_force import _TreeNode
from optuna.samplers._brute_force import _UNEXPANDED_NODE
from optuna.trial import Trial


if TYPE_CHECKING:
    from optuna.samplers._brute_force import _UnexpandedTreeNode

    ChildrenType = dict[float, _TreeNode | _UnexpandedTreeNode]


def _compare_with_expected_suggested_values(study: optuna.Study) -> None:
    expected_suggested_values = [
        {"a": 0, "b": -1.0},
        {"a": 0, "b": -0.5},
        {"a": 0, "b": 0.0},
        {"a": 0, "b": 0.5},
        {"a": 0, "b": 1.0},
        {"a": 1, "c": "x"},
        {"a": 1, "c": "y"},
        {"a": 1, "c": None},
        {"a": 2},
    ]
    all_suggested_values = [t.params for t in study.trials]
    assert len(all_suggested_values) == len(expected_suggested_values)
    for a in all_suggested_values:
        assert a in expected_suggested_values


def conditional_objective(trial: Trial, prune: bool = False) -> float:
    a = trial.suggest_int("a", 0, 2)
    if a == 0:
        b = trial.suggest_float("b", -1.0, 1.0, step=0.5)
        if prune:
            raise optuna.TrialPruned
        return a + b
    elif a == 1:
        c = trial.suggest_categorical("c", ["x", "y", None])
        if c == "x":
            return a + 1
        else:
            return a - 1
    else:
        return a * 2


@pytest.fixture
def template_trials_and_tree() -> tuple[list[optuna.trial.FrozenTrial], _TreeNode]:
    """
    The tree shape of this template trials.
    tree (param_name="a")
    |_ 0: a0_b_node (param_name="b")
    |   |_ 0.0: a0_b0_node (leaf; complete)
    |   |_ 1.0: a0_b1_node (leaf; complete)
    |_ 1: a1_b_node (param_name="b")
    |   |_ 0.0: a1_b0_c_node (param_name="c")
    |   |   |_ 0: a1_b0_c0_node (leaf; complete)
    |   |   |_ 1: a1_b0_c1_node (Unexpanded)
    |   |_ 1.0: a1_b1_node (Unexpanded)
    |_ 2: a2_node (Unexpanded)
    """
    a_dist = optuna.distributions.IntDistribution(0, 2)
    b_dist = optuna.distributions.FloatDistribution(0.0, 1.0, step=1.0)
    c_dist = optuna.distributions.IntDistribution(0, 1)
    trials = []
    for params in [{"a": 0, "b": 0.0}, {"a": 0, "b": 1.0}, {"a": 1, "b": 0.0, "c": 0}]:
        dists = {k: {"a": a_dist, "b": b_dist, "c": c_dist}[k] for k in params}
        s = optuna.trial.TrialState.COMPLETE
        trials.append(optuna.create_trial(state=s, value=0.0, params=params, distributions=dists))
    a_cargs = (a_dist.low, a_dist.high, a_dist.step)
    b_cargs = (b_dist.low, b_dist.high, b_dist.step)
    c_cargs = (c_dist.low, c_dist.high, c_dist.step)
    leaf_node = _TreeNode(children={})  # a0_b0_node, a0_b1_node, a1_b0_c0_node
    unexpanded_node = _UNEXPANDED_NODE  # a1_b0_c1_node, a1_b1_node, a2_node
    a0_b_node_children: ChildrenType = {0.0: leaf_node, 1.0: leaf_node}
    a0_b_node = _TreeNode(param_name="b", children=a0_b_node_children, choices_args=b_cargs)
    a1_b0_c_node_children: ChildrenType = {0: leaf_node, 1: unexpanded_node}
    a1_b0_c_node = _TreeNode(param_name="c", children=a1_b0_c_node_children, choices_args=c_cargs)
    a1_b_node_children: ChildrenType = {0.0: a1_b0_c_node, 1.0: unexpanded_node}
    a1_b_node = _TreeNode("b", children=a1_b_node_children, choices_args=b_cargs)
    tree_children: ChildrenType = {0: a0_b_node, 1: a1_b_node, 2: unexpanded_node}
    tree = _TreeNode(param_name="a", children=tree_children, choices_args=a_cargs)
    return trials, tree


def test_tree_node_add_paths(
    template_trials_and_tree: tuple[list[optuna.trial.FrozenTrial], _TreeNode],
) -> None:
    template_trials, template_tree = template_trials_and_tree
    template_trials.append(deepcopy(template_trials[0]))  # Duplicate a trial for robustness check.
    tree = _TreeNode()
    samplers.BruteForceSampler._populate_tree(tree, template_trials, {})
    assert tree == template_tree


def test_tree_node_add_paths_error() -> None:
    tree = _TreeNode()
    tree.add_path([("a", (0, 2, 1), 0)])
    with pytest.raises(ValueError):
        tree.add_path([("a", (0, 1, 1), 0)])

    tree = _TreeNode()
    tree.add_path([("a", (0, 2, 1), 0)])
    with pytest.raises(ValueError):
        tree.add_path([("b", (0, 2, 1), 0)])


def test_tree_node_count_unexpanded(
    template_trials_and_tree: tuple[list[optuna.trial.FrozenTrial], _TreeNode],
) -> None:
    template_trials, template_tree = template_trials_and_tree
    only_a = {"a": template_trials[0].distributions["a"]}
    running_trial = optuna.create_trial(
        state=optuna.trial.TrialState.RUNNING, params={"a": 2}, distributions=only_a
    )
    template_trials.append(running_trial)
    tree = _TreeNode()
    samplers.BruteForceSampler._populate_tree(tree, template_trials, {})
    n_unexpanded = template_tree.count_unexpanded(exclude_running=True)
    assert template_tree.count_unexpanded(exclude_running=False) == n_unexpanded, (
        "No Running in template"
    )
    assert tree.count_unexpanded(exclude_running=False) == n_unexpanded
    assert tree.count_unexpanded(exclude_running=True) == n_unexpanded - 1


def test_study_optimize_with_single_search_space() -> None:
    study = optuna.create_study(sampler=samplers.BruteForceSampler())
    study.optimize(conditional_objective)
    _compare_with_expected_suggested_values(study)


def test_study_optimize_with_pruned_trials() -> None:
    study = optuna.create_study(sampler=samplers.BruteForceSampler())
    study.optimize(lambda trial: conditional_objective(trial, prune=True))
    _compare_with_expected_suggested_values(study)


def test_study_optimize_with_infinite_search_space() -> None:
    def objective(trial: Trial) -> float:
        return trial.suggest_float("a", 0, 2)

    study = optuna.create_study(sampler=samplers.BruteForceSampler())

    with pytest.raises(ValueError):
        study.optimize(objective)


def test_study_optimize_with_nan() -> None:
    def objective(trial: Trial) -> float:
        trial.suggest_categorical("a", [0.0, float("nan")])
        return 1.0

    study = optuna.create_study(sampler=samplers.BruteForceSampler())
    study.optimize(objective)

    all_suggested_values = [t.params["a"] for t in study.trials]
    assert len(all_suggested_values) == 2
    assert 0.0 in all_suggested_values
    assert np.isnan(all_suggested_values[0]) or np.isnan(all_suggested_values[1])


def test_study_optimize_with_single_search_space_user_added() -> None:
    study = optuna.create_study(sampler=samplers.BruteForceSampler())

    # Manually add a trial. This should not be tried again.
    study.add_trial(
        optuna.create_trial(
            params={"a": 0, "b": -1.0},
            value=0.0,
            distributions={
                "a": optuna.distributions.IntDistribution(0, 2),
                "b": optuna.distributions.FloatDistribution(-1.0, 1.0, step=0.5),
            },
        )
    )

    study.optimize(conditional_objective)
    _compare_with_expected_suggested_values(study)


def test_study_optimize_with_dynamic_range_search_space() -> None:
    def objective_nonconstant_range(trial: Trial) -> float:
        x = trial.suggest_int("x", -1, trial.number)
        return x

    study = optuna.create_study(sampler=samplers.BruteForceSampler())
    with pytest.raises(ValueError):
        study.optimize(objective_nonconstant_range, n_trials=10)


def test_study_optimize_with_increasing_search_space() -> None:
    def objective_increasing_variable(trial: Trial) -> float:
        return sum(trial.suggest_int(f"x{i}", 0, 0) for i in range(2))

    study = optuna.create_study(sampler=samplers.BruteForceSampler())
    study.add_trial(
        optuna.create_trial(
            params={"x0": 0},
            value=0.0,
            distributions={"x0": optuna.distributions.IntDistribution(0, 0)},
        )
    )
    with pytest.raises(ValueError):
        study.optimize(objective_increasing_variable, n_trials=10)


def test_study_optimize_with_decreasing_search_space() -> None:
    def objective_decreasing_variable(trial: Trial) -> float:
        return trial.suggest_int("x0", 0, 0)

    study = optuna.create_study(sampler=samplers.BruteForceSampler())
    study.add_trial(
        optuna.create_trial(
            params={"x0": 0, "x1": 0},
            value=0.0,
            distributions={
                "x0": optuna.distributions.IntDistribution(0, 0),
                "x1": optuna.distributions.IntDistribution(0, 0),
            },
        )
    )
    with pytest.raises(ValueError):
        study.optimize(objective_decreasing_variable, n_trials=10)


def test_study_optimize_with_failed_trials() -> None:
    def objective(trial: Trial) -> float:
        trial.suggest_int("x", 0, 99)
        return np.nan

    study = optuna.create_study(sampler=samplers.BruteForceSampler())
    study.optimize(objective, n_trials=100)

    expected_suggested_values = [{"x": i} for i in range(100)]
    all_suggested_values = [t.params for t in study.trials]
    assert len(all_suggested_values) == len(expected_suggested_values)
    for a in expected_suggested_values:
        assert a in all_suggested_values


def test_parallel_optimize() -> None:
    study = optuna.create_study(sampler=samplers.BruteForceSampler())
    trial1 = study.ask()
    trial2 = study.ask()
    x1 = trial1.suggest_categorical("x", ["a", "b"])
    x2 = trial2.suggest_categorical("x", ["a", "b"])
    assert {x1, x2} == {"a", "b"}


def test_not_avoid_premature_stop() -> None:
    study = optuna.create_study(sampler=samplers.BruteForceSampler(seed=42))

    trials = [study.ask() for _ in range(3)]

    assert trials[0].suggest_int("x", 0, 1) == 0
    assert trials[0].suggest_int("y", 0, 1) == 1

    study.tell(trials[0], 0.0)

    assert trials[1].suggest_int("x", 0, 1) == 1

    assert trials[2].suggest_int("x", 0, 1) == 0
    assert trials[2].suggest_int("y", 0, 1) == 0

    with patch.object(study, "stop", return_value=None) as mock_stop:
        # At this moment, the BruteForceSampler knows:
        #
        #   trials[0]: x = 0, y = 1  # Completed
        #   trials[1]: x = 1         # Running
        #   trials[2]: x = 0, y = 0  # Running
        #
        # Since the sampler assumes that running trials already suggest all parameters, the sampler
        # considers all possible combinations of x and y have been exhausted.
        study.tell(trials[1], 0.0)
        study.tell(trials[2], 0.0)
        assert mock_stop.call_count == 2


def test_avoid_premature_stop() -> None:
    study = optuna.create_study(
        sampler=samplers.BruteForceSampler(seed=42, avoid_premature_stop=True)
    )

    trials = [study.ask() for _ in range(5)]

    assert trials[0].suggest_int("x", 0, 1) == 0
    assert trials[0].suggest_int("y", 0, 1) == 1

    study.tell(trials[0], 0.0)

    assert trials[1].suggest_int("x", 0, 1) == 1

    assert trials[2].suggest_int("x", 0, 1) == 0
    assert trials[2].suggest_int("y", 0, 1) == 0

    with patch.object(study, "stop", return_value=None) as mock_stop:
        # At this moment, the BruteForceSampler knows:
        #
        #   trials[0]: x = 0, y = 1  # Completed
        #   trials[1]: x = 1         # Running
        #   trials[2]: x = 0, y = 0  # Running
        #
        # If `avoid_premature_stop` is `True`, the sampler assumes that running trials may still
        # suggest new parameters, considering the possibility for trials[1] to suggest `y`.
        # So the sampler should not stop.
        study.tell(trials[2], 0.0)
        mock_stop.assert_not_called()

    assert trials[1].suggest_int("y", 0, 1) == 0

    assert trials[3].suggest_int("x", 0, 1) == 1
    assert trials[3].suggest_int("y", 0, 1) == 1

    with patch.object(study, "stop", return_value=None) as mock_stop:
        # At this moment, the BruteForceSampler knows:
        #
        #   trials[0]: x = 0, y = 1  # Completed
        #   trials[1]: x = 1, y = 0  # Running
        #   trials[2]: x = 0, y = 0  # Completed
        #   trials[3]: x = 1, y = 1  # Running
        #
        # However, the BruteForceSampler assumes that trials[3] may possibly have
        # another parameter that has not yet been suggested.
        study.tell(trials[1], 0.0)
        mock_stop.assert_not_called()

    with patch.object(study, "stop", return_value=None) as mock_stop:
        # At this moment, the BruteForceSampler knows:
        #
        #   trials[0]: x = 0, y = 1  # Completed
        #   trials[1]: x = 1, y = 0  # Completed
        #   trials[2]: x = 0, y = 0  # Completed
        #   trials[3]: x = 1, y = 1  # Running
        #
        # All possible combinations of x and y have been exhausted.
        study.tell(trials[3], 0.0)
        mock_stop.assert_called_once()


def test_objective_with_nan() -> None:
    weird_choices = [float("inf"), -float("inf"), float("nan"), None]
    n_params = 3

    def _objective_with_nan(trial: optuna.Trial) -> float:
        [trial.suggest_categorical(f"c{i}", weird_choices) for i in range(n_params)]
        return 0.0

    sampler = optuna.samplers.BruteForceSampler(seed=0)
    study = optuna.create_study(sampler=sampler)
    study.optimize(_objective_with_nan)
    assert len(study.trials) == len(weird_choices) ** n_params


@pytest.mark.parametrize("low,high,step", [(1.0, 3.0, 0.5), (0, 10, 1), (0, 10, 3)])
def test_enumerate_candidates_sorted_uniform(
    low: int | float, high: int | float, step: int | float | None
) -> None:
    candidates = list(_enumerate_candidates(low, high, step))
    assert candidates == sorted(candidates), "candidates must be sorted"
    if len(candidates) >= 2:
        diffs = [candidates[i + 1] - candidates[i] for i in range(len(candidates) - 1)]
        assert all(abs(d - diffs[0]) < 1e-12 for d in diffs), "candidates must be uniformly spaced"


@pytest.mark.parametrize("low,high,step", [(1.0, 3.0, None), (0, 10, None)])
def test_enumerate_candidates_step_is_none(
    low: int | float, high: int | float, step: int | float | None
) -> None:
    with pytest.raises(ValueError):
        _enumerate_candidates(low, high, step)


def test_non_divisible_step_with_high_that_fails_to_fallback_to_divisible_range() -> None:
    study = optuna.create_study(sampler=optuna.samplers.BruteForceSampler())
    study.ask({"x": optuna.distributions.FloatDistribution(0.0, 0.3, step=0.1)})
    with pytest.raises(ValueError):
        study.ask({"x": optuna.distributions.FloatDistribution(0.0, 0.3, step=0.1 - 1e-17)})


def test_non_divisible_step_with_successful_fallback() -> None:
    study = optuna.create_study(sampler=optuna.samplers.BruteForceSampler())
    study.ask({"x": optuna.distributions.FloatDistribution(0.0, 0.5, step=0.2)})
