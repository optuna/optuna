from __future__ import annotations

from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pytest

import optuna
from optuna import samplers
from optuna.samplers._brute_force import _LAZY_NODE
from optuna.samplers._brute_force import _TreeNode
from optuna.trial import Trial


@pytest.fixture
def template_trials() -> list[optuna.trial.FrozenTrial]:
    dists = {
        "a": optuna.distributions.IntDistribution(0, 2),
        "b": optuna.distributions.FloatDistribution(0.0, 1.0, step=1.0),
        "c": optuna.distributions.IntDistribution(0, 1),
    }
    trials = []
    for params in [{"a": 0, "b": 0.0}, {"a": 0, "b": 1.0}, {"a": 1, "b": 0.0, "c": 0}]:
        ds = {k: dists[k] for k in params}
        s = optuna.trial.TrialState.COMPLETE
        trials.append(optuna.create_trial(state=s, value=0.0, params=params, distributions=ds))
    return trials


def test_tree_node_add_paths(template_trials: list[optuna.trial.FrozenTrial]) -> None:
    template_trials.append(deepcopy(template_trials[0]))  # Duplicate a trial for robustness check.
    tree = _TreeNode()
    samplers.BruteForceSampler._populate_tree(tree, template_trials, {})
    leaf_node = _TreeNode(param_name=None, children={})
    a0_node = _TreeNode(param_name="b", children={0.0: leaf_node, 1.0: leaf_node})
    a1_b0_node = _TreeNode(param_name="c", children={0: leaf_node, 1: _LAZY_NODE})
    a1_node = _TreeNode(param_name="b", children={0.0: a1_b0_node, 1.0: _LAZY_NODE})
    assert tree == _TreeNode(param_name="a", children={0: a0_node, 1: a1_node, 2: _LAZY_NODE})


def test_tree_node_add_paths_error() -> None:
    with pytest.raises(ValueError):
        tree = _TreeNode()
        tree.add_path([("a", [0, 1, 2], 0)])
        tree.add_path([("a", [0, 1], 0)])

    with pytest.raises(ValueError):
        tree = _TreeNode()
        tree.add_path([("a", [0, 1, 2], 0)])
        tree.add_path([("b", [0, 1, 2], 0)])


def test_tree_node_count_unexpanded(template_trials: list[optuna.trial.FrozenTrial]) -> None:
    only_a = {"a": template_trials[0].distributions["a"]}
    running_trial = optuna.create_trial(
        state=optuna.trial.TrialState.RUNNING, params={"a": 2}, distributions=only_a
    )
    template_trials.append(running_trial)
    tree = _TreeNode()
    samplers.BruteForceSampler._populate_tree(tree, template_trials, {})
    assert tree.count_unexpanded(exclude_running=False) == 3
    assert tree.count_unexpanded(exclude_running=True) == 2


def test_study_optimize_with_single_search_space() -> None:
    def objective(trial: Trial) -> float:
        a = trial.suggest_int("a", 0, 2)

        if a == 0:
            b = trial.suggest_float("b", -1.0, 1.0, step=0.5)
            return a + b
        elif a == 1:
            c = trial.suggest_categorical("c", ["x", "y", None])
            if c == "x":
                return a + 1
            else:
                return a - 1
        else:
            return a * 2

    study = optuna.create_study(sampler=samplers.BruteForceSampler())
    study.optimize(objective)

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


def test_study_optimize_with_pruned_trials() -> None:
    def objective(trial: Trial) -> float:
        a = trial.suggest_int("a", 0, 2)

        if a == 0:
            trial.suggest_float("b", -1.0, 1.0, step=0.5)
            raise optuna.TrialPruned
        elif a == 1:
            c = trial.suggest_categorical("c", ["x", "y", None])
            if c == "x":
                return a + 1
            else:
                return a - 1
        else:
            return a * 2

    study = optuna.create_study(sampler=samplers.BruteForceSampler())
    study.optimize(objective)

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
    def objective(trial: Trial) -> float:
        a = trial.suggest_int("a", 0, 2)

        if a == 0:
            b = trial.suggest_float("b", -1.0, 1.0, step=0.5)
            return a + b
        elif a == 1:
            c = trial.suggest_categorical("c", ["x", "y", None])
            if c == "x":
                return a + 1
            else:
                return a - 1
        else:
            return a * 2

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

    study.optimize(objective)

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


def test_study_optimize_with_nonconstant_search_space() -> None:
    def objective_nonconstant_range(trial: Trial) -> float:
        x = trial.suggest_int("x", -1, trial.number)
        return x

    study = optuna.create_study(sampler=samplers.BruteForceSampler())
    with pytest.raises(ValueError):
        study.optimize(objective_nonconstant_range, n_trials=10)

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
