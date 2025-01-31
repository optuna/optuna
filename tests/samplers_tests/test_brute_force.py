import time

import numpy as np
import pytest

import optuna
from optuna import samplers
from optuna.samplers._brute_force import _TreeNode
from optuna.trial import Trial


def test_tree_node_add_paths() -> None:
    tree = _TreeNode()
    leafs = [
        tree.add_path([("a", [0, 1, 2], 0), ("b", [0.0, 1.0], 0.0)]),
        tree.add_path([("a", [0, 1, 2], 0), ("b", [0.0, 1.0], 1.0)]),
        tree.add_path([("a", [0, 1, 2], 0), ("b", [0.0, 1.0], 1.0)]),
        tree.add_path([("a", [0, 1, 2], 1), ("b", [0.0, 1.0], 0.0), ("c", [0, 1], 0)]),
        tree.add_path([("a", [0, 1, 2], 1), ("b", [0.0, 1.0], 0.0)]),
    ]
    for leaf in leafs:
        assert leaf is not None
        if leaf.children is None:
            leaf.set_leaf()

    assert tree == _TreeNode(
        param_name="a",
        children={
            0: _TreeNode(
                param_name="b",
                children={
                    0.0: _TreeNode(param_name=None, children={}),
                    1.0: _TreeNode(param_name=None, children={}),
                },
            ),
            1: _TreeNode(
                param_name="b",
                children={
                    0.0: _TreeNode(
                        param_name="c",
                        children={
                            0: _TreeNode(param_name=None, children={}),
                            1: _TreeNode(),
                        },
                    ),
                    1.0: _TreeNode(),
                },
            ),
            2: _TreeNode(),
        },
    )


def test_tree_node_add_paths_error() -> None:
    with pytest.raises(ValueError):
        tree = _TreeNode()
        tree.add_path([("a", [0, 1, 2], 0)])
        tree.add_path([("a", [0, 1], 0)])

    with pytest.raises(ValueError):
        tree = _TreeNode()
        tree.add_path([("a", [0, 1, 2], 0)])
        tree.add_path([("b", [0, 1, 2], 0)])


def test_tree_node_count_unexpanded() -> None:
    tree = _TreeNode(
        param_name="a",
        children={
            0: _TreeNode(
                param_name="b",
                children={
                    0.0: _TreeNode(param_name=None, children={}),
                    1.0: _TreeNode(param_name=None, children={}),
                },
            ),
            1: _TreeNode(
                param_name="b",
                children={
                    0.0: _TreeNode(
                        param_name="c",
                        children={
                            0: _TreeNode(param_name=None, children={}),
                            1: _TreeNode(),
                        },
                    ),
                    1.0: _TreeNode(),
                },
            ),
            2: _TreeNode(is_running=True),
        },
    )
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


def test_parallel_optimize_with_sleep() -> None:
    def objective(trial: Trial) -> float:
        x = trial.suggest_int("x", 0, 1)
        time.sleep(x)
        y = trial.suggest_int("y", 0, 1)
        return x + y

    # Seed is fixed to reproduce the same result.
    # See: https://github.com/optuna/optuna/issues/5780
    study = optuna.create_study(sampler=samplers.BruteForceSampler(seed=42))
    study.optimize(objective, n_jobs=2)
    expected_suggested_values = [
        {"x": 0, "y": 0},
        {"x": 0, "y": 1},
        {"x": 1, "y": 0},
    ]
    all_suggested_values = [t.params for t in study.trials]
    assert len(all_suggested_values) == len(expected_suggested_values)
    for expected_value in expected_suggested_values:
        assert expected_value in all_suggested_values

    study = optuna.create_study(
        sampler=samplers.BruteForceSampler(seed=42, avoid_premature_stop=True)
    )
    study.optimize(objective, n_jobs=2)
    expected_suggested_values = [
        {"x": 0, "y": 0},
        {"x": 0, "y": 1},
        {"x": 1, "y": 0},
        {"x": 1, "y": 1},
    ]
    all_suggested_values = [t.params for t in study.trials]
    for expected_value in expected_suggested_values:
        assert expected_value in all_suggested_values
