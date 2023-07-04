import numpy as np
import pytest

import optuna
from optuna import samplers
from optuna.distributions import CategoricalDistribution
from optuna.samplers._brute_force import _build_tree
from optuna.samplers._brute_force import _TreeNode
from optuna.samplers._brute_force import _TrialInfo
from optuna.trial import Trial


def test_build_tree() -> None:
    tree = _TreeNode()

    distr_a = CategoricalDistribution([0, 1, 2])
    distr_b = CategoricalDistribution([0.0, 1.0])
    distr_c = CategoricalDistribution([0, 1])
    trial_infos = [
        _TrialInfo(is_running=False, params={"a": (distr_a, 0), "b": (distr_b, 0.0)}),
        _TrialInfo(is_running=False, params={"a": (distr_a, 0), "b": (distr_b, 1.0)}),
        _TrialInfo(is_running=False, params={"a": (distr_a, 0), "b": (distr_b, 1.0)}),
        _TrialInfo(
            is_running=False, params={"a": (distr_a, 1), "b": (distr_b, 0.0), "c": (distr_c, 0)}
        ),
        _TrialInfo(is_running=True, params={"a": (distr_a, 1), "b": (distr_b, 0.0)}),
    ]
    tree = _build_tree(trial_infos)

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


def test_build_tree_error() -> None:
    with pytest.raises(ValueError):
        trial_infos = [
            _TrialInfo(is_running=True, params={"a": (CategoricalDistribution([0, 1, 2]), 0)}),
            _TrialInfo(is_running=True, params={"a": (CategoricalDistribution([0, 1]), 0)}),
        ]
        _build_tree(trial_infos)

    with pytest.raises(ValueError):
        trial_infos = [
            _TrialInfo(is_running=True, params={"a": (CategoricalDistribution([0, 1, 2]), 0)}),
            _TrialInfo(is_running=True, params={"b": (CategoricalDistribution([0, 1, 2]), 0)}),
        ]
        _build_tree(trial_infos)


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
            2: _TreeNode(),
        },
    )
    assert tree.count_unexpanded() == 3


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


def test_study_optimize_with_varying_orders() -> None:
    def objective(trial: Trial) -> float:
        if trial.number % 2 == 0:
            trial.suggest_int("a", 0, 1)
            trial.suggest_int("b", 0, 1)
        else:
            trial.suggest_int("b", 0, 1)
            trial.suggest_int("a", 0, 1)
        return 0.0

    study = optuna.create_study(sampler=samplers.BruteForceSampler())

    study.optimize(objective)

    expected_suggested_values = [
        {"a": 0, "b": 0},
        {"a": 0, "b": 1},
        {"a": 1, "b": 0},
        {"a": 1, "b": 1},
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
