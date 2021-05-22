import itertools
import random
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from unittest.mock import Mock
from unittest.mock import patch
from unittest.mock import PropertyMock

import numpy as np
import pytest

import optuna
from optuna.samplers import MOTPESampler
from optuna.samplers._tpe import multi_objective_sampler


class MockSystemAttr:
    def __init__(self) -> None:
        self.value = {}  # type: Dict[str, dict]

    def set_trial_system_attr(self, _: int, key: str, value: dict) -> None:
        self.value[key] = value


def test_sample_relative() -> None:
    sampler = MOTPESampler()
    # Study and frozen-trial are not supposed to be accessed.
    study = Mock(spec=[])
    frozen_trial = Mock(spec=[])
    assert sampler.sample_relative(study, frozen_trial, {}) == {}


def test_infer_relative_search_space() -> None:
    sampler = MOTPESampler()
    # Study and frozen-trial are not supposed to be accessed.
    study = Mock(spec=[])
    frozen_trial = Mock(spec=[])
    assert sampler.infer_relative_search_space(study, frozen_trial) == {}


def test_sample_independent_seed_fix() -> None:
    study = optuna.create_study(directions=["minimize", "maximize"])
    dist = optuna.distributions.UniformDistribution(1.0, 100.0)

    random.seed(128)
    past_trials = [frozen_trial_factory(i, [random.random(), random.random()]) for i in range(16)]
    # Prepare a trial and a sample for later checks.
    trial = frozen_trial_factory(16, [0, 0])
    sampler = MOTPESampler(seed=0)
    attrs = MockSystemAttr()
    with patch.object(study._storage, "get_all_trials", return_value=past_trials), patch.object(
        study._storage, "set_trial_system_attr", side_effect=attrs.set_trial_system_attr
    ), patch.object(study._storage, "get_trial", return_value=trial), patch(
        "optuna.trial.Trial.system_attrs", new_callable=PropertyMock
    ) as mock1, patch(
        "optuna.trial.FrozenTrial.system_attrs",
        new_callable=PropertyMock,
    ) as mock2:
        mock1.return_value = attrs.value
        mock2.return_value = attrs.value
        suggestion = sampler.sample_independent(study, trial, "param-a", dist)

    sampler = MOTPESampler(seed=0)
    attrs = MockSystemAttr()
    with patch.object(study._storage, "get_all_trials", return_value=past_trials), patch.object(
        study._storage, "set_trial_system_attr", side_effect=attrs.set_trial_system_attr
    ), patch.object(study._storage, "get_trial", return_value=trial), patch(
        "optuna.trial.Trial.system_attrs", new_callable=PropertyMock
    ) as mock1, patch(
        "optuna.trial.FrozenTrial.system_attrs",
        new_callable=PropertyMock,
    ) as mock2:
        mock1.return_value = attrs.value
        mock2.return_value = attrs.value
        assert sampler.sample_independent(study, trial, "param-a", dist) == suggestion

    sampler = MOTPESampler(seed=1)
    attrs = MockSystemAttr()
    with patch.object(study._storage, "get_all_trials", return_value=past_trials), patch.object(
        study._storage, "set_trial_system_attr", side_effect=attrs.set_trial_system_attr
    ), patch.object(study._storage, "get_trial", return_value=trial), patch(
        "optuna.trial.Trial.system_attrs", new_callable=PropertyMock
    ) as mock1, patch(
        "optuna.trial.FrozenTrial.system_attrs",
        new_callable=PropertyMock,
    ) as mock2:
        mock1.return_value = attrs.value
        mock2.return_value = attrs.value
        assert sampler.sample_independent(study, trial, "param-a", dist) != suggestion


def test_sample_independent_prior() -> None:
    study = optuna.create_study(directions=["minimize", "maximize"])
    dist = optuna.distributions.UniformDistribution(1.0, 100.0)

    random.seed(128)
    past_trials = [frozen_trial_factory(i, [random.random(), random.random()]) for i in range(16)]

    # Prepare a trial and a sample for later checks.
    trial = frozen_trial_factory(16, [0, 0])
    sampler = MOTPESampler(seed=0)
    attrs = MockSystemAttr()
    with patch.object(study._storage, "get_all_trials", return_value=past_trials), patch.object(
        study._storage, "set_trial_system_attr", side_effect=attrs.set_trial_system_attr
    ), patch.object(study._storage, "get_trial", return_value=trial), patch(
        "optuna.trial.Trial.system_attrs", new_callable=PropertyMock
    ) as mock1, patch(
        "optuna.trial.FrozenTrial.system_attrs",
        new_callable=PropertyMock,
    ) as mock2:
        mock1.return_value = attrs.value
        mock2.return_value = attrs.value
        suggestion = sampler.sample_independent(study, trial, "param-a", dist)

    sampler = MOTPESampler(consider_prior=False, seed=0)
    attrs = MockSystemAttr()
    with patch.object(study._storage, "get_all_trials", return_value=past_trials), patch.object(
        study._storage, "set_trial_system_attr", side_effect=attrs.set_trial_system_attr
    ), patch.object(study._storage, "get_trial", return_value=trial), patch(
        "optuna.trial.Trial.system_attrs", new_callable=PropertyMock
    ) as mock1, patch(
        "optuna.trial.FrozenTrial.system_attrs",
        new_callable=PropertyMock,
    ) as mock2:
        mock1.return_value = attrs.value
        mock2.return_value = attrs.value
        assert sampler.sample_independent(study, trial, "param-a", dist) != suggestion

    sampler = MOTPESampler(prior_weight=0.5, seed=0)
    attrs = MockSystemAttr()
    with patch.object(study._storage, "get_all_trials", return_value=past_trials), patch.object(
        study._storage, "set_trial_system_attr", side_effect=attrs.set_trial_system_attr
    ), patch.object(study._storage, "get_trial", return_value=trial), patch(
        "optuna.trial.Trial.system_attrs", new_callable=PropertyMock
    ) as mock1, patch(
        "optuna.trial.FrozenTrial.system_attrs",
        new_callable=PropertyMock,
    ) as mock2:
        mock1.return_value = attrs.value
        mock2.return_value = attrs.value
        assert sampler.sample_independent(study, trial, "param-a", dist) != suggestion


def test_sample_independent_n_startup_trial() -> None:
    study = optuna.create_study(directions=["minimize", "maximize"])
    dist = optuna.distributions.UniformDistribution(1.0, 100.0)
    random.seed(128)
    past_trials = [frozen_trial_factory(i, [random.random(), random.random()]) for i in range(16)]

    trial = frozen_trial_factory(16, [0, 0])
    sampler = MOTPESampler(n_startup_trials=16, seed=0)
    attrs = MockSystemAttr()
    with patch.object(
        study._storage, "get_all_trials", return_value=past_trials[:15]
    ), patch.object(
        study._storage, "set_trial_system_attr", side_effect=attrs.set_trial_system_attr
    ), patch.object(
        study._storage, "get_trial", return_value=trial
    ), patch(
        "optuna.trial.Trial.system_attrs", new_callable=PropertyMock
    ) as mock1, patch(
        "optuna.trial.FrozenTrial.system_attrs",
        new_callable=PropertyMock,
    ) as mock2, patch.object(
        optuna.samplers.RandomSampler,
        "sample_independent",
        return_value=1.0,
    ) as sample_method:
        mock1.return_value = attrs.value
        mock2.return_value = attrs.value
        sampler.sample_independent(study, trial, "param-a", dist)
    assert sample_method.call_count == 1

    sampler = MOTPESampler(n_startup_trials=16, seed=0)
    attrs = MockSystemAttr()
    with patch.object(study._storage, "get_all_trials", return_value=past_trials), patch.object(
        study._storage, "set_trial_system_attr", side_effect=attrs.set_trial_system_attr
    ), patch.object(study._storage, "get_trial", return_value=trial), patch(
        "optuna.trial.Trial.system_attrs", new_callable=PropertyMock
    ) as mock1, patch(
        "optuna.trial.FrozenTrial.system_attrs",
        new_callable=PropertyMock,
    ) as mock2, patch.object(
        optuna.samplers.RandomSampler,
        "sample_independent",
        return_value=1.0,
    ) as sample_method:
        mock1.return_value = attrs.value
        mock2.return_value = attrs.value
        sampler.sample_independent(study, trial, "param-a", dist)
    assert sample_method.call_count == 0


def test_sample_independent_misc_arguments() -> None:
    study = optuna.create_study(directions=["minimize", "maximize"])
    dist = optuna.distributions.UniformDistribution(1.0, 100.0)
    random.seed(128)
    past_trials = [frozen_trial_factory(i, [random.random(), random.random()]) for i in range(32)]

    # Prepare a trial and a sample for later checks.
    trial = frozen_trial_factory(16, [0, 0])
    sampler = MOTPESampler(seed=0)
    attrs = MockSystemAttr()
    with patch.object(study._storage, "get_all_trials", return_value=past_trials), patch.object(
        study._storage, "set_trial_system_attr", side_effect=attrs.set_trial_system_attr
    ), patch.object(study._storage, "get_trial", return_value=trial), patch(
        "optuna.trial.Trial.system_attrs", new_callable=PropertyMock
    ) as mock1, patch(
        "optuna.trial.FrozenTrial.system_attrs",
        new_callable=PropertyMock,
    ) as mock2:
        mock1.return_value = attrs.value
        mock2.return_value = attrs.value
        suggestion = sampler.sample_independent(study, trial, "param-a", dist)

    # Test misc. parameters.
    sampler = MOTPESampler(n_ehvi_candidates=13, seed=0)
    attrs = MockSystemAttr()
    with patch.object(study._storage, "get_all_trials", return_value=past_trials), patch.object(
        study._storage, "set_trial_system_attr", side_effect=attrs.set_trial_system_attr
    ), patch.object(study._storage, "get_trial", return_value=trial), patch(
        "optuna.trial.Trial.system_attrs", new_callable=PropertyMock
    ) as mock1, patch(
        "optuna.trial.FrozenTrial.system_attrs",
        new_callable=PropertyMock,
    ) as mock2:
        mock1.return_value = attrs.value
        mock2.return_value = attrs.value
        assert sampler.sample_independent(study, trial, "param-a", dist) != suggestion

    sampler = MOTPESampler(gamma=lambda _: 5, seed=0)
    attrs = MockSystemAttr()
    with patch.object(study._storage, "get_all_trials", return_value=past_trials), patch.object(
        study._storage, "set_trial_system_attr", side_effect=attrs.set_trial_system_attr
    ), patch.object(study._storage, "get_trial", return_value=trial), patch(
        "optuna.trial.Trial.system_attrs", new_callable=PropertyMock
    ) as mock1, patch(
        "optuna.trial.FrozenTrial.system_attrs",
        new_callable=PropertyMock,
    ) as mock2:
        mock1.return_value = attrs.value
        mock2.return_value = attrs.value
        assert sampler.sample_independent(study, trial, "param-a", dist) != suggestion

    sampler = MOTPESampler(weights_above=lambda n: np.zeros(n), seed=0)
    attrs = MockSystemAttr()
    with patch.object(study._storage, "get_all_trials", return_value=past_trials), patch.object(
        study._storage, "set_trial_system_attr", side_effect=attrs.set_trial_system_attr
    ), patch.object(study._storage, "get_trial", return_value=trial), patch(
        "optuna.trial.Trial.system_attrs", new_callable=PropertyMock
    ) as mock1, patch(
        "optuna.trial.FrozenTrial.system_attrs",
        new_callable=PropertyMock,
    ) as mock2:
        mock1.return_value = attrs.value
        mock2.return_value = attrs.value
        assert sampler.sample_independent(study, trial, "param-a", dist) != suggestion


def test_sample_independent_uniform_distributions() -> None:
    # Prepare sample from uniform distribution for cheking other distributions.
    study = optuna.create_study(directions=["minimize", "maximize"])
    random.seed(128)
    past_trials = [frozen_trial_factory(i, [random.random(), random.random()]) for i in range(16)]

    uni_dist = optuna.distributions.UniformDistribution(1.0, 100.0)
    trial = frozen_trial_factory(16, [0, 0])
    sampler = MOTPESampler(seed=0)
    attrs = MockSystemAttr()
    with patch.object(study._storage, "get_all_trials", return_value=past_trials), patch.object(
        study._storage, "set_trial_system_attr", side_effect=attrs.set_trial_system_attr
    ), patch.object(study._storage, "get_trial", return_value=trial), patch(
        "optuna.trial.Trial.system_attrs", new_callable=PropertyMock
    ) as mock1, patch(
        "optuna.trial.FrozenTrial.system_attrs",
        new_callable=PropertyMock,
    ) as mock2:
        mock1.return_value = attrs.value
        mock2.return_value = attrs.value
        uniform_suggestion = sampler.sample_independent(study, trial, "param-a", uni_dist)
    assert 1.0 <= uniform_suggestion < 100.0


def test_sample_independent_log_uniform_distributions() -> None:
    """Prepare sample from uniform distribution for cheking other distributions."""

    study = optuna.create_study(directions=["minimize", "maximize"])
    random.seed(128)

    uni_dist = optuna.distributions.UniformDistribution(1.0, 100.0)
    past_trials = [frozen_trial_factory(i, [random.random(), random.random()]) for i in range(16)]
    trial = frozen_trial_factory(16, [0, 0])
    sampler = MOTPESampler(seed=0)
    attrs = MockSystemAttr()
    with patch.object(study._storage, "get_all_trials", return_value=past_trials), patch.object(
        study._storage, "set_trial_system_attr", side_effect=attrs.set_trial_system_attr
    ), patch.object(study._storage, "get_trial", return_value=trial), patch(
        "optuna.trial.Trial.system_attrs", new_callable=PropertyMock
    ) as mock1, patch(
        "optuna.trial.FrozenTrial.system_attrs",
        new_callable=PropertyMock,
    ) as mock2:
        mock1.return_value = attrs.value
        mock2.return_value = attrs.value
        uniform_suggestion = sampler.sample_independent(study, trial, "param-a", uni_dist)

    # Test sample from log-uniform is different from uniform.
    log_dist = optuna.distributions.LogUniformDistribution(1.0, 100.0)
    past_trials = [
        frozen_trial_factory(i, [random.random(), random.random()], log_dist) for i in range(16)
    ]
    trial = frozen_trial_factory(16, [0, 0])
    sampler = MOTPESampler(seed=0)
    attrs = MockSystemAttr()
    with patch.object(study._storage, "get_all_trials", return_value=past_trials), patch.object(
        study._storage, "set_trial_system_attr", side_effect=attrs.set_trial_system_attr
    ), patch.object(study._storage, "get_trial", return_value=trial), patch(
        "optuna.trial.Trial.system_attrs", new_callable=PropertyMock
    ) as mock1, patch(
        "optuna.trial.FrozenTrial.system_attrs",
        new_callable=PropertyMock,
    ) as mock2:
        mock1.return_value = attrs.value
        mock2.return_value = attrs.value
        loguniform_suggestion = sampler.sample_independent(study, trial, "param-a", log_dist)
    assert 1.0 <= loguniform_suggestion < 100.0
    assert uniform_suggestion != loguniform_suggestion


def test_sample_independent_disrete_uniform_distributions() -> None:
    """Test samples from discrete have expected intervals."""

    study = optuna.create_study(directions=["minimize", "maximize"])
    random.seed(128)

    disc_dist = optuna.distributions.DiscreteUniformDistribution(1.0, 100.0, 0.1)

    def value_fn(idx: int) -> float:
        return int(random.random() * 1000) * 0.1

    past_trials = [
        frozen_trial_factory(
            i, [random.random(), random.random()], dist=disc_dist, value_fn=value_fn
        )
        for i in range(16)
    ]

    trial = frozen_trial_factory(16, [0, 0])
    sampler = MOTPESampler(seed=0)
    attrs = MockSystemAttr()
    with patch.object(study._storage, "get_all_trials", return_value=past_trials), patch.object(
        study._storage, "set_trial_system_attr", side_effect=attrs.set_trial_system_attr
    ), patch.object(study._storage, "get_trial", return_value=trial), patch(
        "optuna.trial.Trial.system_attrs", new_callable=PropertyMock
    ) as mock1, patch(
        "optuna.trial.FrozenTrial.system_attrs",
        new_callable=PropertyMock,
    ) as mock2:
        mock1.return_value = attrs.value
        mock2.return_value = attrs.value
        discrete_uniform_suggestion = sampler.sample_independent(
            study, trial, "param-a", disc_dist
        )
    assert 1.0 <= discrete_uniform_suggestion <= 100.0
    assert abs(int(discrete_uniform_suggestion * 10) - discrete_uniform_suggestion * 10) < 1e-3


def test_sample_independent_categorical_distributions() -> None:
    """Test samples are drawn from the specified category."""

    study = optuna.create_study(directions=["minimize", "maximize"])
    random.seed(128)

    categories = [i * 0.3 + 1.0 for i in range(330)]

    def cat_value_fn(idx: int) -> float:
        return categories[random.randint(0, len(categories) - 1)]

    cat_dist = optuna.distributions.CategoricalDistribution(categories)
    past_trials = [
        frozen_trial_factory(
            i, [random.random(), random.random()], dist=cat_dist, value_fn=cat_value_fn
        )
        for i in range(16)
    ]

    trial = frozen_trial_factory(16, [0, 0])
    sampler = MOTPESampler(seed=0)
    attrs = MockSystemAttr()
    with patch.object(study._storage, "get_all_trials", return_value=past_trials), patch.object(
        study._storage, "set_trial_system_attr", side_effect=attrs.set_trial_system_attr
    ), patch.object(study._storage, "get_trial", return_value=trial), patch(
        "optuna.trial.Trial.system_attrs", new_callable=PropertyMock
    ) as mock1, patch(
        "optuna.trial.FrozenTrial.system_attrs",
        new_callable=PropertyMock,
    ) as mock2:
        mock1.return_value = attrs.value
        mock2.return_value = attrs.value
        categorical_suggestion = sampler.sample_independent(study, trial, "param-a", cat_dist)
    assert categorical_suggestion in categories


def test_sample_int_uniform_distributions() -> None:
    """Test sampling from int distribution returns integer."""

    study = optuna.create_study(directions=["minimize", "maximize"])
    random.seed(128)

    def int_value_fn(idx: int) -> float:
        return random.randint(0, 100)

    int_dist = optuna.distributions.IntUniformDistribution(1, 100)
    past_trials = [
        frozen_trial_factory(
            i, [random.random(), random.random()], dist=int_dist, value_fn=int_value_fn
        )
        for i in range(16)
    ]

    trial = frozen_trial_factory(16, [0, 0])
    sampler = MOTPESampler(seed=0)
    attrs = MockSystemAttr()
    with patch.object(study._storage, "get_all_trials", return_value=past_trials), patch.object(
        study._storage, "set_trial_system_attr", side_effect=attrs.set_trial_system_attr
    ), patch.object(study._storage, "get_trial", return_value=trial), patch(
        "optuna.trial.Trial.system_attrs", new_callable=PropertyMock
    ) as mock1, patch(
        "optuna.trial.FrozenTrial.system_attrs",
        new_callable=PropertyMock,
    ) as mock2:
        mock1.return_value = attrs.value
        mock2.return_value = attrs.value
        int_suggestion = sampler.sample_independent(study, trial, "param-a", int_dist)
    assert 1 <= int_suggestion <= 100
    assert isinstance(int_suggestion, int)


@pytest.mark.parametrize(
    "state",
    [
        (optuna.trial.TrialState.FAIL,),
        (optuna.trial.TrialState.PRUNED,),
        (optuna.trial.TrialState.RUNNING,),
        (optuna.trial.TrialState.WAITING,),
    ],
)
def test_sample_independent_handle_unsuccessful_states(state: optuna.trial.TrialState) -> None:
    study = optuna.create_study(directions=["minimize", "maximize"])
    dist = optuna.distributions.UniformDistribution(1.0, 100.0)
    random.seed(128)

    # Prepare sampling result for later tests.
    past_trials = [frozen_trial_factory(i, [random.random(), random.random()]) for i in range(32)]

    trial = frozen_trial_factory(32, [0, 0])
    sampler = MOTPESampler(seed=0)
    attrs = MockSystemAttr()
    with patch.object(study._storage, "get_all_trials", return_value=past_trials), patch.object(
        study._storage, "set_trial_system_attr", side_effect=attrs.set_trial_system_attr
    ), patch.object(study._storage, "get_trial", return_value=trial), patch(
        "optuna.trial.Trial.system_attrs", new_callable=PropertyMock
    ) as mock1, patch(
        "optuna.trial.FrozenTrial.system_attrs",
        new_callable=PropertyMock,
    ) as mock2:
        mock1.return_value = attrs.value
        mock2.return_value = attrs.value
        all_success_suggestion = sampler.sample_independent(study, trial, "param-a", dist)

    # Test unsuccessful trials are handled differently.
    state_fn = build_state_fn(state)
    past_trials = [
        frozen_trial_factory(i, [random.random(), random.random()], state_fn=state_fn)
        for i in range(32)
    ]

    trial = frozen_trial_factory(32, [0, 0])
    sampler = MOTPESampler(seed=0)
    attrs = MockSystemAttr()
    with patch.object(study._storage, "get_all_trials", return_value=past_trials), patch.object(
        study._storage, "set_trial_system_attr", side_effect=attrs.set_trial_system_attr
    ), patch.object(study._storage, "get_trial", return_value=trial), patch(
        "optuna.trial.Trial.system_attrs", new_callable=PropertyMock
    ) as mock1, patch(
        "optuna.trial.FrozenTrial.system_attrs",
        new_callable=PropertyMock,
    ) as mock2:
        mock1.return_value = attrs.value
        mock2.return_value = attrs.value
        partial_unsuccessful_suggestion = sampler.sample_independent(study, trial, "param-a", dist)
    assert partial_unsuccessful_suggestion != all_success_suggestion


def test_sample_independent_ignored_states() -> None:
    """Tests FAIL, RUNNING, and WAITING states are equally."""
    study = optuna.create_study(directions=["minimize", "maximize"])
    dist = optuna.distributions.UniformDistribution(1.0, 100.0)

    suggestions = []
    for state in [
        optuna.trial.TrialState.FAIL,
        optuna.trial.TrialState.RUNNING,
        optuna.trial.TrialState.WAITING,
    ]:
        random.seed(128)
        state_fn = build_state_fn(state)
        past_trials = [
            frozen_trial_factory(i, [random.random(), random.random()], state_fn=state_fn)
            for i in range(32)
        ]
        trial = frozen_trial_factory(32, [0, 0])
        sampler = MOTPESampler(seed=0)
    attrs = MockSystemAttr()
    with patch.object(study._storage, "get_all_trials", return_value=past_trials), patch.object(
        study._storage, "set_trial_system_attr", side_effect=attrs.set_trial_system_attr
    ), patch.object(study._storage, "get_trial", return_value=trial), patch(
        "optuna.trial.Trial.system_attrs", new_callable=PropertyMock
    ) as mock1, patch(
        "optuna.trial.FrozenTrial.system_attrs",
        new_callable=PropertyMock,
    ) as mock2:
        mock1.return_value = attrs.value
        mock2.return_value = attrs.value
        suggestions.append(sampler.sample_independent(study, trial, "param-a", dist))

    assert len(set(suggestions)) == 1


def test_get_observation_pairs() -> None:
    def objective(trial: optuna.trial.Trial) -> Tuple[float, float]:
        trial.suggest_int("x", 5, 5)
        return 5.0, 5.0

    sampler = MOTPESampler(seed=0)
    study = optuna.create_study(directions=["minimize", "maximize"], sampler=sampler)
    study.optimize(objective, n_trials=5)

    assert multi_objective_sampler._get_observation_pairs(study, "x") == (
        [5.0, 5.0, 5.0, 5.0, 5.0],
        [[5.0, -5.0], [5.0, -5.0], [5.0, -5.0], [5.0, -5.0], [5.0, -5.0]],
    )
    assert multi_objective_sampler._get_observation_pairs(study, "y") == (
        [None, None, None, None, None],
        [[5.0, -5.0], [5.0, -5.0], [5.0, -5.0], [5.0, -5.0], [5.0, -5.0]],
    )


def test_calculate_nondomination_rank() -> None:
    # Single objective
    test_case = np.asarray([[10], [20], [20], [30]])
    ranks = list(multi_objective_sampler._calculate_nondomination_rank(test_case))
    assert ranks == [0, 1, 1, 2]

    # Two objectives
    test_case = np.asarray([[10, 30], [10, 10], [20, 20], [30, 10], [15, 15]])
    ranks = list(multi_objective_sampler._calculate_nondomination_rank(test_case))
    assert ranks == [1, 0, 2, 1, 1]

    # Three objectives
    test_case = np.asarray([[5, 5, 4], [5, 5, 5], [9, 9, 0], [5, 7, 5], [0, 0, 9], [0, 9, 9]])
    ranks = list(multi_objective_sampler._calculate_nondomination_rank(test_case))
    assert ranks == [0, 1, 0, 2, 0, 1]


def test_calculate_weights_below() -> None:
    sampler = MOTPESampler()

    # Two samples.
    weights_below = sampler._calculate_weights_below(
        np.array([[0.2, 0.5], [0.9, 0.4], [1, 1]]), np.array([0, 1])
    )
    assert len(weights_below) == 2
    assert weights_below[0] > weights_below[1]
    assert sum(weights_below) > 0

    # Two equally contributed samples.
    weights_below = sampler._calculate_weights_below(
        np.array([[0.2, 0.8], [0.8, 0.2], [1, 1]]), np.array([0, 1])
    )
    assert len(weights_below) == 2
    assert weights_below[0] == weights_below[1]
    assert sum(weights_below) > 0

    # Duplicated samples.
    weights_below = sampler._calculate_weights_below(
        np.array([[0.2, 0.8], [0.2, 0.8], [1, 1]]), np.array([0, 1])
    )
    assert len(weights_below) == 2
    assert weights_below[0] == weights_below[1]
    assert sum(weights_below) > 0

    # Three samples.
    weights_below = sampler._calculate_weights_below(
        np.array([[0.3, 0.3], [0.2, 0.8], [0.8, 0.2], [1, 1]]), np.array([0, 1, 2])
    )
    assert len(weights_below) == 3
    assert weights_below[0] > weights_below[1]
    assert weights_below[0] > weights_below[2]
    assert weights_below[1] == weights_below[2]
    assert sum(weights_below) > 0


def test_solve_hssp() -> None:
    sampler = MOTPESampler(seed=0)

    random.seed(128)

    # Two dimensions
    for i in range(8):
        subset_size = int(random.random() * i) + 1
        test_case = np.asarray([[random.random(), random.random()] for _ in range(8)])
        r = 1.1 * np.max(test_case, axis=0)
        truth = 0.0
        for subset in itertools.permutations(test_case, subset_size):
            truth = max(truth, sampler._compute_hypervolume(np.asarray(subset), r))
        indices = sampler._solve_hssp(test_case, np.arange(len(test_case)), subset_size, r)
        approx = sampler._compute_hypervolume(test_case[indices], r)
        assert approx / truth > 0.6321  # 1 - 1/e

    # Three dimensions
    for i in range(8):
        subset_size = int(random.random() * i) + 1
        test_case = np.asarray(
            [[random.random(), random.random(), random.random()] for _ in range(8)]
        )
        r = 1.1 * np.max(test_case, axis=0)
        truth = 0
        for subset in itertools.permutations(test_case, subset_size):
            truth = max(truth, sampler._compute_hypervolume(np.asarray(subset), r))
        indices = sampler._solve_hssp(test_case, np.arange(len(test_case)), subset_size, r)
        approx = sampler._compute_hypervolume(test_case[indices], r)
        assert approx / truth > 0.6321  # 1 - 1/e


def test_cache() -> None:
    n = 10
    sampler = MOTPESampler(seed=0, n_startup_trials=n)

    def objective(trial: optuna.trial.Trial) -> Tuple[float, float]:
        x = trial.suggest_float("x", 0, 5)

        if trial._trial_id == n:
            assert n in sampler._split_cache
            assert n in sampler._weights_below
        else:
            assert n not in sampler._split_cache
            assert n not in sampler._weights_below

        y = trial.suggest_float("y", 0, 3)
        v0 = 4 * x ** 2 + 4 * y ** 2
        v1 = (x - 5) ** 2 + (y - 5) ** 2
        return v0, v1

    study = optuna.create_study(directions=["minimize", "maximize"], sampler=sampler)

    assert n not in sampler._split_cache
    assert n not in sampler._weights_below

    study.optimize(objective, n_trials=n + 1)

    assert n not in sampler._split_cache
    assert n not in sampler._weights_below


def frozen_trial_factory(
    number: int,
    values: List[float],
    dist: optuna.distributions.BaseDistribution = optuna.distributions.UniformDistribution(
        1.0, 100.0
    ),
    value_fn: Optional[Callable[[int], Union[int, float]]] = None,
    state_fn: Callable[
        [int], optuna.trial.TrialState
    ] = lambda _: optuna.trial.TrialState.COMPLETE,
) -> optuna.trial.FrozenTrial:
    if value_fn is None:
        value = random.random() * 99.0 + 1.0
    else:
        value = value_fn(number)

    trial = optuna.trial.FrozenTrial(
        number=number,
        trial_id=number,
        state=optuna.trial.TrialState.COMPLETE,
        value=None,
        datetime_start=None,
        datetime_complete=None,
        params={"param-a": value},
        distributions={"param-a": dist},
        user_attrs={},
        system_attrs={},
        intermediate_values={},
        values=values,
    )
    return trial


def build_state_fn(state: optuna.trial.TrialState) -> Callable[[int], optuna.trial.TrialState]:
    def state_fn(idx: int) -> optuna.trial.TrialState:
        return [optuna.trial.TrialState.COMPLETE, state][idx % 2]

    return state_fn


def test_reseed_rng() -> None:
    sampler = MOTPESampler()
    original_seed = sampler._rng.seed

    with patch.object(
        sampler._mo_random_sampler, "reseed_rng", wraps=sampler._mo_random_sampler.reseed_rng
    ) as mock_object:
        sampler.reseed_rng()
        assert mock_object.call_count == 1
        assert original_seed != sampler._rng.seed


def test_call_after_trial_of_mo_random_sampler() -> None:
    sampler = MOTPESampler()
    study = optuna.create_study(sampler=sampler)
    with patch.object(
        sampler._mo_random_sampler, "after_trial", wraps=sampler._mo_random_sampler.after_trial
    ) as mock_object:
        study.optimize(lambda _: 1.0, n_trials=1)
        assert mock_object.call_count == 1
