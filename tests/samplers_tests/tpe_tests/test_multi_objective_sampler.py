import itertools
import random
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from unittest.mock import patch
from unittest.mock import PropertyMock

import numpy as np
import pytest

import optuna
from optuna.samplers import _tpe
from optuna.samplers import TPESampler


class MockSystemAttr:
    def __init__(self) -> None:
        self.value = {}  # type: Dict[str, dict]

    def set_trial_system_attr(self, _: int, key: str, value: dict) -> None:
        self.value[key] = value


def suggest(
    sampler: optuna.samplers.BaseSampler,
    study: optuna.Study,
    trial: optuna.trial.FrozenTrial,
    distribution: optuna.distributions.BaseDistribution,
    past_trials: List[optuna.trial.FrozenTrial],
) -> float:
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
        suggestion = sampler.sample_independent(study, trial, "param-a", distribution)
    return suggestion


def test_multi_objective_sample_independent_seed_fix() -> None:
    study = optuna.create_study(directions=["minimize", "maximize"])
    dist = optuna.distributions.FloatDistribution(1.0, 100.0)

    random.seed(128)
    past_trials = [frozen_trial_factory(i, [random.random(), random.random()]) for i in range(16)]
    # Prepare a trial and a sample for later checks.
    trial = frozen_trial_factory(16, [0, 0])
    sampler = TPESampler(seed=0)
    suggestion = suggest(sampler, study, trial, dist, past_trials)

    sampler = TPESampler(seed=0)
    assert suggest(sampler, study, trial, dist, past_trials) == suggestion

    sampler = TPESampler(seed=1)
    assert suggest(sampler, study, trial, dist, past_trials) != suggestion


def test_multi_objective_sample_independent_prior() -> None:
    study = optuna.create_study(directions=["minimize", "maximize"])
    dist = optuna.distributions.FloatDistribution(1.0, 100.0)

    random.seed(128)
    past_trials = [frozen_trial_factory(i, [random.random(), random.random()]) for i in range(16)]

    # Prepare a trial and a sample for later checks.
    trial = frozen_trial_factory(16, [0, 0])
    sampler = TPESampler(seed=0)
    suggestion = suggest(sampler, study, trial, dist, past_trials)

    sampler = TPESampler(consider_prior=False, seed=0)
    assert suggest(sampler, study, trial, dist, past_trials) != suggestion

    sampler = TPESampler(prior_weight=0.5, seed=0)
    assert suggest(sampler, study, trial, dist, past_trials) != suggestion


def test_multi_objective_sample_independent_n_startup_trial() -> None:
    study = optuna.create_study(directions=["minimize", "maximize"])
    dist = optuna.distributions.FloatDistribution(1.0, 100.0)
    random.seed(128)
    past_trials = [frozen_trial_factory(i, [random.random(), random.random()]) for i in range(16)]
    trial = frozen_trial_factory(16, [0, 0])

    def _suggest_and_return_call_count(
        sampler: optuna.samplers.BaseSampler,
        past_trials: List[optuna.trial.FrozenTrial],
    ) -> int:
        attrs = MockSystemAttr()
        with patch.object(
            study._storage, "get_all_trials", return_value=past_trials
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
        return sample_method.call_count

    sampler = TPESampler(n_startup_trials=16, seed=0)
    assert _suggest_and_return_call_count(sampler, past_trials[:-1]) == 1

    sampler = TPESampler(n_startup_trials=16, seed=0)
    assert _suggest_and_return_call_count(sampler, past_trials) == 0


def test_multi_objective_sample_independent_misc_arguments() -> None:
    study = optuna.create_study(directions=["minimize", "maximize"])
    dist = optuna.distributions.FloatDistribution(1.0, 100.0)
    random.seed(128)
    past_trials = [frozen_trial_factory(i, [random.random(), random.random()]) for i in range(32)]

    # Prepare a trial and a sample for later checks.
    trial = frozen_trial_factory(16, [0, 0])
    sampler = TPESampler(seed=0)
    suggestion = suggest(sampler, study, trial, dist, past_trials)

    # Test misc. parameters.
    sampler = TPESampler(n_ei_candidates=13, seed=0)
    assert suggest(sampler, study, trial, dist, past_trials) != suggestion

    sampler = TPESampler(gamma=lambda _: 1, seed=0)
    assert suggest(sampler, study, trial, dist, past_trials) != suggestion


@pytest.mark.parametrize("log, step", [(False, None), (True, None), (False, 0.1)])
def test_multi_objective_sample_independent_float_distributions(
    log: bool, step: Optional[float]
) -> None:
    # Prepare sample from float distribution for checking other distributions.
    study = optuna.create_study(directions=["minimize", "maximize"])
    random.seed(128)
    float_dist = optuna.distributions.FloatDistribution(1.0, 100.0, log=log, step=step)

    if float_dist.step:
        value_fn: Optional[Callable[[int], float]] = (
            lambda number: int(random.random() * 1000) * 0.1
        )
    else:
        value_fn = None

    past_trials = [
        frozen_trial_factory(
            i, [random.random(), random.random()], dist=float_dist, value_fn=value_fn
        )
        for i in range(16)
    ]

    trial = frozen_trial_factory(16, [0, 0])
    sampler = TPESampler(seed=0)
    float_suggestion = suggest(sampler, study, trial, float_dist, past_trials)
    assert 1.0 <= float_suggestion < 100.0

    if float_dist.step == 0.1:
        assert abs(int(float_suggestion * 10) - float_suggestion * 10) < 1e-3

    # Test sample is different when `float_dist.log` is True or float_dist.step != 1.0.
    random.seed(128)
    dist = optuna.distributions.FloatDistribution(1.0, 100.0)
    past_trials = [frozen_trial_factory(i, [random.random(), random.random()]) for i in range(16)]
    trial = frozen_trial_factory(16, [0, 0])
    sampler = TPESampler(seed=0)
    suggestion = suggest(sampler, study, trial, dist, past_trials)
    if float_dist.log or float_dist.step == 0.1:
        assert float_suggestion != suggestion
    else:
        assert float_suggestion == suggestion


def test_multi_objective_sample_independent_categorical_distributions() -> None:
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
    sampler = TPESampler(seed=0)
    categorical_suggestion = suggest(sampler, study, trial, cat_dist, past_trials)
    assert categorical_suggestion in categories


@pytest.mark.parametrize(
    "log, step",
    [
        (False, 1),
        (True, 1),
        (False, 2),
    ],
)
def test_multi_objective_sample_int_distributions(log: bool, step: int) -> None:
    """Test sampling from int distribution returns integer."""

    study = optuna.create_study(directions=["minimize", "maximize"])
    random.seed(128)

    def int_value_fn(idx: int) -> float:
        return random.randint(1, 99)

    int_dist = optuna.distributions.IntDistribution(1, 99, log, step)
    past_trials = [
        frozen_trial_factory(
            i, [random.random(), random.random()], dist=int_dist, value_fn=int_value_fn
        )
        for i in range(16)
    ]

    trial = frozen_trial_factory(16, [0, 0])
    sampler = TPESampler(seed=0)
    int_suggestion = suggest(sampler, study, trial, int_dist, past_trials)
    assert 1 <= int_suggestion <= 99
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
def test_multi_objective_sample_independent_handle_unsuccessful_states(
    state: optuna.trial.TrialState,
) -> None:
    study = optuna.create_study(directions=["minimize", "maximize"])
    dist = optuna.distributions.FloatDistribution(1.0, 100.0)
    random.seed(128)

    # Prepare sampling result for later tests.
    past_trials = [frozen_trial_factory(i, [random.random(), random.random()]) for i in range(32)]

    trial = frozen_trial_factory(32, [0, 0])
    sampler = TPESampler(seed=0)
    all_success_suggestion = suggest(sampler, study, trial, dist, past_trials)

    # Test unsuccessful trials are handled differently.
    state_fn = build_state_fn(state)
    past_trials = [
        frozen_trial_factory(i, [random.random(), random.random()], state_fn=state_fn)
        for i in range(32)
    ]

    trial = frozen_trial_factory(32, [0, 0])
    sampler = TPESampler(seed=0)
    partial_unsuccessful_suggestion = suggest(sampler, study, trial, dist, past_trials)
    assert partial_unsuccessful_suggestion != all_success_suggestion


def test_multi_objective_sample_independent_ignored_states() -> None:
    """Tests FAIL, RUNNING, and WAITING states are equally."""
    study = optuna.create_study(directions=["minimize", "maximize"])
    dist = optuna.distributions.FloatDistribution(1.0, 100.0)

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
        sampler = TPESampler(seed=0)
        suggestions.append(suggest(sampler, study, trial, dist, past_trials))

    assert len(set(suggestions)) == 1


@pytest.mark.parametrize("int_value", [-5, 5, 0])
@pytest.mark.parametrize(
    "categorical_value", [1, 0.0, "A", None, True, float("inf"), float("nan")]
)
@pytest.mark.parametrize("objective_value", [-5.0, 5.0, 0.0, -float("inf"), float("inf")])
@pytest.mark.parametrize("multivariate", [True, False])
@pytest.mark.parametrize("constant_liar", [True, False])
@pytest.mark.filterwarnings("ignore::optuna.exceptions.ExperimentalWarning")
def test_multi_objective_get_observation_pairs(
    int_value: int,
    categorical_value: optuna.distributions.CategoricalChoiceType,
    objective_value: float,
    multivariate: bool,
    constant_liar: bool,
) -> None:
    def objective(trial: optuna.trial.Trial) -> Tuple[float, float]:
        trial.suggest_int("x", int_value, int_value)
        trial.suggest_categorical("y", [categorical_value])
        return objective_value, objective_value

    sampler = TPESampler(seed=0, multivariate=multivariate, constant_liar=constant_liar)
    study = optuna.create_study(directions=["minimize", "maximize"], sampler=sampler)
    study.optimize(objective, n_trials=2)
    study.add_trial(
        optuna.create_trial(
            state=optuna.trial.TrialState.RUNNING,
            params={"x": int_value, "y": categorical_value},
            distributions={
                "x": optuna.distributions.IntDistribution(int_value, int_value),
                "y": optuna.distributions.CategoricalDistribution([categorical_value]),
            },
        )
    )

    assert _tpe.sampler._get_observation_pairs(study, ["x"], constant_liar) == (
        {"x": [int_value, int_value]},
        [(-float("inf"), [objective_value, -objective_value]) for _ in range(2)],
        None,
    )
    assert _tpe.sampler._get_observation_pairs(study, ["y"], constant_liar) == (
        {"y": [0, 0]},
        [(-float("inf"), [objective_value, -objective_value]) for _ in range(2)],
        None,
    )
    assert _tpe.sampler._get_observation_pairs(study, ["x", "y"], constant_liar) == (
        {"x": [int_value, int_value], "y": [0, 0]},
        [(-float("inf"), [objective_value, -objective_value]) for _ in range(2)],
        None,
    )
    assert _tpe.sampler._get_observation_pairs(study, ["z"], constant_liar) == (
        {"z": [None, None]},
        [(-float("inf"), [objective_value, -objective_value]) for _ in range(2)],
        None,
    )


@pytest.mark.parametrize("constraint_value", [-2, 2])
def test_multi_objective_get_observation_pairs_constrained(constraint_value: int) -> None:
    def objective(trial: optuna.trial.Trial) -> Tuple[float, float]:
        trial.suggest_int("x", 5, 5)
        trial.set_user_attr("constraint", (constraint_value, -1))
        return 5.0, 5.0

    sampler = TPESampler(constraints_func=lambda trial: trial.user_attrs["constraint"], seed=0)
    study = optuna.create_study(directions=["minimize", "maximize"], sampler=sampler)
    study.optimize(objective, n_trials=5)

    violations = [max(0, constraint_value) for _ in range(5)]
    assert _tpe.sampler._get_observation_pairs(study, ["x"], constraints_enabled=True) == (
        {"x": [5.0, 5.0, 5.0, 5.0, 5.0]},
        [(-float("inf"), [5.0, -5.0]) for _ in range(5)],
        violations,
    )
    assert _tpe.sampler._get_observation_pairs(study, ["y"], constraints_enabled=True) == (
        {"y": [None, None, None, None, None]},
        [(-float("inf"), [5.0, -5.0]) for _ in range(5)],
        violations,
    )


def test_multi_objective_split_observation_pairs() -> None:
    indices_below, indices_above = _tpe.sampler._split_observation_pairs(
        [
            (-float("inf"), [-2.0, -1.0]),
            (-float("inf"), [3.0, 3.0]),
            (-float("inf"), [0.0, 1.0]),
            (-float("inf"), [-1.0, 0.0]),
        ],
        2,
        None,
    )
    assert list(indices_below) == [0, 3]
    assert list(indices_above) == [1, 2]


def test_multi_objective_split_observation_pairs_with_all_indices_below() -> None:
    indices_below, indices_above = _tpe.sampler._split_observation_pairs(
        [
            (-float("inf"), [1.0, 1.0]),
        ],
        1,
        None,
    )
    assert list(indices_below) == [0]
    assert list(indices_above) == []


def test_calculate_nondomination_rank() -> None:
    # Single objective
    test_case = np.asarray([[10], [20], [20], [30]])
    ranks = list(_tpe.sampler._calculate_nondomination_rank(test_case))
    assert ranks == [0, 1, 1, 2]

    # Two objectives
    test_case = np.asarray([[10, 30], [10, 10], [20, 20], [30, 10], [15, 15]])
    ranks = list(_tpe.sampler._calculate_nondomination_rank(test_case))
    assert ranks == [1, 0, 2, 1, 1]

    # Three objectives
    test_case = np.asarray([[5, 5, 4], [5, 5, 5], [9, 9, 0], [5, 7, 5], [0, 0, 9], [0, 9, 9]])
    ranks = list(_tpe.sampler._calculate_nondomination_rank(test_case))
    assert ranks == [0, 1, 0, 2, 0, 1]

    # The negative values are included.
    test_case = np.asarray(
        [[-5, -5, -4], [-5, -5, 5], [-9, -9, 0], [5, 7, 5], [0, 0, -9], [0, -9, 9]]
    )
    ranks = list(_tpe.sampler._calculate_nondomination_rank(test_case))
    assert ranks == [0, 1, 0, 2, 0, 1]

    # The +inf is included.
    test_case = np.asarray(
        [[1, 1], [1, float("inf")], [float("inf"), 1], [float("inf"), float("inf")]]
    )
    ranks = list(_tpe.sampler._calculate_nondomination_rank(test_case))
    assert ranks == [0, 1, 1, 2]

    # The -inf is included.
    test_case = np.asarray(
        [[1, 1], [1, -float("inf")], [-float("inf"), 1], [-float("inf"), -float("inf")]]
    )
    ranks = list(_tpe.sampler._calculate_nondomination_rank(test_case))
    assert ranks == [2, 1, 1, 0]


def test_calculate_weights_below_for_multi_objective() -> None:
    # No sample.
    weights_below = _tpe.sampler._calculate_weights_below_for_multi_objective(
        [(0, [0.2, 0.5]), (0, [0.9, 0.4]), (0, [1, 1])],
        np.array([], np.int64),
        None,
    )
    assert len(weights_below) == 0

    # One sample.
    weights_below = _tpe.sampler._calculate_weights_below_for_multi_objective(
        [(0, [0.2, 0.5]), (0, [0.9, 0.4]), (0, [1, 1])],
        np.array([0]),
        None,
    )
    assert len(weights_below) == 1
    assert sum(weights_below) > 0

    # Two samples.
    weights_below = _tpe.sampler._calculate_weights_below_for_multi_objective(
        [(0, [0.2, 0.5]), (0, [0.9, 0.4]), (0, [1, 1])],
        np.array([0, 1]),
        None,
    )
    assert len(weights_below) == 2
    assert weights_below[0] > weights_below[1]
    assert sum(weights_below) > 0

    # Two equally contributed samples.
    weights_below = _tpe.sampler._calculate_weights_below_for_multi_objective(
        [(0, [0.2, 0.8]), (0, [0.8, 0.2]), (0, [1, 1])],
        np.array([0, 1]),
        None,
    )
    assert len(weights_below) == 2
    assert weights_below[0] == weights_below[1]
    assert sum(weights_below) > 0

    # Duplicated samples.
    weights_below = _tpe.sampler._calculate_weights_below_for_multi_objective(
        [(0, [0.2, 0.8]), (0, [0.2, 0.8]), (0, [1, 1])],
        np.array([0, 1]),
        None,
    )
    assert len(weights_below) == 2
    assert weights_below[0] == weights_below[1]
    assert sum(weights_below) > 0

    # Three samples.
    weights_below = _tpe.sampler._calculate_weights_below_for_multi_objective(
        [(0, [0.3, 0.3]), (0, [0.2, 0.8]), (0, [0.8, 0.2]), (0, [1, 1])],
        np.array([0, 1, 2]),
        None,
    )
    assert len(weights_below) == 3
    assert weights_below[0] > weights_below[1]
    assert weights_below[0] > weights_below[2]
    assert weights_below[1] == weights_below[2]
    assert sum(weights_below) > 0

    # Zero/negative objective values.
    weights_below = _tpe.sampler._calculate_weights_below_for_multi_objective(
        [(0, [-0.3, -0.3]), (0, [0.0, -0.8]), (0, [-0.8, 0.0]), (0, [1, 1])],
        np.array([0, 1, 2]),
        None,
    )
    assert len(weights_below) == 3
    assert weights_below[0] > weights_below[1]
    assert weights_below[0] > weights_below[2]
    assert np.isclose(weights_below[1], weights_below[2])
    assert sum(weights_below) > 0

    # +/-inf objective values.
    weights_below = _tpe.sampler._calculate_weights_below_for_multi_objective(
        [
            (0, [-float("inf"), -float("inf")]),
            (0, [0.0, -float("inf")]),
            (0, [-float("inf"), 0.0]),
            (0, [float("inf"), float("inf")]),
        ],
        np.array([0, 1, 2]),
        None,
    )
    assert len(weights_below) == 3
    assert all([np.isnan(w) for w in weights_below])

    # Three samples with two infeasible trials.
    weights_below = _tpe.sampler._calculate_weights_below_for_multi_objective(
        [(0, [0.3, 0.3]), (0, [0.2, 0.8]), (0, [0.8, 0.2]), (0, [1, 1])],
        np.array([0, 1, 2]),
        [2, 8, 0],
    )
    assert len(weights_below) == 3
    assert weights_below[0] == _tpe.sampler.EPS
    assert weights_below[1] == _tpe.sampler.EPS
    assert weights_below[2] > 0


def _compute_hssp_truth_and_approx(test_case: np.ndarray, subset_size: int) -> Tuple[float, float]:
    r = 1.1 * np.max(test_case, axis=0)
    truth = 0.0
    for subset in itertools.permutations(test_case, subset_size):
        truth = max(truth, _tpe.sampler._compute_hypervolume(np.asarray(subset), r))
    indices = _tpe.sampler._solve_hssp(test_case, np.arange(len(test_case)), subset_size, r)
    approx = _tpe.sampler._compute_hypervolume(test_case[indices], r)
    return truth, approx


@pytest.mark.parametrize("dim", [2, 3])
def test_solve_hssp(dim: int) -> None:
    random.seed(128)

    for i in range(8):
        subset_size = int(random.random() * i) + 1
        test_case = np.asarray([[random.random() for _ in range(dim)] for _ in range(8)])
        truth, approx = _compute_hssp_truth_and_approx(test_case, subset_size)
        assert approx / truth > 0.6321  # 1 - 1/e


def test_solve_hssp_infinite_loss() -> None:
    random.seed(128)

    subset_size = int(random.random() * 4) + 1
    test_case = np.asarray([[random.random() for _ in range(2)] for _ in range(8)])
    test_case = np.vstack([test_case, [float("inf") for _ in range(2)]])
    truth, approx = _compute_hssp_truth_and_approx(test_case, subset_size)
    assert np.isinf(truth)
    assert np.isinf(approx)

    test_case = np.asarray([[random.random() for _ in range(3)] for _ in range(8)])
    test_case = np.vstack([test_case, [float("inf") for _ in range(3)]])
    truth, approx = _compute_hssp_truth_and_approx(test_case, subset_size)
    assert truth == 0
    assert np.isnan(approx)

    test_case = np.asarray([[random.random() for _ in range(2)] for _ in range(8)])
    test_case = np.vstack([test_case, [-float("inf") for _ in range(2)]])
    truth, approx = _compute_hssp_truth_and_approx(test_case, subset_size)
    assert np.isinf(truth)
    assert np.isinf(approx)

    test_case = np.asarray([[random.random() for _ in range(3)] for _ in range(8)])
    test_case = np.vstack([test_case, [-float("inf") for _ in range(3)]])
    truth, approx = _compute_hssp_truth_and_approx(test_case, subset_size)
    assert np.isinf(truth)
    assert np.isinf(approx)


def frozen_trial_factory(
    number: int,
    values: List[float],
    dist: optuna.distributions.BaseDistribution = optuna.distributions.FloatDistribution(
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
