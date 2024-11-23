from __future__ import annotations

from collections.abc import Callable
import random
from unittest.mock import patch
from unittest.mock import PropertyMock

import numpy as np
import pytest

import optuna
from optuna.samplers import _tpe
from optuna.samplers import TPESampler


class MockSystemAttr:
    def __init__(self) -> None:
        self.value: dict[str, dict] = {}

    def set_trial_system_attr(self, _: int, key: str, value: dict) -> None:
        self.value[key] = value


def suggest(
    sampler: optuna.samplers.BaseSampler,
    study: optuna.Study,
    trial: optuna.trial.FrozenTrial,
    distribution: optuna.distributions.BaseDistribution,
    past_trials: list[optuna.trial.FrozenTrial],
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
        past_trials: list[optuna.trial.FrozenTrial],
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
        study._thread_local.cached_all_trials = None
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
    log: bool, step: float | None
) -> None:
    # Prepare sample from float distribution for checking other distributions.
    study = optuna.create_study(directions=["minimize", "maximize"])
    random.seed(128)
    float_dist = optuna.distributions.FloatDistribution(1.0, 100.0, log=log, step=step)

    if float_dist.step:
        value_fn: Callable[[int], float] | None = lambda number: int(random.random() * 1000) * 0.1
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
    study._thread_local.cached_all_trials = None

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


@pytest.mark.parametrize("direction0", ["minimize", "maximize"])
@pytest.mark.parametrize("direction1", ["minimize", "maximize"])
def test_split_complete_trials_multi_objective(direction0: str, direction1: str) -> None:
    study = optuna.create_study(directions=(direction0, direction1))

    for values in ([-2.0, -1.0], [3.0, 3.0], [0.0, 1.0], [-1.0, 0.0]):
        value0, value1 = values
        if direction0 == "maximize":
            value0 = -value0
        if direction1 == "maximize":
            value1 = -value1
        study.add_trial(
            optuna.create_trial(
                state=optuna.trial.TrialState.COMPLETE,
                values=(value0, value1),
                params={"x": 0},
                distributions={"x": optuna.distributions.FloatDistribution(-1.0, 1.0)},
            )
        )

    below_trials, above_trials = _tpe.sampler._split_complete_trials_multi_objective(
        study.trials,
        study,
        2,
    )
    assert [trial.number for trial in below_trials] == [0, 3]
    assert [trial.number for trial in above_trials] == [1, 2]


def test_split_complete_trials_multi_objective_empty() -> None:
    study = optuna.create_study(directions=("minimize", "minimize"))
    assert _tpe.sampler._split_complete_trials_multi_objective([], study, 0) == ([], [])


def test_calculate_weights_below_for_multi_objective() -> None:
    # No sample.
    study = optuna.create_study(directions=["minimize", "minimize"])
    weights_below = _tpe.sampler._calculate_weights_below_for_multi_objective(study, [], None)
    assert len(weights_below) == 0

    # One sample.
    study = optuna.create_study(directions=["minimize", "minimize"])
    trial0 = optuna.create_trial(values=[0.2, 0.5])
    study.add_trials([trial0])
    weights_below = _tpe.sampler._calculate_weights_below_for_multi_objective(
        study, [trial0], None
    )
    assert len(weights_below) == 1
    assert sum(weights_below) > 0

    # Two samples.
    study = optuna.create_study(directions=["minimize", "minimize"])
    trial0 = optuna.create_trial(values=[0.2, 0.5])
    trial1 = optuna.create_trial(values=[0.9, 0.4])
    study.add_trials([trial0, trial1])
    weights_below = _tpe.sampler._calculate_weights_below_for_multi_objective(
        study,
        [trial0, trial1],
        None,
    )
    assert len(weights_below) == 2
    assert weights_below[0] > weights_below[1]
    assert sum(weights_below) > 0

    # Two equally contributed samples.
    study = optuna.create_study(directions=["minimize", "minimize"])
    trial0 = optuna.create_trial(values=[0.2, 0.8])
    trial1 = optuna.create_trial(values=[0.8, 0.2])
    study.add_trials([trial0, trial1])
    weights_below = _tpe.sampler._calculate_weights_below_for_multi_objective(
        study,
        [trial0, trial1],
        None,
    )
    assert len(weights_below) == 2
    assert weights_below[0] == weights_below[1]
    assert sum(weights_below) > 0

    # Duplicated samples.
    study = optuna.create_study(directions=["minimize", "minimize"])
    trial0 = optuna.create_trial(values=[0.2, 0.8])
    trial1 = optuna.create_trial(values=[0.2, 0.8])
    study.add_trials([trial0, trial1])
    weights_below = _tpe.sampler._calculate_weights_below_for_multi_objective(
        study,
        [trial0, trial1],
        None,
    )
    assert len(weights_below) == 2
    assert weights_below[0] == weights_below[1]
    assert sum(weights_below) > 0

    # Three samples.
    study = optuna.create_study(directions=["minimize", "minimize"])
    trial0 = optuna.create_trial(values=[0.3, 0.3])
    trial1 = optuna.create_trial(values=[0.2, 0.8])
    trial2 = optuna.create_trial(values=[0.8, 0.2])
    study.add_trials([trial0, trial1, trial2])
    weights_below = _tpe.sampler._calculate_weights_below_for_multi_objective(
        study,
        [trial0, trial1, trial2],
        None,
    )
    assert len(weights_below) == 3
    assert weights_below[0] > weights_below[1]
    assert weights_below[0] > weights_below[2]
    assert weights_below[1] == weights_below[2]
    assert sum(weights_below) > 0

    # Zero/negative objective values.
    study = optuna.create_study(directions=["minimize", "minimize"])
    trial0 = optuna.create_trial(values=[-0.3, -0.3])
    trial1 = optuna.create_trial(values=[0.0, -0.8])
    trial2 = optuna.create_trial(values=[-0.8, 0.0])
    study.add_trials([trial0, trial1, trial2])
    weights_below = _tpe.sampler._calculate_weights_below_for_multi_objective(
        study,
        [trial0, trial1, trial2],
        None,
    )
    assert len(weights_below) == 3
    assert weights_below[0] > weights_below[1]
    assert weights_below[0] > weights_below[2]
    assert np.isclose(weights_below[1], weights_below[2])
    assert sum(weights_below) > 0

    # +/-inf objective values.
    study = optuna.create_study(directions=["minimize", "minimize"])
    trial0 = optuna.create_trial(values=[-float("inf"), -float("inf")])
    trial1 = optuna.create_trial(values=[0.0, -float("inf")])
    trial2 = optuna.create_trial(values=[-float("inf"), 0.0])
    study.add_trials([trial0, trial1, trial2])
    weights_below = _tpe.sampler._calculate_weights_below_for_multi_objective(
        study,
        [trial0, trial1, trial2],
        None,
    )
    assert len(weights_below) == 3
    assert not any([np.isnan(w) for w in weights_below])
    assert sum(weights_below) > 0

    # Three samples with two infeasible trials.
    study = optuna.create_study(directions=["minimize", "minimize"])
    trial0 = optuna.create_trial(values=[0.3, 0.3], system_attrs={"constraints": 2})
    trial1 = optuna.create_trial(values=[0.2, 0.8], system_attrs={"constraints": 8})
    trial2 = optuna.create_trial(values=[0.8, 0.2], system_attrs={"constraints": 0})
    study.add_trials([trial0, trial1, trial2])
    weights_below = _tpe.sampler._calculate_weights_below_for_multi_objective(
        study,
        [trial0, trial1, trial2],
        lambda trial: [trial.system_attrs["constraints"]],
    )
    assert len(weights_below) == 3
    assert weights_below[0] == _tpe.sampler.EPS
    assert weights_below[1] == _tpe.sampler.EPS
    assert weights_below[2] > 0


def frozen_trial_factory(
    number: int,
    values: list[float],
    dist: optuna.distributions.BaseDistribution = optuna.distributions.FloatDistribution(
        1.0, 100.0
    ),
    value_fn: Callable[[int], int | float] | None = None,
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
