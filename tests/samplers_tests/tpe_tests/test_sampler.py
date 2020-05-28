import random
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Union
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np
import pytest

import optuna
from optuna.samplers import tpe
from optuna.samplers import TPESampler
from optuna import TrialPruned

if optuna.type_checking.TYPE_CHECKING:
    from optuna.trial import Trial  # NOQA


@pytest.mark.parametrize("use_hyperband", [False, True])
def test_hyperopt_parameters(use_hyperband):
    # type: (bool) -> None

    sampler = TPESampler(**TPESampler.hyperopt_parameters())
    study = optuna.create_study(
        sampler=sampler, pruner=optuna.pruners.HyperbandPruner() if use_hyperband else None
    )
    study.optimize(lambda t: t.suggest_uniform("x", 10, 20), n_trials=50)


def test_sample_relative() -> None:
    sampler = TPESampler()
    # Study and frozen-trial are not supposed to be accessed.
    study = Mock(spec=[])
    frozen_trial = Mock(spec=[])
    assert sampler.sample_relative(study, frozen_trial, {}) == {}


def test_infer_relative_search_space() -> None:
    sampler = TPESampler()
    # Study and frozen-trial are not supposed to be accessed.
    study = Mock(spec=[])
    frozen_trial = Mock(spec=[])
    assert sampler.infer_relative_search_space(study, frozen_trial) == {}


def test_sample_independent_seed_fix() -> None:
    study = optuna.create_study()
    dist = optuna.distributions.UniformDistribution(1.0, 100.0)
    past_trials = [frozen_trial_factory(i, dist=dist) for i in range(1, 8)]

    # Prepare a trial and a sample for later checks.
    trial = frozen_trial_factory(8)
    sampler = TPESampler(n_startup_trials=5, seed=0)
    with patch("optuna.Study.get_trials", return_value=past_trials):
        suggestion = sampler.sample_independent(study, trial, "param-a", dist)

    sampler = TPESampler(n_startup_trials=5, seed=0)
    with patch("optuna.Study.get_trials", return_value=past_trials):
        assert sampler.sample_independent(study, trial, "param-a", dist) == suggestion

    sampler = TPESampler(n_startup_trials=5, seed=1)
    with patch("optuna.Study.get_trials", return_value=past_trials):
        assert sampler.sample_independent(study, trial, "param-a", dist) != suggestion


def test_sample_independent_prior() -> None:
    study = optuna.create_study()
    dist = optuna.distributions.UniformDistribution(1.0, 100.0)
    past_trials = [frozen_trial_factory(i, dist=dist) for i in range(1, 8)]

    # Prepare a trial and a sample for later checks.
    trial = frozen_trial_factory(8)
    sampler = TPESampler(n_startup_trials=5, seed=0)
    with patch("optuna.Study.get_trials", return_value=past_trials):
        suggestion = sampler.sample_independent(study, trial, "param-a", dist)

    sampler = TPESampler(consider_prior=False, n_startup_trials=5, seed=0)
    with patch("optuna.Study.get_trials", return_value=past_trials):
        assert sampler.sample_independent(study, trial, "param-a", dist) != suggestion

    sampler = TPESampler(prior_weight=0.5, n_startup_trials=5, seed=0)
    with patch("optuna.Study.get_trials", return_value=past_trials):
        assert sampler.sample_independent(study, trial, "param-a", dist) != suggestion


def test_sample_independent_n_startup_trial() -> None:
    study = optuna.create_study()
    dist = optuna.distributions.UniformDistribution(1.0, 100.0)
    past_trials = [frozen_trial_factory(i, dist=dist) for i in range(1, 8)]

    trial = frozen_trial_factory(8)
    sampler = TPESampler(n_startup_trials=5, seed=0)
    with patch("optuna.Study.get_trials", return_value=past_trials[:4]):
        with patch.object(
            optuna.samplers.random.RandomSampler, "sample_independent", return_value=1.0
        ) as sample_method:
            sampler.sample_independent(study, trial, "param-a", dist)
    assert sample_method.call_count == 1
    sampler = TPESampler(n_startup_trials=5, seed=0)
    with patch("optuna.Study.get_trials", return_value=past_trials[:5]):
        with patch.object(
            optuna.samplers.random.RandomSampler, "sample_independent", return_value=1.0
        ) as sample_method:
            sampler.sample_independent(study, trial, "param-a", dist)
    assert sample_method.call_count == 0


def test_sample_independent_misc_arguments() -> None:
    study = optuna.create_study()
    dist = optuna.distributions.UniformDistribution(1.0, 100.0)
    past_trials = [frozen_trial_factory(i, dist=dist) for i in range(1, 8)]

    # Prepare a trial and a sample for later checks.
    trial = frozen_trial_factory(8)
    sampler = TPESampler(n_startup_trials=5, seed=0)
    with patch("optuna.Study.get_trials", return_value=past_trials):
        suggestion = sampler.sample_independent(study, trial, "param-a", dist)

    # Test misc. parameters.
    sampler = TPESampler(n_ei_candidates=13, n_startup_trials=5, seed=0)
    with patch("optuna.Study.get_trials", return_value=past_trials):
        assert sampler.sample_independent(study, trial, "param-a", dist) != suggestion

    sampler = TPESampler(gamma=lambda _: 5, n_startup_trials=5, seed=0)
    with patch("optuna.Study.get_trials", return_value=past_trials):
        assert sampler.sample_independent(study, trial, "param-a", dist) != suggestion

    sampler = TPESampler(
        weights=lambda i: np.asarray([i * 0.11 for i in range(7)]), n_startup_trials=5, seed=0
    )
    with patch("optuna.Study.get_trials", return_value=past_trials):
        assert sampler.sample_independent(study, trial, "param-a", dist) != suggestion


def test_sample_independent_uniform_distributions() -> None:
    study = optuna.create_study()

    # Prepare sample from uniform distribution for cheking other distributions.
    uni_dist = optuna.distributions.UniformDistribution(1.0, 100.0)
    past_trials = [frozen_trial_factory(i, dist=uni_dist) for i in range(1, 8)]
    trial = frozen_trial_factory(8)
    sampler = TPESampler(n_startup_trials=5, seed=0)
    with patch("optuna.Study.get_trials", return_value=past_trials):
        uniform_suggestion = sampler.sample_independent(study, trial, "param-a", uni_dist)
    assert 1.0 <= uniform_suggestion < 100.0


def test_sample_independent_log_uniform_distributions() -> None:
    """Prepare sample from uniform distribution for cheking other distributions."""
    study = optuna.create_study()

    uni_dist = optuna.distributions.UniformDistribution(1.0, 100.0)
    past_trials = [frozen_trial_factory(i, dist=uni_dist) for i in range(1, 8)]
    trial = frozen_trial_factory(8)
    sampler = TPESampler(n_startup_trials=5, seed=0)
    with patch("optuna.Study.get_trials", return_value=past_trials):
        uniform_suggestion = sampler.sample_independent(study, trial, "param-a", uni_dist)

    # Test sample from log-uniform is different from uniform.
    log_dist = optuna.distributions.LogUniformDistribution(1.0, 100.0)
    past_trials = [frozen_trial_factory(i, dist=log_dist) for i in range(1, 8)]
    trial = frozen_trial_factory(8)
    sampler = TPESampler(n_startup_trials=5, seed=0)
    with patch("optuna.Study.get_trials", return_value=past_trials):
        loguniform_suggestion = sampler.sample_independent(study, trial, "param-a", log_dist)
    assert 1.0 <= loguniform_suggestion < 100.0
    assert uniform_suggestion != loguniform_suggestion


def test_sample_independent_disrete_uniform_distributions() -> None:
    """Test samples from discrete have expected intervals."""

    study = optuna.create_study()
    disc_dist = optuna.distributions.DiscreteUniformDistribution(1.0, 100.0, 0.1)

    def value_fn(idx: int) -> float:
        random.seed(idx)
        return int(random.random() * 1000) * 0.1

    past_trials = [frozen_trial_factory(i, dist=disc_dist, value_fn=value_fn) for i in range(1, 8)]
    trial = frozen_trial_factory(8)
    sampler = TPESampler(n_startup_trials=5, seed=0)
    with patch("optuna.Study.get_trials", return_value=past_trials):
        discrete_uniform_suggestion = sampler.sample_independent(
            study, trial, "param-a", disc_dist
        )
    assert 1.0 <= discrete_uniform_suggestion <= 100.0
    assert abs(int(discrete_uniform_suggestion * 10) - discrete_uniform_suggestion * 10) < 1e-3


def test_sample_independent_categorical_distributions() -> None:
    """Test samples are drawn from the specified category."""

    study = optuna.create_study()
    categories = [i * 0.3 + 1.0 for i in range(330)]

    def cat_value_fn(idx: int) -> float:
        random.seed(idx)
        return categories[random.randint(0, len(categories) - 1)]

    cat_dist = optuna.distributions.CategoricalDistribution(categories)
    past_trials = [
        frozen_trial_factory(i, dist=cat_dist, value_fn=cat_value_fn) for i in range(1, 8)
    ]
    trial = frozen_trial_factory(8)
    sampler = TPESampler(n_startup_trials=5, seed=0)
    with patch("optuna.Study.get_trials", return_value=past_trials):
        categorical_suggestion = sampler.sample_independent(study, trial, "param-a", cat_dist)
    assert categorical_suggestion in categories


def test_sample_int_uniform_distributions() -> None:
    """Test sampling from int distribution returns integer."""

    study = optuna.create_study()

    def int_value_fn(idx: int) -> float:
        random.seed(idx)
        return random.randint(0, 100)

    int_dist = optuna.distributions.IntUniformDistribution(1, 100)
    past_trials = [
        frozen_trial_factory(i, dist=int_dist, value_fn=int_value_fn) for i in range(1, 8)
    ]
    trial = frozen_trial_factory(8)
    sampler = TPESampler(n_startup_trials=5, seed=0)
    with patch("optuna.Study.get_trials", return_value=past_trials):
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
    study = optuna.create_study()
    dist = optuna.distributions.UniformDistribution(1.0, 100.0)

    # Prepare sampling result for later tests.
    past_trials = [frozen_trial_factory(i, dist=dist) for i in range(1, 30)]
    trial = frozen_trial_factory(30)
    sampler = TPESampler(n_startup_trials=5, seed=0)
    with patch("optuna.Study.get_trials", return_value=past_trials):
        all_success_suggestion = sampler.sample_independent(study, trial, "param-a", dist)

    # Test unsuccessful trials are handled differently.
    state_fn = build_state_fn(state)
    past_trials = [frozen_trial_factory(i, dist=dist, state_fn=state_fn) for i in range(1, 30)]
    trial = frozen_trial_factory(30)
    sampler = TPESampler(n_startup_trials=5, seed=0)
    with patch("optuna.Study.get_trials", return_value=past_trials):
        partial_unsuccessful_suggestion = sampler.sample_independent(study, trial, "param-a", dist)
    assert partial_unsuccessful_suggestion != all_success_suggestion


def test_sample_independent_ignored_states() -> None:
    """Tests FAIL, RUNNING, and WAITING states are equally."""
    study = optuna.create_study()
    dist = optuna.distributions.UniformDistribution(1.0, 100.0)

    suggestions = []
    for state in [
        optuna.trial.TrialState.FAIL,
        optuna.trial.TrialState.RUNNING,
        optuna.trial.TrialState.WAITING,
    ]:
        state_fn = build_state_fn(state)
        past_trials = [frozen_trial_factory(i, dist=dist, state_fn=state_fn) for i in range(1, 30)]
        trial = frozen_trial_factory(30)
        sampler = TPESampler(n_startup_trials=5, seed=0)
        with patch("optuna.Study.get_trials", return_value=past_trials):
            suggestions.append(sampler.sample_independent(study, trial, "param-a", dist))

    assert len(set(suggestions)) == 1


def test_sample_independent_pruned_state() -> None:
    """Tests PRUNED state is treated differently from both FAIL and COMPLETE."""
    study = optuna.create_study()
    dist = optuna.distributions.UniformDistribution(1.0, 100.0)

    suggestions = []
    for state in [
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.FAIL,
        optuna.trial.TrialState.PRUNED,
    ]:
        state_fn = build_state_fn(state)
        past_trials = [frozen_trial_factory(i, dist=dist, state_fn=state_fn) for i in range(1, 30)]
        trial = frozen_trial_factory(30)
        sampler = TPESampler(n_startup_trials=5, seed=0)
        with patch("optuna.Study.get_trials", return_value=past_trials):
            suggestions.append(sampler.sample_independent(study, trial, "param-a", dist))

    assert len(set(suggestions)) == 3


def test_get_observation_pairs():
    # type: () -> None

    def objective(trial):
        # type: (Trial) -> float

        x = trial.suggest_int("x", 5, 5)
        if trial.number == 0:
            return x
        elif trial.number == 1:
            trial.report(1, 4)
            trial.report(2, 7)
            raise TrialPruned()
        elif trial.number == 2:
            trial.report(float("nan"), 3)
            raise TrialPruned()
        elif trial.number == 3:
            raise TrialPruned()
        else:
            raise RuntimeError()

    # Test direction=minimize.
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=5, catch=(RuntimeError,))
    trial_number = study._storage.create_new_trial(study._study_id)  # Create a running trial.
    trial = study._storage.get_trial(trial_number)

    assert tpe.sampler._get_observation_pairs(study, "x", trial) == (
        [5.0, 5.0, 5.0, 5.0],
        [
            (-float("inf"), 5.0),  # COMPLETE
            (-7, 2),  # PRUNED (with intermediate values)
            (-3, float("inf")),  # PRUNED (with a NaN intermediate value; it's treated as infinity)
            (float("inf"), 0.0),  # PRUNED (without intermediate values)
        ],
    )
    assert tpe.sampler._get_observation_pairs(study, "y", trial) == (
        [None, None, None, None],
        [
            (-float("inf"), 5.0),  # COMPLETE
            (-7, 2),  # PRUNED (with intermediate values)
            (-3, float("inf")),  # PRUNED (with a NaN intermediate value; it's treated as infinity)
            (float("inf"), 0.0),  # PRUNED (without intermediate values)
        ],
    )

    # Test direction=maximize.
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=4)
    study._storage.create_new_trial(study._study_id)  # Create a running trial.

    assert tpe.sampler._get_observation_pairs(study, "x", trial) == (
        [5.0, 5.0, 5.0, 5.0],
        [
            (-float("inf"), -5.0),  # COMPLETE
            (-7, -2),  # PRUNED (with intermediate values)
            (-3, float("inf")),  # PRUNED (with a NaN intermediate value; it's treated as infinity)
            (float("inf"), 0.0),  # PRUNED (without intermediate values)
        ],
    )
    assert tpe.sampler._get_observation_pairs(study, "y", trial) == (
        [None, None, None, None],
        [
            (-float("inf"), -5.0),  # COMPLETE
            (-7, -2),  # PRUNED (with intermediate values)
            (-3, float("inf")),  # PRUNED (with a NaN intermediate value; it's treated as infinity)
            (float("inf"), 0.0),  # PRUNED (without intermediate values)
        ],
    )


def frozen_trial_factory(
    idx: int,
    dist: optuna.distributions.BaseDistribution = optuna.distributions.UniformDistribution(
        1.0, 100.0
    ),
    state_fn: Callable[
        [int], optuna.trial.TrialState
    ] = lambda _: optuna.trial.TrialState.COMPLETE,
    value_fn: Optional[Callable[[int], Union[int, float]]] = None,
    target_fn: Callable[[float], float] = lambda val: (val - 20.0) ** 2,
    interm_val_fn: Callable[[int], Dict[int, float]] = lambda _: {},
) -> optuna.trial.FrozenTrial:
    if value_fn is None:
        random.seed(idx)
        value = random.random() * 99.0 + 1.0
    else:
        value = value_fn(idx)
    return optuna.trial.FrozenTrial(
        number=idx,
        state=state_fn(idx),
        value=target_fn(value),
        datetime_start=None,
        datetime_complete=None,
        params={"param-a": value},
        distributions={"param-a": dist},
        user_attrs={},
        system_attrs={},
        intermediate_values=interm_val_fn(idx),
        trial_id=idx + 123,
    )


def build_state_fn(state: optuna.trial.TrialState) -> Callable[[int], optuna.trial.TrialState]:
    def state_fn(idx: int) -> optuna.trial.TrialState:
        return [optuna.trial.TrialState.COMPLETE, state][idx % 2]

    return state_fn


def test_reseed_rng() -> None:
    sampler = TPESampler()
    original_seed = sampler._rng.seed

    with patch.object(
        sampler._random_sampler, "reseed_rng", wraps=sampler._random_sampler.reseed_rng
    ) as mock_object:
        sampler.reseed_rng()
        assert mock_object.call_count == 1
        assert original_seed != sampler._rng.seed
