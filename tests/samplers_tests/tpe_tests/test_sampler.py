import random
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Union
from unittest.mock import Mock
from unittest.mock import patch
import warnings

import numpy as np
import pytest

import optuna
from optuna import distributions
from optuna import TrialPruned
from optuna.samplers import _tpe
from optuna.samplers import TPESampler
from optuna.trial import Trial


@pytest.mark.parametrize("use_hyperband", [False, True])
def test_hyperopt_parameters(use_hyperband: bool) -> None:

    sampler = TPESampler(**TPESampler.hyperopt_parameters())
    study = optuna.create_study(
        sampler=sampler, pruner=optuna.pruners.HyperbandPruner() if use_hyperband else None
    )
    study.optimize(lambda t: t.suggest_float("x", 10, 20), n_trials=50)


def test_multivariate_experimental_warning() -> None:
    with pytest.warns(optuna.exceptions.ExperimentalWarning):
        optuna.samplers.TPESampler(multivariate=True)


def test_infer_relative_search_space() -> None:
    sampler = TPESampler()
    search_space = {
        "a": distributions.UniformDistribution(1.0, 100.0),
        "b": distributions.LogUniformDistribution(1.0, 100.0),
        "c": distributions.DiscreteUniformDistribution(1.0, 100.0, 3.0),
        "d": distributions.IntUniformDistribution(1, 100),
        "e": distributions.IntUniformDistribution(0, 100, step=2),
        "f": distributions.IntLogUniformDistribution(1, 100),
        "g": distributions.CategoricalDistribution(["x", "y", "z"]),
    }

    def obj(t: Trial) -> float:
        t.suggest_float("a", 1.0, 100.0)
        t.suggest_float("b", 1.0, 100.0, log=True)
        t.suggest_float("c", 1.0, 100.0, step=3.0)
        t.suggest_int("d", 1, 100)
        t.suggest_int("e", 0, 100, step=2)
        t.suggest_int("f", 1, 100, log=True)
        t.suggest_categorical("g", ["x", "y", "z"])
        return 0.0

    # Study and frozen-trial are not supposed to be accessed.
    study1 = Mock(spec=[])
    frozen_trial = Mock(spec=[])
    assert sampler.infer_relative_search_space(study1, frozen_trial) == {}

    study2 = optuna.create_study(sampler=sampler)
    study2.optimize(obj, n_trials=1)
    assert sampler.infer_relative_search_space(study2, study2.best_trial) == {}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = TPESampler(multivariate=True)
    study3 = optuna.create_study(sampler=sampler)
    study3.optimize(obj, n_trials=1)
    assert sampler.infer_relative_search_space(study3, study3.best_trial) == search_space


@pytest.mark.parametrize("multivariate", [False, True])
def test_sample_relative_empty_input(multivariate: bool) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = TPESampler(multivariate=multivariate)
    # A frozen-trial is not supposed to be accessed.
    study = optuna.create_study()
    frozen_trial = Mock(spec=[])
    assert sampler.sample_relative(study, frozen_trial, {}) == {}


def test_sample_relative_seed_fix() -> None:
    study = optuna.create_study()
    dist = optuna.distributions.UniformDistribution(1.0, 100.0)
    past_trials = [frozen_trial_factory(i, dist=dist) for i in range(1, 8)]

    # Prepare a trial and a sample for later checks.
    trial = frozen_trial_factory(8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = TPESampler(n_startup_trials=5, seed=0, multivariate=True)
    with patch.object(study._storage, "get_all_trials", return_value=past_trials):
        suggestion = sampler.sample_relative(study, trial, {"param-a": dist})

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = TPESampler(n_startup_trials=5, seed=0, multivariate=True)
    with patch.object(study._storage, "get_all_trials", return_value=past_trials):
        assert sampler.sample_relative(study, trial, {"param-a": dist}) == suggestion

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = TPESampler(n_startup_trials=5, seed=1, multivariate=True)
    with patch.object(study._storage, "get_all_trials", return_value=past_trials):
        assert sampler.sample_relative(study, trial, {"param-a": dist}) != suggestion


def test_sample_relative_prior() -> None:
    study = optuna.create_study()
    dist = optuna.distributions.UniformDistribution(1.0, 100.0)
    past_trials = [frozen_trial_factory(i, dist=dist) for i in range(1, 8)]

    # Prepare a trial and a sample for later checks.
    trial = frozen_trial_factory(8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = TPESampler(n_startup_trials=5, seed=0, multivariate=True)
    with patch.object(study._storage, "get_all_trials", return_value=past_trials):
        suggestion = sampler.sample_relative(study, trial, {"param-a": dist})

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = TPESampler(consider_prior=False, n_startup_trials=5, seed=0, multivariate=True)
    with patch.object(study._storage, "get_all_trials", return_value=past_trials):
        assert sampler.sample_relative(study, trial, {"param-a": dist}) != suggestion

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = TPESampler(prior_weight=0.2, n_startup_trials=5, seed=0, multivariate=True)
    with patch.object(study._storage, "get_all_trials", return_value=past_trials):
        assert sampler.sample_relative(study, trial, {"param-a": dist}) != suggestion


def test_sample_relative_n_startup_trial() -> None:
    study = optuna.create_study()
    dist = optuna.distributions.UniformDistribution(1.0, 100.0)
    past_trials = [frozen_trial_factory(i, dist=dist) for i in range(1, 8)]

    trial = frozen_trial_factory(8)
    # sample_relative returns {} for only 4 observations.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = TPESampler(n_startup_trials=5, seed=0, multivariate=True)
    with patch.object(study._storage, "get_all_trials", return_value=past_trials[:4]):
        assert sampler.sample_relative(study, trial, {"param-a": dist}) == {}
    # sample_relative returns some value for only 7 observations.
    with patch.object(study._storage, "get_all_trials", return_value=past_trials):
        assert "param-a" in sampler.sample_relative(study, trial, {"param-a": dist}).keys()


def test_sample_relative_misc_arguments() -> None:
    study = optuna.create_study()
    dist = optuna.distributions.UniformDistribution(1.0, 100.0)
    past_trials = [frozen_trial_factory(i, dist=dist) for i in range(1, 40)]

    # Prepare a trial and a sample for later checks.
    trial = frozen_trial_factory(40)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = TPESampler(n_startup_trials=5, seed=0, multivariate=True)
    with patch.object(study._storage, "get_all_trials", return_value=past_trials):
        suggestion = sampler.sample_relative(study, trial, {"param-a": dist})

    # Test misc. parameters.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = TPESampler(n_ei_candidates=13, n_startup_trials=5, seed=0, multivariate=True)
    with patch.object(study._storage, "get_all_trials", return_value=past_trials):
        assert sampler.sample_relative(study, trial, {"param-a": dist}) != suggestion

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = TPESampler(gamma=lambda _: 5, n_startup_trials=5, seed=0, multivariate=True)
    with patch.object(study._storage, "get_all_trials", return_value=past_trials):
        assert sampler.sample_relative(study, trial, {"param-a": dist}) != suggestion

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = TPESampler(
            weights=lambda n: np.asarray([i ** 2 + 1 for i in range(n)]),
            n_startup_trials=5,
            seed=0,
            multivariate=True,
        )
    with patch.object(study._storage, "get_all_trials", return_value=past_trials):
        assert sampler.sample_relative(study, trial, {"param-a": dist}) != suggestion


def test_sample_relative_uniform_distributions() -> None:
    study = optuna.create_study()

    # Prepare sample from uniform distribution for cheking other distributions.
    uni_dist = optuna.distributions.UniformDistribution(1.0, 100.0)
    past_trials = [frozen_trial_factory(i, dist=uni_dist) for i in range(1, 8)]
    trial = frozen_trial_factory(8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = TPESampler(n_startup_trials=5, seed=0, multivariate=True)
    with patch.object(study._storage, "get_all_trials", return_value=past_trials):
        uniform_suggestion = sampler.sample_relative(study, trial, {"param-a": uni_dist})
    assert 1.0 <= uniform_suggestion["param-a"] < 100.0


def test_sample_relative_log_uniform_distributions() -> None:
    """Prepare sample from uniform distribution for cheking other distributions."""

    study = optuna.create_study()

    uni_dist = optuna.distributions.UniformDistribution(1.0, 100.0)
    past_trials = [frozen_trial_factory(i, dist=uni_dist) for i in range(1, 8)]
    trial = frozen_trial_factory(8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = TPESampler(n_startup_trials=5, seed=0, multivariate=True)
    with patch.object(study._storage, "get_all_trials", return_value=past_trials):
        uniform_suggestion = sampler.sample_relative(study, trial, {"param-a": uni_dist})

    # Test sample from log-uniform is different from uniform.
    log_dist = optuna.distributions.LogUniformDistribution(1.0, 100.0)
    past_trials = [frozen_trial_factory(i, dist=log_dist) for i in range(1, 8)]
    trial = frozen_trial_factory(8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = TPESampler(n_startup_trials=5, seed=0, multivariate=True)
    with patch.object(study._storage, "get_all_trials", return_value=past_trials):
        loguniform_suggestion = sampler.sample_relative(study, trial, {"param-a": log_dist})
    assert 1.0 <= loguniform_suggestion["param-a"] < 100.0
    assert uniform_suggestion["param-a"] != loguniform_suggestion["param-a"]


def test_sample_relative_disrete_uniform_distributions() -> None:
    """Test samples from discrete have expected intervals."""

    study = optuna.create_study()
    disc_dist = optuna.distributions.DiscreteUniformDistribution(1.0, 100.0, 0.1)

    def value_fn(idx: int) -> float:
        random.seed(idx)
        return int(random.random() * 1000) * 0.1

    past_trials = [frozen_trial_factory(i, dist=disc_dist, value_fn=value_fn) for i in range(1, 8)]
    trial = frozen_trial_factory(8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = TPESampler(n_startup_trials=5, seed=0, multivariate=True)
    with patch.object(study._storage, "get_all_trials", return_value=past_trials):
        discrete_uniform_suggestion = sampler.sample_relative(study, trial, {"param-a": disc_dist})
    assert 1.0 <= discrete_uniform_suggestion["param-a"] <= 100.0
    np.testing.assert_almost_equal(
        int(discrete_uniform_suggestion["param-a"] * 10),
        discrete_uniform_suggestion["param-a"] * 10,
    )


def test_sample_relative_categorical_distributions() -> None:
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = TPESampler(n_startup_trials=5, seed=0, multivariate=True)
    with patch.object(study._storage, "get_all_trials", return_value=past_trials):
        categorical_suggestion = sampler.sample_relative(study, trial, {"param-a": cat_dist})
    assert categorical_suggestion["param-a"] in categories


@pytest.mark.parametrize("step", [1, 2])
def test_sample_relative_int_uniform_distributions(step: int) -> None:
    """Test sampling from int distribution returns integer."""

    study = optuna.create_study()

    def int_value_fn(idx: int) -> float:
        random.seed(idx)
        return step * random.randint(0, 100 // step)

    int_dist = optuna.distributions.IntUniformDistribution(0, 100, step=step)
    past_trials = [
        frozen_trial_factory(i, dist=int_dist, value_fn=int_value_fn) for i in range(1, 8)
    ]
    trial = frozen_trial_factory(8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = TPESampler(n_startup_trials=5, seed=0, multivariate=True)
    with patch.object(study._storage, "get_all_trials", return_value=past_trials):
        int_suggestion = sampler.sample_relative(study, trial, {"param-a": int_dist})
    assert 1 <= int_suggestion["param-a"] <= 100
    assert isinstance(int_suggestion["param-a"], int)
    assert int_suggestion["param-a"] % step == 0


def test_sample_relative_int_loguniform_distributions() -> None:
    """Test sampling from int distribution returns integer."""

    study = optuna.create_study()

    def int_value_fn(idx: int) -> float:
        random.seed(idx)
        return random.randint(0, 100)

    intlog_dist = optuna.distributions.IntLogUniformDistribution(1, 100)
    past_trials = [
        frozen_trial_factory(i, dist=intlog_dist, value_fn=int_value_fn) for i in range(1, 8)
    ]
    trial = frozen_trial_factory(8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = TPESampler(n_startup_trials=5, seed=0, multivariate=True)
    with patch.object(study._storage, "get_all_trials", return_value=past_trials):
        intlog_suggestion = sampler.sample_relative(study, trial, {"param-a": intlog_dist})
    assert 1 <= intlog_suggestion["param-a"] <= 100
    assert isinstance(intlog_suggestion["param-a"], int)


@pytest.mark.parametrize(
    "state",
    [
        optuna.trial.TrialState.FAIL,
        optuna.trial.TrialState.PRUNED,
        optuna.trial.TrialState.RUNNING,
        optuna.trial.TrialState.WAITING,
    ],
)
def test_sample_relative_handle_unsuccessful_states(
    state: optuna.trial.TrialState,
) -> None:
    dist = optuna.distributions.UniformDistribution(1.0, 100.0)

    # Prepare sampling result for later tests.
    study = optuna.create_study()
    for i in range(1, 100):
        trial = frozen_trial_factory(i, dist=dist)
        study._storage.create_new_trial(study._study_id, template_trial=trial)
    trial = frozen_trial_factory(100)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = TPESampler(n_startup_trials=5, seed=0, multivariate=True)
    all_success_suggestion = sampler.sample_relative(study, trial, {"param-a": dist})

    # Test unsuccessful trials are handled differently.
    study = optuna.create_study()
    state_fn = build_state_fn(state)
    for i in range(1, 100):
        trial = frozen_trial_factory(i, dist=dist, state_fn=state_fn)
        study._storage.create_new_trial(study._study_id, template_trial=trial)
    trial = frozen_trial_factory(100)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = TPESampler(n_startup_trials=5, seed=0, multivariate=True)
    partial_unsuccessful_suggestion = sampler.sample_relative(study, trial, {"param-a": dist})

    assert partial_unsuccessful_suggestion != all_success_suggestion


def test_sample_relative_ignored_states() -> None:
    """Tests FAIL, RUNNING, and WAITING states are equally."""

    dist = optuna.distributions.UniformDistribution(1.0, 100.0)

    suggestions = []
    for state in [
        optuna.trial.TrialState.FAIL,
        optuna.trial.TrialState.RUNNING,
        optuna.trial.TrialState.WAITING,
    ]:
        study = optuna.create_study()
        state_fn = build_state_fn(state)
        for i in range(1, 30):
            trial = frozen_trial_factory(i, dist=dist, state_fn=state_fn)
            study._storage.create_new_trial(study._study_id, template_trial=trial)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
            sampler = TPESampler(n_startup_trials=5, seed=0, multivariate=True)
        suggestions.append(sampler.sample_relative(study, trial, {"param-a": dist})["param-a"])

    assert len(set(suggestions)) == 1


def test_sample_relative_pruned_state() -> None:
    """Tests PRUNED state is treated differently from both FAIL and COMPLETE."""

    dist = optuna.distributions.UniformDistribution(1.0, 100.0)

    suggestions = []
    for state in [
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.FAIL,
        optuna.trial.TrialState.PRUNED,
    ]:
        study = optuna.create_study()
        state_fn = build_state_fn(state)
        for i in range(1, 40):
            trial = frozen_trial_factory(i, dist=dist, state_fn=state_fn)
            study._storage.create_new_trial(study._study_id, template_trial=trial)
        trial = frozen_trial_factory(40)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
            sampler = TPESampler(n_startup_trials=5, seed=0, multivariate=True)
        suggestions.append(sampler.sample_relative(study, trial, {"param-a": dist})["param-a"])

    assert len(set(suggestions)) == 3


def test_sample_independent_seed_fix() -> None:
    study = optuna.create_study()
    dist = optuna.distributions.UniformDistribution(1.0, 100.0)
    past_trials = [frozen_trial_factory(i, dist=dist) for i in range(1, 8)]

    # Prepare a trial and a sample for later checks.
    trial = frozen_trial_factory(8)
    sampler = TPESampler(n_startup_trials=5, seed=0)
    with patch.object(study._storage, "get_all_trials", return_value=past_trials):
        suggestion = sampler.sample_independent(study, trial, "param-a", dist)

    sampler = TPESampler(n_startup_trials=5, seed=0)
    with patch.object(study._storage, "get_all_trials", return_value=past_trials):
        assert sampler.sample_independent(study, trial, "param-a", dist) == suggestion

    sampler = TPESampler(n_startup_trials=5, seed=1)
    with patch.object(study._storage, "get_all_trials", return_value=past_trials):
        assert sampler.sample_independent(study, trial, "param-a", dist) != suggestion


def test_sample_independent_prior() -> None:
    study = optuna.create_study()
    dist = optuna.distributions.UniformDistribution(1.0, 100.0)
    past_trials = [frozen_trial_factory(i, dist=dist) for i in range(1, 8)]

    # Prepare a trial and a sample for later checks.
    trial = frozen_trial_factory(8)
    sampler = TPESampler(n_startup_trials=5, seed=0)
    with patch.object(study._storage, "get_all_trials", return_value=past_trials):
        suggestion = sampler.sample_independent(study, trial, "param-a", dist)

    sampler = TPESampler(consider_prior=False, n_startup_trials=5, seed=0)
    with patch.object(study._storage, "get_all_trials", return_value=past_trials):
        assert sampler.sample_independent(study, trial, "param-a", dist) != suggestion

    sampler = TPESampler(prior_weight=0.5, n_startup_trials=5, seed=0)
    with patch.object(study._storage, "get_all_trials", return_value=past_trials):
        assert sampler.sample_independent(study, trial, "param-a", dist) != suggestion


def test_sample_independent_n_startup_trial() -> None:
    study = optuna.create_study()
    dist = optuna.distributions.UniformDistribution(1.0, 100.0)
    past_trials = [frozen_trial_factory(i, dist=dist) for i in range(1, 8)]

    trial = frozen_trial_factory(8)
    sampler = TPESampler(n_startup_trials=5, seed=0)
    with patch.object(study._storage, "get_all_trials", return_value=past_trials[:4]):
        with patch.object(
            optuna.samplers.RandomSampler, "sample_independent", return_value=1.0
        ) as sample_method:
            sampler.sample_independent(study, trial, "param-a", dist)
    assert sample_method.call_count == 1
    sampler = TPESampler(n_startup_trials=5, seed=0)
    with patch.object(study._storage, "get_all_trials", return_value=past_trials):
        with patch.object(
            optuna.samplers.RandomSampler, "sample_independent", return_value=1.0
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
    with patch.object(study._storage, "get_all_trials", return_value=past_trials):
        suggestion = sampler.sample_independent(study, trial, "param-a", dist)

    # Test misc. parameters.
    sampler = TPESampler(n_ei_candidates=13, n_startup_trials=5, seed=0)
    with patch.object(study._storage, "get_all_trials", return_value=past_trials):
        assert sampler.sample_independent(study, trial, "param-a", dist) != suggestion

    sampler = TPESampler(gamma=lambda _: 5, n_startup_trials=5, seed=0)
    with patch.object(study._storage, "get_all_trials", return_value=past_trials):
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
    with patch.object(study._storage, "get_all_trials", return_value=past_trials):
        uniform_suggestion = sampler.sample_independent(study, trial, "param-a", uni_dist)
    assert 1.0 <= uniform_suggestion < 100.0


def test_sample_independent_log_uniform_distributions() -> None:
    """Prepare sample from uniform distribution for cheking other distributions."""

    study = optuna.create_study()

    uni_dist = optuna.distributions.UniformDistribution(1.0, 100.0)
    past_trials = [frozen_trial_factory(i, dist=uni_dist) for i in range(1, 8)]
    trial = frozen_trial_factory(8)
    sampler = TPESampler(n_startup_trials=5, seed=0)
    with patch.object(study._storage, "get_all_trials", return_value=past_trials):
        uniform_suggestion = sampler.sample_independent(study, trial, "param-a", uni_dist)

    # Test sample from log-uniform is different from uniform.
    log_dist = optuna.distributions.LogUniformDistribution(1.0, 100.0)
    past_trials = [frozen_trial_factory(i, dist=log_dist) for i in range(1, 8)]
    trial = frozen_trial_factory(8)
    sampler = TPESampler(n_startup_trials=5, seed=0)
    with patch.object(study._storage, "get_all_trials", return_value=past_trials):
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
    with patch.object(study._storage, "get_all_trials", return_value=past_trials):
        categorical_suggestion = sampler.sample_independent(study, trial, "param-a", cat_dist)
    assert categorical_suggestion in categories


def test_sample_independent_int_uniform_distributions() -> None:
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
    with patch.object(study._storage, "get_all_trials", return_value=past_trials):
        int_suggestion = sampler.sample_independent(study, trial, "param-a", int_dist)
    assert 1 <= int_suggestion <= 100
    assert isinstance(int_suggestion, int)


def test_sample_independent_int_loguniform_distributions() -> None:
    """Test sampling from int distribution returns integer."""

    study = optuna.create_study()

    def int_value_fn(idx: int) -> float:
        random.seed(idx)
        return random.randint(0, 100)

    intlog_dist = optuna.distributions.IntLogUniformDistribution(1, 100)
    past_trials = [
        frozen_trial_factory(i, dist=intlog_dist, value_fn=int_value_fn) for i in range(1, 8)
    ]
    trial = frozen_trial_factory(8)
    sampler = TPESampler(n_startup_trials=5, seed=0)
    with patch.object(study._storage, "get_all_trials", return_value=past_trials):
        intlog_suggestion = sampler.sample_independent(study, trial, "param-a", intlog_dist)
    assert 1 <= intlog_suggestion <= 100
    assert isinstance(intlog_suggestion, int)


@pytest.mark.parametrize(
    "state",
    [
        optuna.trial.TrialState.FAIL,
        optuna.trial.TrialState.PRUNED,
        optuna.trial.TrialState.RUNNING,
        optuna.trial.TrialState.WAITING,
    ],
)
def test_sample_independent_handle_unsuccessful_states(state: optuna.trial.TrialState) -> None:
    dist = optuna.distributions.UniformDistribution(1.0, 100.0)

    # Prepare sampling result for later tests.
    study = optuna.create_study()
    for i in range(1, 30):
        trial = frozen_trial_factory(i, dist=dist)
        study._storage.create_new_trial(study._study_id, template_trial=trial)
    trial = frozen_trial_factory(30)
    sampler = TPESampler(n_startup_trials=5, seed=0)
    all_success_suggestion = sampler.sample_independent(study, trial, "param-a", dist)

    # Test unsuccessful trials are handled differently.
    state_fn = build_state_fn(state)
    study = optuna.create_study()
    for i in range(1, 30):
        trial = frozen_trial_factory(i, dist=dist, state_fn=state_fn)
        study._storage.create_new_trial(study._study_id, template_trial=trial)
    trial = frozen_trial_factory(30)
    sampler = TPESampler(n_startup_trials=5, seed=0)
    partial_unsuccessful_suggestion = sampler.sample_independent(study, trial, "param-a", dist)

    assert partial_unsuccessful_suggestion != all_success_suggestion


def test_sample_independent_ignored_states() -> None:
    """Tests FAIL, RUNNING, and WAITING states are equally."""

    dist = optuna.distributions.UniformDistribution(1.0, 100.0)

    suggestions = []
    for state in [
        optuna.trial.TrialState.FAIL,
        optuna.trial.TrialState.RUNNING,
        optuna.trial.TrialState.WAITING,
    ]:
        study = optuna.create_study()
        state_fn = build_state_fn(state)
        for i in range(1, 30):
            trial = frozen_trial_factory(i, dist=dist, state_fn=state_fn)
            study._storage.create_new_trial(study._study_id, template_trial=trial)
        trial = frozen_trial_factory(30)
        sampler = TPESampler(n_startup_trials=5, seed=0)
        suggestions.append(sampler.sample_independent(study, trial, "param-a", dist))

    assert len(set(suggestions)) == 1


def test_sample_independent_pruned_state() -> None:
    """Tests PRUNED state is treated differently from both FAIL and COMPLETE."""

    dist = optuna.distributions.UniformDistribution(1.0, 100.0)

    suggestions = []
    for state in [
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.FAIL,
        optuna.trial.TrialState.PRUNED,
    ]:
        study = optuna.create_study()
        state_fn = build_state_fn(state)
        for i in range(1, 30):
            trial = frozen_trial_factory(i, dist=dist, state_fn=state_fn)
            study._storage.create_new_trial(study._study_id, template_trial=trial)
        trial = frozen_trial_factory(30)
        sampler = TPESampler(n_startup_trials=5, seed=0)
        suggestions.append(sampler.sample_independent(study, trial, "param-a", dist))

    assert len(set(suggestions)) == 3


def test_get_observation_pairs() -> None:
    def objective(trial: Trial) -> float:

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

    assert _tpe.sampler._get_observation_pairs(study, "x") == (
        [5.0, 5.0, 5.0, 5.0],
        [
            (-float("inf"), 5.0),  # COMPLETE
            (-7, 2),  # PRUNED (with intermediate values)
            (-3, float("inf")),  # PRUNED (with a NaN intermediate value; it's treated as infinity)
            (float("inf"), 0.0),  # PRUNED (without intermediate values)
        ],
    )
    assert _tpe.sampler._get_observation_pairs(study, "y") == (
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

    assert _tpe.sampler._get_observation_pairs(study, "x") == (
        [5.0, 5.0, 5.0, 5.0],
        [
            (-float("inf"), -5.0),  # COMPLETE
            (-7, -2),  # PRUNED (with intermediate values)
            (-3, float("inf")),  # PRUNED (with a NaN intermediate value; it's treated as infinity)
            (float("inf"), 0.0),  # PRUNED (without intermediate values)
        ],
    )
    assert _tpe.sampler._get_observation_pairs(study, "y") == (
        [None, None, None, None],
        [
            (-float("inf"), -5.0),  # COMPLETE
            (-7, -2),  # PRUNED (with intermediate values)
            (-3, float("inf")),  # PRUNED (with a NaN intermediate value; it's treated as infinity)
            (float("inf"), 0.0),  # PRUNED (without intermediate values)
        ],
    )


def test_get_multivariate_observation_pairs() -> None:
    def objective(trial: Trial) -> float:

        x = trial.suggest_int("x", 5, 5)
        y = trial.suggest_int("y", 6, 6)
        if trial.number == 0:
            return x + y
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

    assert _tpe.sampler._get_multivariate_observation_pairs(study, ["x", "y"]) == (
        {"x": [5.0, 5.0, 5.0, 5.0], "y": [6.0, 6.0, 6.0, 6.0]},
        [
            (-float("inf"), 11.0),  # COMPLETE
            (-7, 2),  # PRUNED (with intermediate values)
            (-3, float("inf")),  # PRUNED (with a NaN intermediate value; it's treated as infinity)
            (float("inf"), 0.0),  # PRUNED (without intermediate values)
        ],
    )

    # Test direction=maximize.
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=4)
    study._storage.create_new_trial(study._study_id)  # Create a running trial.

    assert _tpe.sampler._get_multivariate_observation_pairs(study, ["x", "y"]) == (
        {"x": [5.0, 5.0, 5.0, 5.0], "y": [6.0, 6.0, 6.0, 6.0]},
        [
            (-float("inf"), -11.0),  # COMPLETE
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
        trial_id=idx,
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


def test_call_after_trial_of_random_sampler() -> None:
    sampler = TPESampler()
    study = optuna.create_study(sampler=sampler)
    with patch.object(
        sampler._random_sampler, "after_trial", wraps=sampler._random_sampler.after_trial
    ) as mock_object:
        study.optimize(lambda _: 1.0, n_trials=1)
        assert mock_object.call_count == 1


def test_mixed_relative_search_space_pruned_and_completed_trials() -> None:
    def objective(trial: Trial) -> float:
        if trial.number == 0:
            trial.suggest_uniform("param1", 0, 1)
            raise optuna.exceptions.TrialPruned()

        if trial.number == 1:
            trial.suggest_uniform("param2", 0, 1)
            return 0

        return 0

    sampler = TPESampler(n_startup_trials=1, multivariate=True)
    study = optuna.create_study(sampler=sampler)

    study.optimize(objective, 3)
