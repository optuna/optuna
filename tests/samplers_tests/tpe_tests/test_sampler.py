from __future__ import annotations

from collections.abc import Callable
import random
from unittest.mock import Mock
from unittest.mock import patch
import warnings

import _pytest.capture
import numpy as np
import pytest

import optuna
from optuna import distributions
from optuna.samplers import _tpe
from optuna.samplers import TPESampler
from optuna.samplers._base import _CONSTRAINTS_KEY
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


def test_constraints_func_experimental_warning() -> None:
    with pytest.warns(optuna.exceptions.ExperimentalWarning):
        optuna.samplers.TPESampler(constraints_func=lambda _: (0,))


def test_warn_independent_sampling(capsys: _pytest.capture.CaptureFixture) -> None:
    def objective(trial: Trial) -> float:
        x = trial.suggest_categorical("x", ["a", "b"])
        if x == "a":
            return trial.suggest_float("y", 0, 1)
        else:
            return trial.suggest_float("z", 0, 1)

    # We need to reconstruct our default handler to properly capture stderr.
    optuna.logging._reset_library_root_logger()
    optuna.logging.enable_default_handler()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    sampler = TPESampler(multivariate=True, warn_independent_sampling=True, n_startup_trials=0)
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=10)

    _, err = capsys.readouterr()
    assert err


def test_warn_independent_sampling_group(capsys: _pytest.capture.CaptureFixture) -> None:
    def objective(trial: Trial) -> float:
        x = trial.suggest_categorical("x", ["a", "b"])
        if x == "a":
            return trial.suggest_float("y", 0, 1)
        else:
            return trial.suggest_float("z", 0, 1)

    # We need to reconstruct our default handler to properly capture stderr.
    optuna.logging._reset_library_root_logger()
    optuna.logging.enable_default_handler()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    sampler = TPESampler(
        multivariate=True, warn_independent_sampling=True, group=True, n_startup_trials=0
    )
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=10)

    _, err = capsys.readouterr()
    assert err == ""


def test_infer_relative_search_space() -> None:
    sampler = TPESampler()
    search_space = {
        "a": distributions.FloatDistribution(1.0, 100.0),
        "b": distributions.FloatDistribution(1.0, 100.0, log=True),
        "c": distributions.FloatDistribution(1.0, 100.0, step=3.0),
        "d": distributions.IntDistribution(1, 100),
        "e": distributions.IntDistribution(0, 100, step=2),
        "f": distributions.IntDistribution(1, 100, log=True),
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


def test_sample_relative_prior() -> None:
    study = optuna.create_study()
    dist = optuna.distributions.FloatDistribution(1.0, 100.0)
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
    dist = optuna.distributions.FloatDistribution(1.0, 100.0)
    past_trials = [frozen_trial_factory(i, dist=dist) for i in range(1, 8)]

    trial = frozen_trial_factory(8)
    # sample_relative returns {} for only 4 observations.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = TPESampler(n_startup_trials=5, seed=0, multivariate=True)
    with patch.object(study._storage, "get_all_trials", return_value=past_trials[:4]):
        assert sampler.sample_relative(study, trial, {"param-a": dist}) == {}
    # sample_relative returns some value for only 7 observations.
    study._thread_local.cached_all_trials = None
    with patch.object(study._storage, "get_all_trials", return_value=past_trials):
        assert "param-a" in sampler.sample_relative(study, trial, {"param-a": dist}).keys()


def test_sample_relative_misc_arguments() -> None:
    study = optuna.create_study()
    dist = optuna.distributions.FloatDistribution(1.0, 100.0)
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
            weights=lambda n: np.asarray([i**2 + 1 for i in range(n)]),
            n_startup_trials=5,
            seed=0,
            multivariate=True,
        )
    with patch.object(study._storage, "get_all_trials", return_value=past_trials):
        assert sampler.sample_relative(study, trial, {"param-a": dist}) != suggestion


def test_sample_relative_uniform_distributions() -> None:
    study = optuna.create_study()

    # Prepare sample from uniform distribution for cheking other distributions.
    uni_dist = optuna.distributions.FloatDistribution(1.0, 100.0)
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

    uni_dist = optuna.distributions.FloatDistribution(1.0, 100.0)
    past_trials = [frozen_trial_factory(i, dist=uni_dist) for i in range(1, 8)]
    trial = frozen_trial_factory(8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = TPESampler(n_startup_trials=5, seed=0, multivariate=True)
    with patch.object(study._storage, "get_all_trials", return_value=past_trials):
        uniform_suggestion = sampler.sample_relative(study, trial, {"param-a": uni_dist})

    # Test sample from log-uniform is different from uniform.
    log_dist = optuna.distributions.FloatDistribution(1.0, 100.0, log=True)
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
    disc_dist = optuna.distributions.FloatDistribution(1.0, 100.0, step=0.1)

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

    int_dist = optuna.distributions.IntDistribution(0, 100, step=step)
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

    intlog_dist = optuna.distributions.IntDistribution(1, 100, log=True)
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
    dist = optuna.distributions.FloatDistribution(1.0, 100.0)

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

    dist = optuna.distributions.FloatDistribution(1.0, 100.0)

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

    dist = optuna.distributions.FloatDistribution(1.0, 100.0)

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


def test_sample_independent_prior() -> None:
    study = optuna.create_study()
    dist = optuna.distributions.FloatDistribution(1.0, 100.0)
    past_trials = [frozen_trial_factory(i, dist=dist) for i in range(1, 8)]

    # Prepare a trial and a sample for later checks.
    trial = frozen_trial_factory(8)
    sampler = TPESampler(n_startup_trials=5, seed=0)
    with patch.object(study._storage, "get_all_trials", return_value=past_trials):
        suggestion = sampler.sample_independent(study, trial, "param-a", dist)

    sampler = TPESampler(consider_prior=False, n_startup_trials=5, seed=0)
    with patch.object(study._storage, "get_all_trials", return_value=past_trials):
        assert sampler.sample_independent(study, trial, "param-a", dist) != suggestion

    sampler = TPESampler(prior_weight=0.1, n_startup_trials=5, seed=0)
    with patch.object(study._storage, "get_all_trials", return_value=past_trials):
        assert sampler.sample_independent(study, trial, "param-a", dist) != suggestion


def test_sample_independent_n_startup_trial() -> None:
    study = optuna.create_study()
    dist = optuna.distributions.FloatDistribution(1.0, 100.0)
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
    study._thread_local.cached_all_trials = None
    with patch.object(study._storage, "get_all_trials", return_value=past_trials):
        with patch.object(
            optuna.samplers.RandomSampler, "sample_independent", return_value=1.0
        ) as sample_method:
            sampler.sample_independent(study, trial, "param-a", dist)
    assert sample_method.call_count == 0


def test_sample_independent_misc_arguments() -> None:
    study = optuna.create_study()
    dist = optuna.distributions.FloatDistribution(1.0, 100.0)
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
        weights=lambda i: np.asarray([10 - j for j in range(i)]), n_startup_trials=5, seed=0
    )
    with patch("optuna.Study._get_trials", return_value=past_trials):
        assert sampler.sample_independent(study, trial, "param-a", dist) != suggestion


def test_sample_independent_uniform_distributions() -> None:
    study = optuna.create_study()

    # Prepare sample from uniform distribution for cheking other distributions.
    uni_dist = optuna.distributions.FloatDistribution(1.0, 100.0)
    past_trials = [frozen_trial_factory(i, dist=uni_dist) for i in range(1, 8)]
    trial = frozen_trial_factory(8)
    sampler = TPESampler(n_startup_trials=5, seed=0)
    with patch.object(study._storage, "get_all_trials", return_value=past_trials):
        uniform_suggestion = sampler.sample_independent(study, trial, "param-a", uni_dist)
    assert 1.0 <= uniform_suggestion < 100.0


def test_sample_independent_log_uniform_distributions() -> None:
    """Prepare sample from uniform distribution for cheking other distributions."""

    study = optuna.create_study()

    uni_dist = optuna.distributions.FloatDistribution(1.0, 100.0)
    past_trials = [frozen_trial_factory(i, dist=uni_dist) for i in range(1, 8)]
    trial = frozen_trial_factory(8)
    sampler = TPESampler(n_startup_trials=5, seed=0)
    with patch.object(study._storage, "get_all_trials", return_value=past_trials):
        uniform_suggestion = sampler.sample_independent(study, trial, "param-a", uni_dist)

    # Test sample from log-uniform is different from uniform.
    log_dist = optuna.distributions.FloatDistribution(1.0, 100.0, log=True)
    past_trials = [frozen_trial_factory(i, dist=log_dist) for i in range(1, 8)]
    trial = frozen_trial_factory(8)
    sampler = TPESampler(n_startup_trials=5, seed=0)
    with patch.object(study._storage, "get_all_trials", return_value=past_trials):
        loguniform_suggestion = sampler.sample_independent(study, trial, "param-a", log_dist)
    assert 1.0 <= loguniform_suggestion < 100.0
    assert uniform_suggestion != loguniform_suggestion


def test_sample_independent_discrete_uniform_distributions() -> None:
    """Test samples from discrete have expected intervals."""

    study = optuna.create_study()
    disc_dist = optuna.distributions.FloatDistribution(1.0, 100.0, step=0.1)

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

    int_dist = optuna.distributions.IntDistribution(1, 100)
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

    intlog_dist = optuna.distributions.IntDistribution(1, 100, log=True)
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
    dist = optuna.distributions.FloatDistribution(1.0, 100.0)

    # Prepare sampling result for later tests.
    study = optuna.create_study()
    for i in range(1, 30):
        trial = frozen_trial_factory(i, dist=dist)
        study._storage.create_new_trial(study._study_id, template_trial=trial)
    trial = frozen_trial_factory(30)
    sampler = TPESampler(n_startup_trials=5, seed=2)
    all_success_suggestion = sampler.sample_independent(study, trial, "param-a", dist)

    # Test unsuccessful trials are handled differently.
    state_fn = build_state_fn(state)
    study = optuna.create_study()
    for i in range(1, 30):
        trial = frozen_trial_factory(i, dist=dist, state_fn=state_fn)
        study._storage.create_new_trial(study._study_id, template_trial=trial)
    trial = frozen_trial_factory(30)
    sampler = TPESampler(n_startup_trials=5, seed=2)
    partial_unsuccessful_suggestion = sampler.sample_independent(study, trial, "param-a", dist)

    assert partial_unsuccessful_suggestion != all_success_suggestion


def test_sample_independent_ignored_states() -> None:
    """Tests FAIL, RUNNING, and WAITING states are equally."""

    dist = optuna.distributions.FloatDistribution(1.0, 100.0)

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

    dist = optuna.distributions.FloatDistribution(1.0, 100.0)

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
        sampler = TPESampler(n_startup_trials=5, seed=2)
        suggestions.append(sampler.sample_independent(study, trial, "param-a", dist))

    assert len(set(suggestions)) == 3


def test_constrained_sample_independent_zero_startup() -> None:
    """Tests TPESampler with constrained option works when n_startup_trials=0."""
    study = optuna.create_study()
    dist = optuna.distributions.FloatDistribution(1.0, 100.0)
    trial = frozen_trial_factory(30)
    sampler = TPESampler(n_startup_trials=0, seed=2, constraints_func=lambda _: (0,))
    sampler.sample_independent(study, trial, "param-a", dist)


@pytest.mark.parametrize("direction", ["minimize", "maximize"])
@pytest.mark.parametrize("constant_liar", [True, False])
@pytest.mark.parametrize("constraints", [True, False])
def test_split_trials(direction: str, constant_liar: bool, constraints: bool) -> None:
    study = optuna.create_study(direction=direction)

    for value in [-float("inf"), 0, 1, float("inf")]:
        study.add_trial(
            optuna.create_trial(
                state=optuna.trial.TrialState.COMPLETE,
                value=(value if direction == "minimize" else -value),
                params={"x": 0},
                distributions={"x": optuna.distributions.FloatDistribution(-1.0, 1.0)},
                system_attrs={_CONSTRAINTS_KEY: [-1]},
            )
        )

    for step in [2, 1]:
        for value in [-float("inf"), 0, 1, float("inf"), float("nan")]:
            study.add_trial(
                optuna.create_trial(
                    state=optuna.trial.TrialState.PRUNED,
                    params={"x": 0},
                    distributions={"x": optuna.distributions.FloatDistribution(-1.0, 1.0)},
                    system_attrs={_CONSTRAINTS_KEY: [-1]},
                    intermediate_values={step: (value if direction == "minimize" else -value)},
                )
            )

    study.add_trial(
        optuna.create_trial(
            state=optuna.trial.TrialState.PRUNED,
            params={"x": 0},
            distributions={"x": optuna.distributions.FloatDistribution(-1.0, 1.0)},
            system_attrs={_CONSTRAINTS_KEY: [-1]},
        )
    )

    if constraints:
        for value in [1, 2, float("inf")]:
            study.add_trial(
                optuna.create_trial(
                    state=optuna.trial.TrialState.COMPLETE,
                    value=0,
                    params={"x": 0},
                    distributions={"x": optuna.distributions.FloatDistribution(-1.0, 1.0)},
                    system_attrs={_CONSTRAINTS_KEY: [value]},
                )
            )

    study.add_trial(
        optuna.create_trial(
            state=optuna.trial.TrialState.RUNNING,
            params={"x": 0},
            distributions={"x": optuna.distributions.FloatDistribution(-1.0, 1.0)},
        )
    )

    study.add_trial(
        optuna.create_trial(
            state=optuna.trial.TrialState.FAIL,
        )
    )

    study.add_trial(
        optuna.create_trial(
            state=optuna.trial.TrialState.WAITING,
        )
    )

    if constant_liar:
        states = [
            optuna.trial.TrialState.COMPLETE,
            optuna.trial.TrialState.PRUNED,
            optuna.trial.TrialState.RUNNING,
        ]
    else:
        states = [optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED]

    trials = study.get_trials(states=states)
    finished_trials = study.get_trials(
        states=(optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    )
    for n_below in range(len(finished_trials) + 1):
        below_trials, above_trials = _tpe.sampler._split_trials(
            study,
            trials,
            n_below,
            constraints,
        )

        below_trial_numbers = [trial.number for trial in below_trials]
        assert below_trial_numbers == list(range(n_below))
        above_trial_numbers = [trial.number for trial in above_trials]
        assert above_trial_numbers == list(range(n_below, len(trials)))


@pytest.mark.parametrize(
    "directions", [["minimize", "minimize"], ["maximize", "maximize"], ["minimize", "maximize"]]
)
def test_split_trials_for_multiobjective_constant_liar(directions: list[str]) -> None:
    study = optuna.create_study(directions=directions)
    # 16 Trials (#0 -- #15) that should be sorted by non-dominated sort and HSSP.
    for obj1 in [-float("inf"), 0, 1, float("inf")]:
        val1 = obj1 if directions[0] == "minimize" else -obj1
        for obj2 in [-float("inf"), 0, 1, float("inf")]:
            val2 = obj2 if directions[1] == "minimize" else -obj2
            study.add_trial(
                optuna.create_trial(
                    state=optuna.trial.TrialState.COMPLETE,
                    values=[val1, val2],
                    params={"x": 0},
                    distributions={"x": optuna.distributions.FloatDistribution(-1.0, 1.0)},
                )
            )

    # 5 Trials (#16 -- # 20) that should come at the end of the sorting.
    n_running_trials = 5
    for _ in range(n_running_trials):
        study.add_trial(
            optuna.create_trial(
                state=optuna.trial.TrialState.RUNNING,
                params={"x": 0},
                distributions={"x": optuna.distributions.FloatDistribution(-1.0, 1.0)},
            )
        )

    # 2 Trials (#21, #22) that should be ignored in the sorting.
    study.add_trial(optuna.create_trial(state=optuna.trial.TrialState.FAIL))
    study.add_trial(optuna.create_trial(state=optuna.trial.TrialState.WAITING))

    states = [optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.RUNNING]
    trials = study.get_trials(states=states)
    finished_trials = study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))
    # Below is the relation of `non-domination rank: trial_numbers`.
    # 0: [0], 1: [1,4], 2: [2,5,8], 3: [3,6,9,12], 4: [7,10,13], 5: [11,14], 6: [15]
    # NOTE(nabenabe0928): As each `values` includes `inf`, ref_point also includes `inf`,
    # leading to an arbitrary hypervolume contribution to be `inf`. That is why HSSP starts to pick
    # an earlier trial number in each non-domination rank.
    ground_truth = [0, 1, 4, 2, 8, 5, 3, 6, 9, 12, 7, 10, 13, 11, 14, 15]
    n_completed_trials = len(ground_truth)
    # NOTE(nabenabe0928): Running trials (#16 -- #20) must come at the end.
    ground_truth += [n_completed_trials + i for i in range(n_running_trials)]
    for n_below in range(1, len(finished_trials) + 1):
        below_trials, above_trials = _tpe.sampler._split_trials(
            study, trials, n_below, constraints_enabled=False
        )
        below_trial_numbers = [trial.number for trial in below_trials]
        assert below_trial_numbers == sorted(ground_truth[:n_below])
        above_trial_numbers = [trial.number for trial in above_trials]
        assert above_trial_numbers == sorted(ground_truth[n_below:])


@pytest.mark.parametrize("direction", ["minimize", "maximize"])
def test_split_complete_trials_single_objective(direction: str) -> None:
    study = optuna.create_study(direction=direction)

    for value in [-float("inf"), 0, 1, float("inf")]:
        study.add_trial(
            optuna.create_trial(
                state=optuna.trial.TrialState.COMPLETE,
                value=(value if direction == "minimize" else -value),
                params={"x": 0},
                distributions={"x": optuna.distributions.FloatDistribution(-1.0, 1.0)},
            )
        )

    for n_below in range(len(study.trials) + 1):
        below_trials, above_trials = _tpe.sampler._split_complete_trials_single_objective(
            study.trials,
            study,
            n_below,
        )
        assert [trial.number for trial in below_trials] == list(range(n_below))
        assert [trial.number for trial in above_trials] == list(range(n_below, len(study.trials)))


def test_split_complete_trials_single_objective_empty() -> None:
    study = optuna.create_study()
    assert _tpe.sampler._split_complete_trials_single_objective([], study, 0) == ([], [])


@pytest.mark.parametrize("direction", ["minimize", "maximize"])
def test_split_pruned_trials(direction: str) -> None:
    study = optuna.create_study(direction=direction)

    for step in [2, 1]:
        for value in [-float("inf"), 0, 1, float("inf"), float("nan")]:
            study.add_trial(
                optuna.create_trial(
                    state=optuna.trial.TrialState.PRUNED,
                    params={"x": 0},
                    distributions={"x": optuna.distributions.FloatDistribution(-1.0, 1.0)},
                    intermediate_values={step: (value if direction == "minimize" else -value)},
                )
            )

    study.add_trial(
        optuna.create_trial(
            state=optuna.trial.TrialState.PRUNED,
            params={"x": 0},
            distributions={"x": optuna.distributions.FloatDistribution(-1.0, 1.0)},
        )
    )

    for n_below in range(len(study.trials) + 1):
        below_trials, above_trials = _tpe.sampler._split_pruned_trials(
            study.trials,
            study,
            n_below,
        )
        assert [trial.number for trial in below_trials] == list(range(n_below))
        assert [trial.number for trial in above_trials] == list(range(n_below, len(study.trials)))


def test_split_pruned_trials_empty() -> None:
    study = optuna.create_study()
    assert _tpe.sampler._split_pruned_trials([], study, 0) == ([], [])


@pytest.mark.parametrize("direction", ["minimize", "maximize"])
def test_split_infeasible_trials(direction: str) -> None:
    study = optuna.create_study(direction=direction)

    for value in [1, 2, float("inf")]:
        study.add_trial(
            optuna.create_trial(
                state=optuna.trial.TrialState.COMPLETE,
                value=0,
                params={"x": 0},
                distributions={"x": optuna.distributions.FloatDistribution(-1.0, 1.0)},
                system_attrs={_CONSTRAINTS_KEY: [value]},
            )
        )

    for n_below in range(len(study.trials) + 1):
        below_trials, above_trials = _tpe.sampler._split_infeasible_trials(study.trials, n_below)
        assert [trial.number for trial in below_trials] == list(range(n_below))
        assert [trial.number for trial in above_trials] == list(range(n_below, len(study.trials)))


def test_split_infeasible_trials_empty() -> None:
    assert _tpe.sampler._split_infeasible_trials([], 0) == ([], [])


def frozen_trial_factory(
    idx: int,
    dist: optuna.distributions.BaseDistribution = optuna.distributions.FloatDistribution(
        1.0, 100.0
    ),
    state_fn: Callable[
        [int], optuna.trial.TrialState
    ] = lambda _: optuna.trial.TrialState.COMPLETE,
    value_fn: Callable[[int], int | float] | None = None,
    target_fn: Callable[[float], float] = lambda val: (val - 20.0) ** 2,
    interm_val_fn: Callable[[int], dict[int, float]] = lambda _: {},
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
            trial.suggest_float("param1", 0, 1)
            raise optuna.exceptions.TrialPruned()

        if trial.number == 1:
            trial.suggest_float("param2", 0, 1)
            return 0

        return 0

    sampler = TPESampler(n_startup_trials=1, multivariate=True)
    study = optuna.create_study(sampler=sampler)

    study.optimize(objective, 3)


def test_group() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = TPESampler(multivariate=True, group=True)
    study = optuna.create_study(sampler=sampler)

    with patch.object(sampler, "_sample_relative", wraps=sampler._sample_relative) as mock:
        study.optimize(lambda t: t.suggest_int("x", 0, 10), n_trials=2)
        assert mock.call_count == 1
    assert study.trials[-1].distributions == {"x": distributions.IntDistribution(low=0, high=10)}

    with patch.object(sampler, "_sample_relative", wraps=sampler._sample_relative) as mock:
        study.optimize(
            lambda t: t.suggest_int("y", 0, 10) + t.suggest_float("z", -3, 3), n_trials=1
        )
        assert mock.call_count == 1
    assert study.trials[-1].distributions == {
        "y": distributions.IntDistribution(low=0, high=10),
        "z": distributions.FloatDistribution(low=-3, high=3),
    }

    with patch.object(sampler, "_sample_relative", wraps=sampler._sample_relative) as mock:
        study.optimize(
            lambda t: t.suggest_int("y", 0, 10)
            + t.suggest_float("z", -3, 3)
            + t.suggest_float("u", 1e-2, 1e2, log=True)
            + bool(t.suggest_categorical("v", ["A", "B", "C"])),
            n_trials=1,
        )
        assert mock.call_count == 2
    assert study.trials[-1].distributions == {
        "u": distributions.FloatDistribution(low=1e-2, high=1e2, log=True),
        "v": distributions.CategoricalDistribution(choices=["A", "B", "C"]),
        "y": distributions.IntDistribution(low=0, high=10),
        "z": distributions.FloatDistribution(low=-3, high=3),
    }

    with patch.object(sampler, "_sample_relative", wraps=sampler._sample_relative) as mock:
        study.optimize(lambda t: t.suggest_float("u", 1e-2, 1e2, log=True), n_trials=1)
        assert mock.call_count == 3
    assert study.trials[-1].distributions == {
        "u": distributions.FloatDistribution(low=1e-2, high=1e2, log=True)
    }

    with patch.object(sampler, "_sample_relative", wraps=sampler._sample_relative) as mock:
        study.optimize(
            lambda t: t.suggest_int("y", 0, 10) + t.suggest_int("w", 2, 8, log=True), n_trials=1
        )
        assert mock.call_count == 4
    assert study.trials[-1].distributions == {
        "y": distributions.IntDistribution(low=0, high=10),
        "w": distributions.IntDistribution(low=2, high=8, log=True),
    }

    with patch.object(sampler, "_sample_relative", wraps=sampler._sample_relative) as mock:
        study.optimize(lambda t: t.suggest_int("x", 0, 10), n_trials=1)
        assert mock.call_count == 6
    assert study.trials[-1].distributions == {"x": distributions.IntDistribution(low=0, high=10)}


def test_invalid_multivariate_and_group() -> None:
    with pytest.raises(ValueError):
        _ = TPESampler(multivariate=False, group=True)


def test_group_experimental_warning() -> None:
    with pytest.warns(optuna.exceptions.ExperimentalWarning):
        _ = TPESampler(multivariate=True, group=True)


def test_constant_liar_experimental_warning() -> None:
    with pytest.warns(optuna.exceptions.ExperimentalWarning):
        _ = TPESampler(constant_liar=True)


@pytest.mark.parametrize("multivariate", [True, False])
@pytest.mark.parametrize("multiobjective", [True, False])
def test_constant_liar_with_running_trial(multivariate: bool, multiobjective: bool) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = TPESampler(multivariate=multivariate, constant_liar=True, n_startup_trials=0)

    directions = ["minimize"] * 2 if multiobjective else ["minimize"]
    study = optuna.create_study(sampler=sampler, directions=directions)

    # Add a complete trial.
    trial0 = study.ask()
    trial0.suggest_int("x", 0, 10)
    trial0.suggest_float("y", 0, 10)
    trial0.suggest_categorical("z", [0, 1, 2])
    study.tell(trial0, [0, 0] if multiobjective else 0)

    # Add running trials.
    trial1 = study.ask()
    trial1.suggest_int("x", 0, 10)
    trial2 = study.ask()
    trial2.suggest_float("y", 0, 10)
    trial3 = study.ask()
    trial3.suggest_categorical("z", [0, 1, 2])

    # Test suggestion with running trials.
    trial = study.ask()
    trial.suggest_int("x", 0, 10)
    trial.suggest_float("y", 0, 10)
    trial.suggest_categorical("z", [0, 1, 2])
    study.tell(trial, [0, 0] if multiobjective else 0)


def test_categorical_distance_func_experimental_warning() -> None:
    with pytest.warns(optuna.exceptions.ExperimentalWarning):
        _ = TPESampler(categorical_distance_func={"c": lambda x, y: 0.0})
