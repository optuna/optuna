from __future__ import annotations

from collections.abc import Sequence
import importlib
import sys
import warnings

from _pytest.logging import LogCaptureFixture
import numpy as np
import pytest

import optuna
import optuna._gp.acqf as acqf_module
import optuna._gp.gp as optuna_gp
import optuna._gp.optim_mixed as optim_mixed
import optuna._gp.prior as prior
import optuna._gp.search_space as gp_search_space
from optuna.samplers import GPSampler
from optuna.samplers._gp.sampler import _get_constraint_vals_and_feasibility
from optuna.trial import FrozenTrial


def test_after_convergence(caplog: LogCaptureFixture) -> None:
    # A large `optimal_trials` causes the instability in the kernel inversion, leading to
    # instability in the variance calculation.
    X_uniform = [(i + 1) / 10 for i in range(10)]
    X_uniform_near_optimal = [(i + 1) / 1e5 for i in range(20)]
    X_optimal = [0.0] * 10
    X = np.array(X_uniform + X_uniform_near_optimal + X_optimal)
    score_vals = -(X - np.mean(X)) / np.std(X)
    search_space = gp_search_space.SearchSpace(
        {"a": optuna.distributions.FloatDistribution(0.0, 1.0)}
    )
    gpr = optuna_gp.fit_kernel_params(
        X=X[:, np.newaxis],
        Y=score_vals,
        is_categorical=np.array([False]),
        log_prior=prior.default_log_prior,
        minimum_noise=prior.DEFAULT_MINIMUM_NOISE_VAR,
        deterministic_objective=False,
    )
    acqf_params = acqf_module.LogEI(
        gpr=gpr, search_space=search_space, threshold=np.max(score_vals)
    )
    caplog.clear()
    optuna.logging.enable_propagation()
    optim_mixed.optimize_acqf_mixed(acqf_params, rng=np.random.RandomState(42))
    # len(caplog.text) > 0 means the optimization has already converged.
    assert len(caplog.text) > 0, "Did you change the kernel implementation?"


def test_after_convergence_with_zero_sum_probabilities() -> None:
    X_uniform = [(i + 1) / 10 for i in range(10)]
    X_uniform_near_optimal = [(i + 1) / 1e5 for i in range(20)]
    X_optimal = [0.0] * 60
    X = np.array(X_uniform + X_uniform_near_optimal + X_optimal)
    score_vals = -(X - np.mean(X)) / np.std(X)
    search_space = gp_search_space.SearchSpace(
        {"a": optuna.distributions.FloatDistribution(0.0, 1.0)}
    )
    gpr = optuna_gp.fit_kernel_params(
        X=X[:, np.newaxis],
        Y=score_vals,
        is_categorical=np.array([False]),
        log_prior=prior.default_log_prior,
        minimum_noise=prior.DEFAULT_MINIMUM_NOISE_VAR,
        deterministic_objective=False,
    )
    acqf_params = acqf_module.LogEI(
        gpr=gpr, search_space=search_space, threshold=np.max(score_vals)
    )
    sampled_xs = search_space.sample_normalized_params(2048, rng=np.random.RandomState(22))
    f_vals = acqf_params.eval_acqf_no_grad(sampled_xs)
    max_i = np.argmax(f_vals)
    probs = np.exp(f_vals - f_vals[max_i])
    probs[max_i] = 0.0
    assert np.isfinite(f_vals).all()
    assert not np.any(probs)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        optim_mixed.optimize_acqf_mixed(acqf_params, rng=np.random.RandomState(22))


@pytest.mark.parametrize("constraint_value", [-1.0, 0.0, 1.0, -float("inf"), float("inf")])
@pytest.mark.parametrize("n_objectives", [1, 2])
@pytest.mark.filterwarnings("ignore:.*GPSampler cannot handle infinite values*")
def test_constraints_func(constraint_value: float, n_objectives: int) -> None:
    n_trials = 5
    constraints_func_call_count = 0

    def constraints_func(trial: FrozenTrial) -> Sequence[float]:
        nonlocal constraints_func_call_count
        constraints_func_call_count += 1

        return (constraint_value + trial.number,)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        sampler = GPSampler(n_startup_trials=2, constraints_func=constraints_func)

    def objective(trial: optuna.Trial) -> float | tuple[float, float]:
        x = trial.suggest_float("x", 0, 1)
        if n_objectives == 1:
            return x
        else:
            return x, (x - 2) ** 2

    study = optuna.create_study(directions=["minimize"] * n_objectives, sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    assert len(study.trials) == n_trials
    assert constraints_func_call_count == n_trials
    for trial in study.trials:
        for x, y in zip(trial.constraints.values(), (constraint_value + trial.number,)):
            assert x == y


@pytest.mark.parametrize("n_objectives", [1, 2])
def test_constraints_func_nan(n_objectives: int) -> None:
    n_trials = 5

    def constraints_func(_: FrozenTrial) -> Sequence[float]:
        return (float("nan"),)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        sampler = GPSampler(n_startup_trials=2, constraints_func=constraints_func)

    def objective(
        trial: optuna.Trial | optuna.trial.FrozenTrial,
    ) -> tuple[float] | tuple[float, float]:
        x = trial.suggest_float("x", 0, 1)
        if n_objectives == 1:
            return (x,)
        else:
            return x, (x - 2) ** 2

    study = optuna.create_study(directions=["minimize"] * n_objectives, sampler=sampler)
    with pytest.raises(ValueError):
        study.optimize(objective, n_trials=n_trials)

    trials = study.get_trials()
    assert len(trials) == 1  # The error stops optimization, but completed trials are recorded.
    assert all(0 <= x <= 1 for x in trials[0].params.values())  # The params are normal.
    assert trials[0].values == list(objective(trials[0]))  # The values are normal.
    assert len(trials[0].constraints) == 0  # No constraints are set.


@pytest.mark.parametrize("keys", [("latency", "memory"), ("1", "0")])
def test_constraint_values_follow_names(keys: tuple[str, str]) -> None:
    study = optuna.create_study()
    trials = [
        optuna.trial.create_trial(value=0, constraints={keys[0]: -1.0, keys[1]: 2.0}),
        optuna.trial.create_trial(value=0, constraints={keys[1]: -3.0, keys[0]: 4.0}),
        optuna.trial.create_trial(value=0, constraints={keys[0]: 0.0, keys[1]: -2.0}),
    ]

    values, is_feasible = _get_constraint_vals_and_feasibility(study, trials)

    np.testing.assert_array_equal(values, [[-1.0, 2.0], [4.0, -3.0], [0.0, -2.0]])
    np.testing.assert_array_equal(is_feasible, [False, False, True])


@pytest.mark.parametrize("constraints", [{"memory": -1.0}, {}])
def test_inconsistent_constraint_names(constraints: dict[str, float]) -> None:
    study = optuna.create_study()
    trials = [
        optuna.trial.create_trial(value=0, constraints={"latency": -1.0}),
        optuna.trial.create_trial(value=0, constraints=constraints),
    ]

    with pytest.raises(ValueError):
        _get_constraint_vals_and_feasibility(study, trials)


@pytest.mark.parametrize("n_objectives", [1, 2])
def test_named_constraint_order_does_not_change_suggestion(n_objectives: int) -> None:
    suggestions = []
    for reverse_order in [False, True]:
        study = optuna.create_study(
            directions=["minimize"] * n_objectives,
            sampler=GPSampler(seed=0, n_startup_trials=1),
        )
        for i, x in enumerate(np.linspace(0, 1, 6)):
            constraints = {"low": float(0.15 - x), "high": float(x - 0.85)}
            if reverse_order and i % 2:
                constraints = dict(reversed(list(constraints.items())))
            study.add_trial(
                optuna.trial.create_trial(
                    values=[(x - 0.2) ** 2, (x - 0.8) ** 2][:n_objectives],
                    params={"x": x},
                    distributions={"x": optuna.distributions.FloatDistribution(0, 1)},
                    constraints=constraints,
                )
            )
        suggestions.append(study.ask().suggest_float("x", 0, 1))

    assert suggestions[0] == pytest.approx(suggestions[1], abs=1e-8)


def test_behavior_without_greenlet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "greenlet", None)
    import optuna._gp.batched_lbfgsb as optimization_module

    importlib.reload(optimization_module)
    assert optimization_module._greenlet_imports.is_successful() is False

    # See if optimization still works without greenlet
    import optuna

    sampler = optuna.samplers.GPSampler(seed=42)
    study = optuna.create_study(sampler=sampler)
    study.optimize(lambda trial: trial.suggest_float("x", -10, 10) ** 2, n_trials=15)


def test_gpsampler_with_cuda_default_device() -> None:
    """Test that GPSampler works when torch.set_default_device('cuda') is set.

    Regression test for issue #6113. When users set torch.set_default_device('cuda'),
    GPSampler should still work by forcing CPU device internally.
    """
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Set CUDA as the default device (simulating user's global setting)
    original_device = torch.get_default_device()
    torch.set_default_device("cuda")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
            sampler = GPSampler(n_startup_trials=2, seed=42)

        study = optuna.create_study(sampler=sampler)
        # Run enough trials to trigger GP sampling (past startup trials)
        study.optimize(lambda trial: trial.suggest_float("x", -10, 10) ** 2, n_trials=5)

        assert len(study.trials) == 5
    finally:
        # Restore original device setting
        if original_device is None:
            torch.set_default_device(None)
        else:
            torch.set_default_device(original_device)
