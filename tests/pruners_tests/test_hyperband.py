from __future__ import annotations

from collections.abc import Callable
from unittest import mock

import pytest

import optuna


MIN_RESOURCE = 1
MAX_RESOURCE = 16
REDUCTION_FACTOR = 2
N_BRACKETS = 4
EARLY_STOPPING_RATE_LOW = 0
EARLY_STOPPING_RATE_HIGH = 3
N_REPORTS = 10
EXPECTED_N_TRIALS_PER_BRACKET = 10


def test_hyperband_pruner_intermediate_values() -> None:
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=MIN_RESOURCE, max_resource=MAX_RESOURCE, reduction_factor=REDUCTION_FACTOR
    )

    study = optuna.study.create_study(sampler=optuna.samplers.RandomSampler(), pruner=pruner)

    def objective(trial: optuna.trial.Trial) -> float:
        for i in range(N_REPORTS):
            trial.report(i, step=i)

        return 1.0

    study.optimize(objective, n_trials=N_BRACKETS * EXPECTED_N_TRIALS_PER_BRACKET)

    trials = study.trials
    assert len(trials) == N_BRACKETS * EXPECTED_N_TRIALS_PER_BRACKET


def test_bracket_study() -> None:
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=MIN_RESOURCE, max_resource=MAX_RESOURCE, reduction_factor=REDUCTION_FACTOR
    )
    study = optuna.study.create_study(sampler=optuna.samplers.RandomSampler(), pruner=pruner)
    bracket_study = pruner._create_bracket_study(study, 0)

    with pytest.raises(AttributeError):
        bracket_study.optimize(lambda *args: 1.0)

    with pytest.raises(AttributeError):
        bracket_study.set_user_attr("abc", 100)

    for attr in ("user_attrs", "system_attrs"):
        with pytest.raises(AttributeError):
            getattr(bracket_study, attr)

    with pytest.raises(AttributeError):
        bracket_study.trials_dataframe()

    bracket_study.get_trials()
    bracket_study.direction
    bracket_study._storage
    bracket_study._study_id
    bracket_study.pruner
    bracket_study.study_name
    # As `_BracketStudy` is defined inside `HyperbandPruner`,
    # we cannot do `assert isinstance(bracket_study, _BracketStudy)`.
    # This is why the below line is ignored by mypy checks.
    bracket_study._bracket_id  # type: ignore


def test_hyperband_max_resource_is_auto() -> None:
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=MIN_RESOURCE, reduction_factor=REDUCTION_FACTOR
    )
    study = optuna.study.create_study(sampler=optuna.samplers.RandomSampler(), pruner=pruner)

    def objective(trial: optuna.trial.Trial) -> float:
        for i in range(N_REPORTS):
            trial.report(1.0, i)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return 1.0

    study.optimize(objective, n_trials=N_BRACKETS * EXPECTED_N_TRIALS_PER_BRACKET)

    assert N_REPORTS == pruner._max_resource


def test_hyperband_max_resource_value_error() -> None:
    with pytest.raises(ValueError):
        _ = optuna.pruners.HyperbandPruner(max_resource="not_appropriate")


@pytest.mark.parametrize(
    "sampler_init_func",
    [
        lambda: optuna.samplers.RandomSampler(),
        (lambda: optuna.samplers.TPESampler(n_startup_trials=1)),
        (
            lambda: optuna.samplers.GridSampler(
                search_space={"value": [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]}
            )
        ),
        (lambda: optuna.samplers.CmaEsSampler(n_startup_trials=1)),
    ],
)
def test_hyperband_filter_study(
    sampler_init_func: Callable[[], optuna.samplers.BaseSampler],
) -> None:
    def objective(trial: optuna.trial.Trial) -> float:
        return trial.suggest_float("value", 0.0, 1.0)

    n_trials = 8
    n_brackets = 4
    expected_n_trials_per_bracket = n_trials // n_brackets
    with mock.patch(
        "optuna.pruners.HyperbandPruner._get_bracket_id",
        new=mock.Mock(side_effect=lambda study, trial: trial.number % n_brackets),
    ):
        for method_name in [
            "infer_relative_search_space",
            "sample_relative",
            "sample_independent",
        ]:
            sampler = sampler_init_func()
            pruner = optuna.pruners.HyperbandPruner(
                min_resource=MIN_RESOURCE,
                max_resource=MAX_RESOURCE,
                reduction_factor=REDUCTION_FACTOR,
            )
            with mock.patch(
                "optuna.samplers.{}.{}".format(sampler.__class__.__name__, method_name),
                wraps=getattr(sampler, method_name),
            ) as method_mock:
                study = optuna.study.create_study(sampler=sampler, pruner=pruner)
                study.optimize(objective, n_trials=n_trials)
                args = method_mock.call_args[0]
                study = args[0]
                trials = study.get_trials()
                assert len(trials) == expected_n_trials_per_bracket


@pytest.mark.parametrize(
    "pruner_init_func",
    [
        lambda: optuna.pruners.NopPruner(),
        lambda: optuna.pruners.MedianPruner(),
        lambda: optuna.pruners.ThresholdPruner(lower=0.5),
        lambda: optuna.pruners.SuccessiveHalvingPruner(),
    ],
)
def test_hyperband_no_filter_study(
    pruner_init_func: Callable[[], optuna.pruners.BasePruner],
) -> None:
    def objective(trial: optuna.trial.Trial) -> float:
        return trial.suggest_float("value", 0.0, 1.0)

    n_trials = 10
    for method_name in [
        "infer_relative_search_space",
        "sample_relative",
        "sample_independent",
    ]:
        sampler = optuna.samplers.RandomSampler()
        pruner = pruner_init_func()
        with mock.patch(
            "optuna.samplers.{}.{}".format(sampler.__class__.__name__, method_name),
            wraps=getattr(sampler, method_name),
        ) as method_mock:
            study = optuna.study.create_study(sampler=sampler, pruner=pruner)
            study.optimize(objective, n_trials=n_trials)
            args = method_mock.call_args[0]
            study = args[0]
            trials = study.get_trials()
            assert len(trials) == n_trials


@pytest.mark.parametrize(
    "sampler_init_func",
    [
        lambda: optuna.samplers.RandomSampler(),
        (lambda: optuna.samplers.TPESampler(n_startup_trials=1)),
        (
            lambda: optuna.samplers.GridSampler(
                search_space={"value": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
            )
        ),
        (lambda: optuna.samplers.CmaEsSampler(n_startup_trials=1)),
    ],
)
def test_hyperband_no_call_of_filter_study_in_should_prune(
    sampler_init_func: Callable[[], optuna.samplers.BaseSampler],
) -> None:
    def objective(trial: optuna.trial.Trial) -> float:
        with mock.patch("optuna.pruners._filter_study") as method_mock:
            for i in range(N_REPORTS):
                trial.report(i, step=i)
                if trial.should_prune():
                    method_mock.assert_not_called()
                    raise optuna.TrialPruned()
                else:
                    method_mock.assert_not_called()

        return 1.0

    sampler = sampler_init_func()
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=MIN_RESOURCE, max_resource=MAX_RESOURCE, reduction_factor=REDUCTION_FACTOR
    )
    study = optuna.study.create_study(sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=10)


def test_incompatibility_between_bootstrap_count_and_auto_max_resource() -> None:
    with pytest.raises(ValueError):
        optuna.pruners.HyperbandPruner(max_resource="auto", bootstrap_count=1)


def test_hyperband_pruner_and_grid_sampler() -> None:
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=MIN_RESOURCE, max_resource=MAX_RESOURCE, reduction_factor=REDUCTION_FACTOR
    )

    search_space = {"x": [-50, 0, 50], "y": [-99, 0, 99]}
    sampler = optuna.samplers.GridSampler(search_space)

    study = optuna.study.create_study(sampler=sampler, pruner=pruner)

    def objective(trial: optuna.trial.Trial) -> float:
        for i in range(N_REPORTS):
            trial.report(i, step=i)

        x = trial.suggest_float("x", -100, 100)
        y = trial.suggest_int("y", -100, 100)
        return x**2 + y**2

    study.optimize(objective, n_trials=10)

    trials = study.trials
    assert len(trials) == 9
