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
                f"optuna.samplers.{sampler.__class__.__name__}.{method_name}",
                wraps=getattr(sampler, method_name),
            ) as method_mock:
                study = optuna.study.create_study(sampler=sampler, pruner=pruner)
                study.optimize(objective, n_trials=n_trials)
                args = method_mock.call_args[0]
                study = args[0]
                trials = study.get_trials()
                assert len(trials) == expected_n_trials_per_bracket


@pytest.mark.parametrize("n_trials_per_bracket", [6, 11], ids=["startup", "post_startup"])
@pytest.mark.parametrize("constant_liar", [False, True])
def test_hyperband_tpe_ignores_other_bracket_values(
    n_trials_per_bracket: int, constant_liar: bool
) -> None:
    n_trials = 2 * n_trials_per_bracket
    distributions: dict[str, optuna.distributions.BaseDistribution] = {
        name: optuna.distributions.FloatDistribution(0.0, 1.0) for name in ("x", "y")
    }
    suggestions = []
    with mock.patch.object(
        optuna.pruners.HyperbandPruner,
        "_get_bracket_id",
        side_effect=lambda study, trial: trial.number % 2,
    ):
        for other_bracket_value in (-1.0, 1.0):
            study = optuna.create_study(
                sampler=optuna.samplers.TPESampler(seed=7, constant_liar=constant_liar),
                pruner=optuna.pruners.HyperbandPruner(max_resource=2, reduction_factor=2),
            )
            for number in range(n_trials):
                params = {
                    "x": (number + 1) / (n_trials + 1),
                    "y": (number * 7 % n_trials + 1) / (n_trials + 1),
                }
                value = (
                    sum((value - 0.5) ** 2 for value in params.values())
                    if number % 2 == 0
                    else other_bracket_value * (1.0 + sum(params.values()))
                )
                study.add_trial(
                    optuna.trial.create_trial(
                        value=value, params=params, distributions=distributions
                    )
                )
            trial = study.ask()
            suggestions.append({name: trial.suggest_float(name, 0.0, 1.0) for name in ("x", "y")})

    assert suggestions[0] == suggestions[1]
    random_trial = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=7)).ask()
    random_suggestion = {name: random_trial.suggest_float(name, 0.0, 1.0) for name in ("x", "y")}
    assert (suggestions[0] == random_suggestion) == (n_trials_per_bracket < 10)


def test_hyperband_tpe_ignores_running_trials_in_other_brackets() -> None:
    distribution = optuna.distributions.FloatDistribution(0.0, 1.0)
    suggestions = []
    with mock.patch.object(
        optuna.pruners.HyperbandPruner,
        "_get_bracket_id",
        side_effect=lambda study, trial: trial.number % 2,
    ):
        for local_x, other_x in ((0.7, 0.2), (0.7, 0.7), (0.2, 0.2)):
            study = optuna.create_study(
                sampler=optuna.samplers.TPESampler(seed=7, n_startup_trials=1, constant_liar=True),
                pruner=optuna.pruners.HyperbandPruner(max_resource=2, reduction_factor=2),
            )
            for x, value in ((0.3, 1.0), (0.6, 0.0)):
                study.add_trial(
                    optuna.trial.create_trial(
                        value=value, params={"x": x}, distributions={"x": distribution}
                    )
                )
            for x in (local_x, other_x):
                study.add_trial(
                    optuna.trial.create_trial(
                        state=optuna.trial.TrialState.RUNNING,
                        params={"x": x},
                        distributions={"x": distribution},
                    )
                )
            suggestions.append(study.ask().suggest_float("x", 0.0, 1.0))

    assert suggestions[0] == suggestions[1]
    assert suggestions[0] != suggestions[2]


@pytest.mark.parametrize("group", [False, True])
def test_hyperband_tpe_infers_each_bracket_search_space(group: bool) -> None:
    sampler = optuna.samplers.TPESampler(seed=7, n_startup_trials=1, group=group)
    study = optuna.create_study(
        sampler=sampler,
        pruner=optuna.pruners.HyperbandPruner(max_resource=2, reduction_factor=2),
    )
    distribution = optuna.distributions.FloatDistribution(0.0, 1.0)
    search_spaces: list[dict[str, optuna.distributions.BaseDistribution]] = [
        {name: distribution for name in names} for names in (("x", "y"), ("x", "z"))
    ]
    for search_space in search_spaces:
        study.add_trial(
            optuna.trial.create_trial(
                value=0.0,
                params=dict.fromkeys(search_space, 0.5),
                distributions=search_space,
            )
        )

    with mock.patch.object(
        optuna.pruners.HyperbandPruner,
        "_get_bracket_id",
        side_effect=lambda study, trial: trial.number % 2,
    ):
        for search_space in [*search_spaces, search_spaces[0]]:
            with mock.patch.object(
                sampler, "_sample_relative", wraps=sampler._sample_relative
            ) as sample_relative:
                trial = study.ask(fixed_distributions=search_space)
            assert trial.relative_search_space == search_space
            assert [call.args[2] for call in sample_relative.call_args_list] == [search_space]
            study.tell(trial, 0.0)


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
            f"optuna.samplers.{sampler.__class__.__name__}.{method_name}",
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


def test_hyperband_pruner_and_brute_force_sampler() -> None:
    def objective(trial: optuna.trial.Trial) -> int:
        x = trial.suggest_int("x", 0, 2)
        y = trial.suggest_int("y", 0, 2)
        for i in range(10):
            trial.report(step=i, value=i * (x + y) / 10)
            if trial.should_prune():
                raise optuna.TrialPruned
        return x + y

    sampler = optuna.samplers.BruteForceSampler()
    pruner = optuna.pruners.HyperbandPruner()
    study = optuna.create_study(sampler=sampler, pruner=pruner)
    study.optimize(objective)

    trials = study.trials
    assert len(trials) == 9
