from __future__ import annotations

import pytest

import optuna


def test_wilcoxon_pruner_constructor() -> None:
    optuna.pruners.WilcoxonPruner()
    optuna.pruners.WilcoxonPruner(p_threshold=0)
    optuna.pruners.WilcoxonPruner(p_threshold=1)
    optuna.pruners.WilcoxonPruner(p_threshold=0.05)
    optuna.pruners.WilcoxonPruner(n_startup_steps=5)

    with pytest.raises(ValueError):
        optuna.pruners.WilcoxonPruner(p_threshold=-0.1)

    with pytest.raises(ValueError):
        optuna.pruners.WilcoxonPruner(n_startup_steps=-5)


def test_wilcoxon_pruner_first_trial() -> None:
    # A pruner is not activated at a first trial.
    pruner = optuna.pruners.WilcoxonPruner()
    study = optuna.study.create_study(pruner=pruner)
    trial = study.ask()
    assert not trial.should_prune()
    trial.report(1, 1)
    assert not trial.should_prune()
    trial.report(2, 2)
    assert not trial.should_prune()


@pytest.mark.parametrize(
    "p_threshold,step_values,expected_should_prune",
    [
        (0.2, [-1, 1, 2, 3, 4, 5, -2, -3], [False, False, False, True, True, True, True, True]),
        (0.15, [-1, 1, 2, 3, 4, 5, -2, -3], [False, False, False, False, True, True, True, False]),
    ],
)
def test_wilcoxon_pruner_normal(
    p_threshold: float,
    step_values: list[float],
    expected_should_prune: list[bool],
) -> None:
    pruner = optuna.pruners.WilcoxonPruner(n_startup_steps=0, p_threshold=p_threshold)
    study = optuna.study.create_study(pruner=pruner)

    # Insert the best trial
    study.add_trial(
        optuna.trial.create_trial(
            value=0,
            params={},
            distributions={},
            intermediate_values={step: 0 for step in range(10)},
        )
    )

    trial = study.ask()

    should_prune = [False] * len(step_values)

    for step_i in range(len(step_values)):
        trial.report(step_values[step_i], step_i)
        should_prune[step_i] = trial.should_prune()

    assert should_prune == expected_should_prune


@pytest.mark.parametrize(
    "best_intermediate_values,intermediate_values",
    [
        ({1: 1}, {1: 1, 2: 2}),  # Current trial has more steps than best trial
        ({1: 1}, {1: float("nan")}),  # NaN value
        ({1: float("nan")}, {1: 1}),  # NaN value
        ({1: 1}, {1: float("inf")}),  # NaN value
        ({1: float("inf")}, {1: 1}),  # NaN value
    ],
)
def test_wilcoxon_pruner_warn_bad_best_trial(
    best_intermediate_values: dict[int, float],
    intermediate_values: dict[int, float],
) -> None:
    pruner = optuna.pruners.WilcoxonPruner()
    study = optuna.study.create_study(pruner=pruner)

    # Insert best trial
    study.add_trial(
        optuna.trial.create_trial(
            value=0, params={}, distributions={}, intermediate_values=best_intermediate_values
        )
    )
    trial = study.ask()
    with pytest.warns(UserWarning):
        for step, value in intermediate_values.items():
            trial.report(value, step)
            trial.should_prune()
