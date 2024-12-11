from __future__ import annotations

import pytest

import optuna


@pytest.mark.parametrize("direction_value", [("minimize", 2), ("maximize", 0.5)])
def test_successive_halving_pruner_intermediate_values(direction_value: tuple[str, float]) -> None:
    direction, intermediate_value = direction_value
    pruner = optuna.pruners.SuccessiveHalvingPruner(
        min_resource=1, reduction_factor=2, min_early_stopping_rate=0
    )
    study = optuna.study.create_study(direction=direction, pruner=pruner)

    trial = study.ask()
    trial.report(1, 1)

    # A pruner is not activated at a first trial.
    assert not trial.should_prune()

    trial = study.ask()
    # A pruner is not activated if a trial has no intermediate values.
    assert not trial.should_prune()

    trial.report(intermediate_value, 1)
    # A pruner is activated if a trial has an intermediate value.
    assert trial.should_prune()


def test_successive_halving_pruner_rung_check() -> None:
    pruner = optuna.pruners.SuccessiveHalvingPruner(
        min_resource=1, reduction_factor=2, min_early_stopping_rate=0
    )
    study = optuna.study.create_study(pruner=pruner)

    # Report 7 trials in advance.
    for i in range(7):
        trial = study.ask()
        trial.report(0.1 * (i + 1), step=7)
        pruner.prune(study=study, trial=study._storage.get_trial(trial._trial_id))

    # Report a trial that has the 7-th value from bottom.
    trial = study.ask()
    trial.report(0.75, step=7)
    pruner.prune(study=study, trial=study._storage.get_trial(trial._trial_id))

    trial_system_attrs = trial.storage.get_trial_system_attrs(trial._trial_id)
    assert "completed_rung_0" in trial_system_attrs
    assert "completed_rung_1" not in trial_system_attrs

    # Report a trial that has the third value from bottom.
    trial = study.ask()
    trial.report(0.25, step=7)
    trial.should_prune()
    trial_system_attrs = trial.storage.get_trial_system_attrs(trial._trial_id)
    assert "completed_rung_1" in trial_system_attrs
    assert "completed_rung_2" not in trial_system_attrs

    # Report a trial that has the lowest value.
    trial = study.ask()
    trial.report(0.05, step=7)
    trial.should_prune()
    trial_system_attrs = trial.storage.get_trial_system_attrs(trial._trial_id)
    assert "completed_rung_2" in trial_system_attrs
    assert "completed_rung_3" not in trial_system_attrs


def test_successive_halving_pruner_first_trial_is_not_pruned() -> None:
    pruner = optuna.pruners.SuccessiveHalvingPruner(
        min_resource=1, reduction_factor=2, min_early_stopping_rate=0
    )
    study = optuna.study.create_study(pruner=pruner)
    trial = study.ask()

    for i in range(10):
        trial.report(1, step=i)

        # The first trial is not pruned.
        assert not trial.should_prune()

    # The trial completed until rung 3.
    trial_system_attrs = trial.storage.get_trial_system_attrs(trial._trial_id)
    assert "completed_rung_0" in trial_system_attrs
    assert "completed_rung_1" in trial_system_attrs
    assert "completed_rung_2" in trial_system_attrs
    assert "completed_rung_3" in trial_system_attrs
    assert "completed_rung_4" not in trial_system_attrs


def test_successive_halving_pruner_with_nan() -> None:
    pruner = optuna.pruners.SuccessiveHalvingPruner(
        min_resource=2, reduction_factor=2, min_early_stopping_rate=0
    )
    study = optuna.study.create_study(pruner=pruner)
    trial = study.ask()

    # A pruner is not activated if the step is not a rung completion point.
    trial.report(float("nan"), step=1)
    assert not trial.should_prune()

    # A pruner is activated if the step is a rung completion point and
    # the intermediate value is NaN.
    trial.report(float("nan"), step=2)
    assert trial.should_prune()


@pytest.mark.parametrize("n_reports", range(3))
@pytest.mark.parametrize("n_trials", [1, 2])
def test_successive_halving_pruner_with_auto_min_resource(n_reports: int, n_trials: int) -> None:
    pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource="auto")
    study = optuna.study.create_study(sampler=optuna.samplers.RandomSampler(), pruner=pruner)

    assert pruner._min_resource is None

    def objective(trial: optuna.trial.Trial) -> float:
        for i in range(n_reports):
            trial.report(1.0 / (i + 1), i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return 1.0

    study.optimize(objective, n_trials=n_trials)
    if n_reports > 0 and n_trials > 1:
        assert pruner._min_resource is not None and pruner._min_resource > 0
    else:
        assert pruner._min_resource is None


def test_successive_halving_pruner_with_invalid_str_to_min_resource() -> None:
    with pytest.raises(ValueError):
        optuna.pruners.SuccessiveHalvingPruner(min_resource="fixed")


def test_successive_halving_pruner_min_resource_parameter() -> None:
    # min_resource=0: Error (must be `min_resource >= 1`).
    with pytest.raises(ValueError):
        optuna.pruners.SuccessiveHalvingPruner(
            min_resource=0, reduction_factor=2, min_early_stopping_rate=0
        )

    # min_resource=1: The rung 0 ends at step 1.
    pruner = optuna.pruners.SuccessiveHalvingPruner(
        min_resource=1, reduction_factor=2, min_early_stopping_rate=0
    )
    study = optuna.study.create_study(pruner=pruner)
    trial = study.ask()

    trial.report(1, step=1)
    assert not trial.should_prune()
    trial_system_attrs = trial.storage.get_trial_system_attrs(trial._trial_id)
    assert "completed_rung_0" in trial_system_attrs
    assert "completed_rung_1" not in trial_system_attrs

    # min_resource=2: The rung 0 ends at step 2.
    pruner = optuna.pruners.SuccessiveHalvingPruner(
        min_resource=2, reduction_factor=2, min_early_stopping_rate=0
    )
    study = optuna.study.create_study(pruner=pruner)
    trial = study.ask()

    trial.report(1, step=1)
    assert not trial.should_prune()
    trial_system_attrs = trial.storage.get_trial_system_attrs(trial._trial_id)
    assert "completed_rung_0" not in trial_system_attrs

    trial.report(1, step=2)
    assert not trial.should_prune()
    trial_system_attrs = trial.storage.get_trial_system_attrs(trial._trial_id)
    assert "completed_rung_0" in trial_system_attrs
    assert "completed_rung_1" not in trial_system_attrs


def test_successive_halving_pruner_reduction_factor_parameter() -> None:
    # reduction_factor=1: Error (must be `reduction_factor >= 2`).
    with pytest.raises(ValueError):
        optuna.pruners.SuccessiveHalvingPruner(
            min_resource=1, reduction_factor=1, min_early_stopping_rate=0
        )

    # reduction_factor=2: The rung 0 ends at step 1.
    pruner = optuna.pruners.SuccessiveHalvingPruner(
        min_resource=1, reduction_factor=2, min_early_stopping_rate=0
    )
    study = optuna.study.create_study(pruner=pruner)
    trial = study.ask()

    trial.report(1, step=1)
    assert not trial.should_prune()
    trial_system_attrs = trial.storage.get_trial_system_attrs(trial._trial_id)
    assert "completed_rung_0" in trial_system_attrs
    assert "completed_rung_1" not in trial_system_attrs

    # reduction_factor=3: The rung 1 ends at step 3.
    pruner = optuna.pruners.SuccessiveHalvingPruner(
        min_resource=1, reduction_factor=3, min_early_stopping_rate=0
    )
    study = optuna.study.create_study(pruner=pruner)
    trial = study.ask()

    trial.report(1, step=1)
    assert not trial.should_prune()
    trial_system_attrs = trial.storage.get_trial_system_attrs(trial._trial_id)
    assert "completed_rung_0" in trial_system_attrs
    assert "completed_rung_1" not in trial_system_attrs

    trial.report(1, step=2)
    assert not trial.should_prune()
    trial_system_attrs = trial.storage.get_trial_system_attrs(trial._trial_id)
    assert "completed_rung_1" not in trial_system_attrs

    trial.report(1, step=3)
    assert not trial.should_prune()
    trial_system_attrs = trial.storage.get_trial_system_attrs(trial._trial_id)
    assert "completed_rung_1" in trial_system_attrs
    assert "completed_rung_2" not in trial_system_attrs


def test_successive_halving_pruner_min_early_stopping_rate_parameter() -> None:
    # min_early_stopping_rate=-1: Error (must be `min_early_stopping_rate >= 0`).
    with pytest.raises(ValueError):
        optuna.pruners.SuccessiveHalvingPruner(
            min_resource=1, reduction_factor=2, min_early_stopping_rate=-1
        )

    # min_early_stopping_rate=0: The rung 0 ends at step 1.
    pruner = optuna.pruners.SuccessiveHalvingPruner(
        min_resource=1, reduction_factor=2, min_early_stopping_rate=0
    )
    study = optuna.study.create_study(pruner=pruner)
    trial = study.ask()

    trial.report(1, step=1)
    assert not trial.should_prune()
    trial_system_attrs = trial.storage.get_trial_system_attrs(trial._trial_id)
    assert "completed_rung_0" in trial_system_attrs

    # min_early_stopping_rate=1: The rung 0 ends at step 2.
    pruner = optuna.pruners.SuccessiveHalvingPruner(
        min_resource=1, reduction_factor=2, min_early_stopping_rate=1
    )
    study = optuna.study.create_study(pruner=pruner)
    trial = study.ask()

    trial.report(1, step=1)
    assert not trial.should_prune()
    trial_system_attrs = trial.storage.get_trial_system_attrs(trial._trial_id)
    assert "completed_rung_0" not in trial_system_attrs
    assert "completed_rung_1" not in trial_system_attrs

    trial.report(1, step=2)
    assert not trial.should_prune()
    trial_system_attrs = trial.storage.get_trial_system_attrs(trial._trial_id)
    assert "completed_rung_0" in trial_system_attrs
    assert "completed_rung_1" not in trial_system_attrs


def test_successive_halving_pruner_bootstrap_parameter() -> None:
    with pytest.raises(ValueError):
        optuna.pruners.SuccessiveHalvingPruner(bootstrap_count=-1)

    with pytest.raises(ValueError):
        optuna.pruners.SuccessiveHalvingPruner(bootstrap_count=1, min_resource="auto")

    pruner = optuna.pruners.SuccessiveHalvingPruner(
        min_resource=1, reduction_factor=2, bootstrap_count=1
    )
    study = optuna.study.create_study(pruner=pruner)

    trial1 = study.ask()
    trial2 = study.ask()

    trial1.report(1, step=1)
    assert trial1.should_prune()

    trial2.report(1, step=1)
    assert not trial2.should_prune()
