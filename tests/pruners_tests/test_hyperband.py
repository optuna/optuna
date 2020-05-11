import datetime
import pytest
from unittest import mock

import optuna
from optuna import type_checking

if type_checking.TYPE_CHECKING:
    from optuna.trial import Trial  # NOQA

MIN_RESOURCE = 1
MAX_RESOURCE = 16
REDUCTION_FACTOR = 2
N_BRACKETS = 4
EARLY_STOPPING_RATE_LOW = 0
EARLY_STOPPING_RATE_HIGH = 3
N_REPORTS = 10
EXPECTED_N_TRIALS_PER_BRACKET = 10


def test_hyperband_experimental_warning() -> None:

    with pytest.warns(optuna.exceptions.ExperimentalWarning):
        optuna.pruners.HyperbandPruner(
            min_resource=MIN_RESOURCE, max_resource=MAX_RESOURCE, reduction_factor=REDUCTION_FACTOR
        )


def test_hyperband_deprecation_warning_n_brackets() -> None:
    with pytest.deprecated_call():
        optuna.pruners.HyperbandPruner(
            min_resource=MIN_RESOURCE,
            max_resource=MAX_RESOURCE,
            reduction_factor=REDUCTION_FACTOR,
            n_brackets=N_BRACKETS,
        )


def test_hyperband_deprecation_warning_min_early_stopping_rate_low() -> None:
    with pytest.deprecated_call():
        optuna.pruners.HyperbandPruner(
            min_resource=MIN_RESOURCE,
            max_resource=MAX_RESOURCE,
            reduction_factor=REDUCTION_FACTOR,
            min_early_stopping_rate_low=EARLY_STOPPING_RATE_LOW,
        )


def test_hyperband_pruner_intermediate_values():
    # type: () -> None

    pruner = optuna.pruners.HyperbandPruner(
        min_resource=MIN_RESOURCE, max_resource=MAX_RESOURCE, reduction_factor=REDUCTION_FACTOR
    )

    study = optuna.study.create_study(sampler=optuna.samplers.RandomSampler(), pruner=pruner)

    def objective(trial):
        # type: (Trial) -> float

        for i in range(N_REPORTS):
            trial.report(i, step=i)

        return 1.0

    study.optimize(objective, n_trials=N_BRACKETS * EXPECTED_N_TRIALS_PER_BRACKET)

    trials = study.trials
    assert len(trials) == N_BRACKETS * EXPECTED_N_TRIALS_PER_BRACKET


def test_bracket_study():
    # type: () -> None

    pruner = optuna.pruners.HyperbandPruner(
        min_resource=MIN_RESOURCE, max_resource=MAX_RESOURCE, reduction_factor=REDUCTION_FACTOR
    )
    study = optuna.study.create_study(sampler=optuna.samplers.RandomSampler(), pruner=pruner)
    bracket_study = pruner._create_bracket_study(study, 0)

    with pytest.raises(AttributeError):
        bracket_study.optimize(lambda *args: 1.0)

    for attr in ("set_user_attr", "set_system_attr"):
        with pytest.raises(AttributeError):
            getattr(bracket_study, attr)("abc", 100)

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


def test_hyperband_max_resource_is_auto():
    # type: () -> None

    pruner = optuna.pruners.HyperbandPruner(
        min_resource=MIN_RESOURCE, reduction_factor=REDUCTION_FACTOR
    )
    study = optuna.study.create_study(sampler=optuna.samplers.RandomSampler(), pruner=pruner)

    def objective(trial):
        # type: (Trial) -> float

        for i in range(N_REPORTS):
            trial.report(1.0, i)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return 1.0

    study.optimize(objective, n_trials=N_BRACKETS * EXPECTED_N_TRIALS_PER_BRACKET)

    assert N_REPORTS == pruner._max_resource


def test_hyperband_max_resource_value_error():
    # type: () -> None

    with pytest.raises(ValueError):
        _ = optuna.pruners.HyperbandPruner(max_resource="not_appropriate")


def test_hyperband_filter_study():
    # type: () -> None

    pruner = optuna.pruners.HyperbandPruner(
        min_resource=MIN_RESOURCE, max_resource=MAX_RESOURCE, reduction_factor=REDUCTION_FACTOR
    )

    def objective(t: optuna.trial.Trial) -> float:
        return 1.0

    def side_effect(s: optuna.study.Study, t: optuna.trial.FrozenTrial) -> int:
        if t.number % 2:
            return 0
        else:
            return 1

    with mock.patch(
        "optuna.pruners.HyperbandPruner._get_bracket_id",
        new=mock.MagicMock(side_effect=side_effect),
    ):
        study = optuna.study.create_study(pruner=pruner)
        study.optimize(objective, n_trials=10)

        trial = optuna.trial.FrozenTrial(
            number=10,
            trial_id=10,
            state=optuna.trial.TrialState.COMPLETE,
            value=0,
            datetime_start=datetime.datetime.now(),
            datetime_complete=datetime.datetime.now(),
            params={},
            distributions={},
            user_attrs={},
            system_attrs={},
            intermediate_values={},
        )
        filtered_study = optuna.pruners._filter_study(study, trial)

        filtered_trials = filtered_study.get_trials(deepcopy=False)
        assert len(filtered_trials) == 5
        assert isinstance(study.pruner, optuna.pruners.HyperbandPruner)
        assert study.pruner._get_bracket_id(study, trial) == 1
        for t in filtered_trials:
            assert study.pruner._get_bracket_id(study, t) == 1
