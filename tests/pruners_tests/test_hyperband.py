import pytest

import optuna
from optuna import type_checking

if type_checking.TYPE_CHECKING:
    from optuna.trial import Trial  # NOQA

MIN_RESOURCE = 1
REDUCTION_FACTOR = 2
EARLY_STOPPING_RATE_LOW = 0
EARLY_STOPPING_RATE_HIGH = 3
N_REPORTS = 10
EXPECTED_N_TRIALS_PER_BRACKET = 10


def test_hyperband_pruner_intermediate_values():
    # type: () -> None

    pruner = optuna.pruners.HyperbandPruner(
        min_resource=MIN_RESOURCE,
        reduction_factor=REDUCTION_FACTOR,
        min_early_stopping_rate_low=EARLY_STOPPING_RATE_LOW,
        min_early_stopping_rate_high=EARLY_STOPPING_RATE_HIGH
    )
    assert pruner._n_pruners == EARLY_STOPPING_RATE_HIGH - EARLY_STOPPING_RATE_LOW + 1
    n_pruners = pruner._n_pruners

    study = optuna.study.create_study(sampler=optuna.samplers.RandomSampler(), pruner=pruner)

    def objective(trial):
        # type: (Trial) -> float

        for i in range(N_REPORTS):
            trial.report(i)

        return 1.0

    study.optimize(objective, n_trials=n_pruners * EXPECTED_N_TRIALS_PER_BRACKET)

    trials = study.trials
    assert len(trials) == n_pruners * EXPECTED_N_TRIALS_PER_BRACKET


def test_bracket_study():
    # type: () -> None

    pruner = optuna.pruners.HyperbandPruner(
        min_resource=MIN_RESOURCE,
        reduction_factor=REDUCTION_FACTOR,
        min_early_stopping_rate_low=EARLY_STOPPING_RATE_LOW,
        min_early_stopping_rate_high=EARLY_STOPPING_RATE_HIGH
    )
    study = optuna.study.create_study(pruner=pruner)
    bracket_study = pruner._create_bracket_study(study, 0)

    with pytest.raises(Exception):
        bracket_study.optimize(lambda *args: 1.0)
