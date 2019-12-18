import optuna
from optuna import type_checking

if type_checking.TYPE_CHECKING:
    from typing import Dict  # NOQA
    from typing import List  # NOQA
    from typing import Tuple  # NOQA

    from optuna.structs import FrozenTrial  # NOQA
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
    assert pruner.n_pruners == EARLY_STOPPING_RATE_HIGH - EARLY_STOPPING_RATE_LOW + 1
    n_pruners = pruner.n_pruners

    study = optuna.study.create_study(pruner=pruner)

    def objective(trial):
        # type: (Trial) -> float

        for i in range(N_REPORTS):
            trial.report(i)

        return 1.0

    study.optimize(objective, n_trials=n_pruners * EXPECTED_N_TRIALS_PER_BRACKET)

    trials = study.trials
    bracket_user_attrs = [pruner.__class__.__name__ + '_{}'.format(i) for i in range(n_pruners)]

    assert len(trials) == n_pruners * EXPECTED_N_TRIALS_PER_BRACKET

    bracket_trials = {
        key: [t for t in trials if t.user_attrs['pruner_metadata'] == key]
        for key in bracket_user_attrs
    }  # type: Dict[str, List[FrozenTrial]]

    for key, value in bracket_trials.items():
        assert len(value) > 0
