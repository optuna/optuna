import datetime

import pytest

from optuna.distributions import UniformDistribution
from optuna.structs import FrozenTrial
from optuna.structs import StudyDirection
from optuna.structs import StudySummary
from optuna.structs import TrialState


def test_frozen_trial_deprecated():
    # type: () -> None

    with pytest.deprecated_call():
        FrozenTrial(
            number=0,
            trial_id=0,
            state=TrialState.COMPLETE,
            value=0.2,
            datetime_start=datetime.datetime.now(),
            datetime_complete=datetime.datetime.now(),
            params={"x": 10},
            distributions={"x": UniformDistribution(5, 12)},
            user_attrs={},
            system_attrs={},
            intermediate_values={},
        )


def test_study_summary_deprecated():
    # type: () -> None

    with pytest.deprecated_call():
        StudySummary(
            study_name="test",
            direction=StudyDirection.NOT_SET,
            best_trial=None,
            user_attrs={},
            system_attrs={},
            n_trials=0,
            datetime_start=datetime.datetime.now(),
            study_id=0,
        )
