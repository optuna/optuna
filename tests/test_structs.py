import copy
import datetime
import warnings

import pytest

import optuna
from optuna.distributions import LogUniformDistribution
from optuna.distributions import UniformDistribution
from optuna.structs import FrozenTrial
from optuna.structs import TrialState

if optuna.type_checking.TYPE_CHECKING:
    from typing import Any  # NOQA
    from typing import Dict  # NOQA
    from typing import List  # NOQA
    from typing import Tuple  # NOQA

    from optuna.distributions import BaseDistribution  # NOQA


def test_frozen_trial_validate():
    # type: () -> None

    # Valid.
    valid_trial = _create_frozen_trial()
    valid_trial._validate()

    # Invalid: `datetime_start` is not set.
    invalid_trial = copy.copy(valid_trial)
    invalid_trial.datetime_start = None
    with pytest.raises(ValueError):
        invalid_trial._validate()

    # Invalid: `state` is `RUNNING` and `datetime_complete` is set.
    invalid_trial = copy.copy(valid_trial)
    invalid_trial.state = TrialState.RUNNING
    with pytest.raises(ValueError):
        invalid_trial._validate()

    # Invalid: `state` is not `RUNNING` and `datetime_complete` is not set.
    for state in [TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL]:
        invalid_trial = copy.copy(valid_trial)
        invalid_trial.state = state
        invalid_trial.datetime_complete = None
        with pytest.raises(ValueError):
            invalid_trial._validate()

    # Invalid: `state` is `COMPLETE` and `value` is not set.
    invalid_trial = copy.copy(valid_trial)
    invalid_trial.value = None
    with pytest.raises(ValueError):
        invalid_trial._validate()

    # Invalid: Inconsistent `params` and `distributions`
    inconsistent_pairs = [
        # `params` has an extra element.
        ({
            'x': 0.1,
            'y': 0.5
        }, {
            'x': UniformDistribution(0, 1)
        }),

        # `distributions` has an extra element.
        ({
            'x': 0.1
        }, {
            'x': UniformDistribution(0, 1),
            'y': LogUniformDistribution(0, 1)
        }),

        # The value of `x` isn't contained in the distribution.
        ({
            'x': -0.5
        }, {
            'x': UniformDistribution(0, 1)
        })
    ]  # type: List[Tuple[Dict[str, Any], Dict[str, BaseDistribution]]]

    for params, distributions in inconsistent_pairs:
        invalid_trial = copy.copy(valid_trial)
        invalid_trial.params = params
        invalid_trial.distributions = distributions
        with pytest.raises(ValueError):
            invalid_trial._validate()


def test_frozen_trial_eq_ne():
    # type: () -> None

    trial = _create_frozen_trial()

    trial_other = copy.copy(trial)
    assert trial == trial_other

    trial_other.value = 0.3
    assert trial != trial_other


def test_frozen_trial_lt():
    # type: () -> None

    trial = _create_frozen_trial()

    trial_other = copy.copy(trial)
    assert not trial < trial_other

    trial_other.number = trial.number + 1
    assert trial < trial_other
    assert not trial_other < trial

    with pytest.raises(TypeError):
        trial < 1

    assert trial <= trial_other
    assert not trial_other <= trial

    with pytest.raises(TypeError):
        trial <= 1

    # A list of FrozenTrials is sortable.
    trials = [trial_other, trial]
    trials.sort()
    assert trials[0] is trial
    assert trials[1] is trial_other


def _create_frozen_trial():
    # type: () -> FrozenTrial

    return FrozenTrial(number=0,
                       trial_id=0,
                       state=TrialState.COMPLETE,
                       value=0.2,
                       datetime_start=datetime.datetime.now(),
                       datetime_complete=datetime.datetime.now(),
                       params={'x': 10},
                       distributions={'x': UniformDistribution(5, 12)},
                       user_attrs={},
                       system_attrs={},
                       intermediate_values={})


def test_frozen_trial_repr():
    # type: () -> None

    trial = FrozenTrial(number=0,
                        trial_id=0,
                        state=TrialState.COMPLETE,
                        value=0.2,
                        datetime_start=datetime.datetime.now(),
                        datetime_complete=datetime.datetime.now(),
                        params={'x': 10},
                        distributions={'x': UniformDistribution(5, 12)},
                        user_attrs={},
                        system_attrs={},
                        intermediate_values={})

    assert trial == eval(repr(trial))


def test_study_summary_study_id():
    # type: () -> None

    study = optuna.create_study()
    summaries = study._storage.get_all_study_summaries()
    assert len(summaries) == 1

    summary = summaries[0]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=DeprecationWarning)
        assert summary.study_id == summary._study_id

    with pytest.warns(DeprecationWarning):
        summary.study_id


def test_study_summary_eq_ne():
    # type: () -> None

    storage = optuna.storages.RDBStorage('sqlite:///:memory:')

    optuna.create_study(storage=storage)
    study = optuna.create_study(storage=storage)

    summaries = study._storage.get_all_study_summaries()
    assert len(summaries) == 2

    assert summaries[0] == copy.deepcopy(summaries[0])
    assert not summaries[0] != copy.deepcopy(summaries[0])

    assert not summaries[0] == summaries[1]
    assert summaries[0] != summaries[1]

    assert not summaries[0] == 1
    assert summaries[0] != 1


def test_study_summary_lt_le():
    # type: () -> None

    storage = optuna.storages.RDBStorage('sqlite:///:memory:')

    optuna.create_study(storage=storage)
    study = optuna.create_study(storage=storage)

    summaries = study._storage.get_all_study_summaries()
    assert len(summaries) == 2

    summary_0 = summaries[0]
    summary_1 = summaries[1]

    assert summary_0 < summary_1
    assert not summary_1 < summary_0

    with pytest.raises(TypeError):
        summary_0 < 1

    assert summary_0 <= summary_0
    assert not summary_1 <= summary_0

    with pytest.raises(TypeError):
        summary_0 <= 1

    # A list of StudySummaries is sortable.
    summaries.reverse()
    summaries.sort()
    assert summaries[0] == summary_0
    assert summaries[1] == summary_1
