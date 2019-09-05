from datetime import datetime
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
    valid_trial = FrozenTrial(number=0,
                              trial_id=0,
                              state=TrialState.COMPLETE,
                              value=0.2,
                              datetime_start=datetime.now(),
                              datetime_complete=datetime.now(),
                              params={'x': 10},
                              distributions={'x': UniformDistribution(5, 12)},
                              user_attrs={},
                              system_attrs={},
                              intermediate_values={})
    valid_trial._validate()

    # Invalid: `datetime_start` is not set.
    invalid_trial = valid_trial._replace(datetime_start=None)
    with pytest.raises(ValueError):
        invalid_trial._validate()

    # Invalid: `state` is `RUNNING` and `datetime_complete` is set.
    invalid_trial = valid_trial._replace(state=TrialState.RUNNING)
    with pytest.raises(ValueError):
        invalid_trial._validate()

    # Invalid: `state` is not `RUNNING` and `datetime_complete` is not set.
    for state in [TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL]:
        invalid_trial = valid_trial._replace(state=state, datetime_complete=None)
        with pytest.raises(ValueError):
            invalid_trial._validate()

    # Invalid: `state` is `COMPLETE` and `value` is not set.
    invalid_trial = valid_trial._replace(value=None)
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
        invalid_trial = valid_trial._replace(params=params, distributions=distributions)
        with pytest.raises(ValueError):
            invalid_trial._validate()
