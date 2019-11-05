import numpy as np
import pytest

import optuna
from optuna import samplers
from optuna.distributions import CategoricalDistribution
from optuna.distributions import DiscreteUniformDistribution
from optuna.distributions import IntUniformDistribution
from optuna.distributions import LogUniformDistribution
from optuna.distributions import UniformDistribution
from optuna.samplers import BaseSampler

if optuna.type_checking.TYPE_CHECKING:
    import typing  # NOQA
    from typing import Any  # NOQA
    from typing import Dict  # NOQA

    from optuna.distributions import BaseDistribution  # NOQA
    from optuna.structs import FrozenTrial  # NOQA
    from optuna.study import Study  # NOQA
    from optuna.trial import T  # NOQA
    from optuna.trial import Trial  # NOQA


def test_study_optimize():
    # type: () -> None

    def objective(trial):
        # type: (Trial) -> float

        a = trial.suggest_int('a', 0, 100)
        b = trial.suggest_uniform('b', -0.1, 0.1)
        c = trial.suggest_categorical('c', ('x', 'y'))

        if c == 'x':
            return a
        else:
            return b

    grid = {
        'a': list(range(0, 100, 20)),
        'b': np.arange(-0.1, 0.1, 0.02),
        'c': ['x', 'y'],
    }
    sampler = samplers.GridSampler(grid)
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=5 * 10 * 2)

    ids = sorted([t.system_attrs['grid_id'] for t in study.trials])
    assert ids == list(range(100))
