import itertools
import numpy as np
import pytest

import optuna
from optuna import samplers

if optuna.type_checking.TYPE_CHECKING:
    from optuna.trial import Trial  # NOQA


def test_study_optimize():
    # type: () -> None

    def objective(trial):
        # type: (Trial) -> float

        a = trial.suggest_int('a', 0, 100)
        b = trial.suggest_uniform('b', -0.1, 0.1)
        c = trial.suggest_categorical('c', ('x', 'y'))
        d = trial.suggest_discrete_uniform('d', -5, 5, 1)
        e = trial.suggest_loguniform('e', 0.0001, 1)

        if c == 'x':
            return a * d
        else:
            return b * e

    # Test that all combinations of the grid is sampled.
    grid = {
        'a': list(range(0, 100, 20)),
        'b': np.arange(-0.1, 0.1, 0.05),
        'c': ['x', 'y'],
        'd': [-0.5, 0.5],
        'e': [0.1]
    }
    n_trials = int(np.prod([len(v) for v in grid.values()]))
    study = optuna.create_study(sampler=samplers.GridSampler(grid))
    study.optimize(objective, n_trials=n_trials)

    grid_product = itertools.product(*grid.values())
    all_suggested_values = [tuple([p for p in t.params.values()]) for t in study.trials]
    assert set(grid_product) == set(all_suggested_values)

    ids = sorted([t.system_attrs['grid_id'] for t in study.trials])
    assert ids == list(range(n_trials))

    # Test a non-existing parameter name in the grid.
    grid = {'a': list(range(0, 100, 20))}
    study = optuna.create_study(sampler=samplers.GridSampler(grid))
    with pytest.raises(ValueError):
        study.optimize(objective)

    # Test an value with out of range.
    grid = {
        'a': [110],  # 110 is out of range specified by the suggest method.
        'b': [0],
        'c': ['x'],
        'd': [0],
        'e': [0.1]
    }
    study = optuna.create_study(sampler=samplers.GridSampler(grid))
    with pytest.raises(ValueError):
        study.optimize(objective)
