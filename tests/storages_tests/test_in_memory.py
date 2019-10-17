import pytest

import optuna
from optuna.storages import InMemoryStorage


@pytest.mark.parametrize('study_direction', ['minimize', 'maximize'])
def test_update_cache(study_direction):
    # type: (str) -> None

    storage = InMemoryStorage()

    assert storage.best_trial_id is None

    # If the direction is 'minimize', the objective function is weakly decreasing.
    # Otherwise, it is weakly increasing.
    sign = -1 if study_direction == 'minimize' else 1

    study = optuna.create_study(storage=storage, direction=study_direction)
    study.optimize(lambda x: sign * x.number, n_trials=1)
    storage._update_cache(0)
    assert storage.best_trial_id == 0

    study.optimize(lambda x: sign * x.number, n_trials=1)
    storage._update_cache(1)
    assert storage.best_trial_id == 1

    # The objective value is equal to the best value.
    study.optimize(lambda x: sign * (x.number - 1), n_trials=1)
    storage._update_cache(2)
    assert storage.best_trial_id == 1

    # The objective value is inferior to the best value.
    study.optimize(lambda x: sign * (x.number - 2), n_trials=1)
    storage._update_cache(3)
    assert storage.best_trial_id == 1


def test_update_cache_none_value():
    # type: () -> None

    storage = InMemoryStorage()

    study = optuna.create_study(storage=storage)
    study.optimize(lambda x: -1 * x.number, n_trials=1)
    storage._update_cache(0)
    assert storage.best_trial_id == 0

    # The objective value is None.
    study.optimize(lambda x: None, n_trials=1)  # type: ignore
    storage._update_cache(1)
    assert storage.best_trial_id == 0
