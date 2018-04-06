import pytest
from typing import Any  # NOQA
from typing import Callable  # NOQA

from pfnopt.storages import InMemoryStorage
from pfnopt.storages import RDBStorage


@pytest.mark.parametrize('storage_init_func', [
    lambda: InMemoryStorage(),
    lambda: RDBStorage('sqlite:///:memory:')
])
def test_set_trial_user_attrs(storage_init_func):
    # type: (Callable) -> None

    storage = storage_init_func()

    trial_id_1 = storage.create_new_trial_id(storage.create_new_study_id())

    def check_set_and_get(trial_id, key, value):
        # type: (int, str, Any) -> None

        storage.set_trial_user_attr(trial_id, key, value)
        assert storage.get_trial(trial_id).user_attrs[key] == value

    # Test setting value.
    check_set_and_get(trial_id_1, 'dataset', 'MNIST')
    check_set_and_get(trial_id_1, 'none', None)
    check_set_and_get(
        trial_id_1,
        'json_serializable',
        {'baseline_score': 0.001, 'tags': ['image', 'classification']})
    assert len(storage.get_trial(trial_id_1).user_attrs) == 3

    # Test overwriting value.
    check_set_and_get(trial_id_1, 'dataset', 'ImageNet')
    assert len(storage.get_trial(trial_id_1).user_attrs) == 3

    # Test another trial.
    trial_id_2 = storage.create_new_trial_id(storage.create_new_study_id())
    check_set_and_get(trial_id_2, 'baseline_score', 0.001)
    assert len(storage.get_trial(trial_id_2).user_attrs) == 1
