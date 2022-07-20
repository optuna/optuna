import os

from optuna.distributions import FloatDistribution
from optuna.storages._journal.storage import JournalStorage
from optuna.study._study_direction import StudyDirection


def cleanup():
    os.system("rm -rf ./openlock")
    os.system("rm ./operation_logs")


def create_and_delete_test():
    cleanup()
    storage = JournalStorage("operation_logs")
    study_id = storage.create_new_study()
    storage.delete_study(study_id)


def delete_study_test():
    cleanup()
    storage = JournalStorage("operation_logs")
    storage.delete_study(0)


def set_study_attrs_test():
    cleanup()
    storage = JournalStorage("operation_logs")
    study_id = storage.create_new_study()
    storage.set_study_user_attr(study_id, "user_key", "user_val")
    storage.set_study_system_attr(study_id, "system_key", "system_val")
    storage.set_study_directions(study_id, [StudyDirection.MAXIMIZE, StudyDirection.MINIMIZE])
    storage.delete_study(study_id)


def get_study_attrs_test():
    cleanup()
    storage = JournalStorage("operation_logs")
    study_name = "test_study"
    study_id = storage.create_new_study(study_name)

    user_key = "user_key"
    user_val = "user_val"
    storage.set_study_user_attr(study_id, user_key, user_val)

    sys_key = "sys_key"
    sys_val = "sys_val"
    storage.set_study_system_attr(study_id, sys_key, sys_val)

    directions = [StudyDirection.MAXIMIZE, StudyDirection.MINIMIZE]
    storage.set_study_directions(study_id, directions)

    assert study_id == storage.get_study_id_from_name(study_name)
    assert study_name == storage.get_study_name_from_id(study_id)
    assert directions == storage.get_study_directions(study_id)
    assert user_val == storage.get_study_user_attrs(study_id)[user_key]
    assert sys_val == storage.get_study_system_attrs(study_id)[sys_key]

    study_name2 = study_name + "2"
    study_id2 = storage.create_new_study(study_name2)

    frozen_studies = storage.get_all_studies()
    assert len(frozen_studies) == 2
    frozen_study_names = [fs.study_name for fs in frozen_studies]
    assert study_name in frozen_study_names
    assert study_name2 in frozen_study_names

    storage.delete_study(study_id)
    storage.delete_study(study_id2)


def create_trial_test():
    cleanup()
    storage = JournalStorage("operation_logs")
    study_name = "create_trial_test"
    study_id = storage.create_new_study(study_name)

    trial_id = storage.create_new_trial(study_id, None)

    # Set new params.
    storage.set_trial_param(
        trial_id, "test_set_trial_param", 0.5, FloatDistribution(low=1.0, high=2.0)
    )
    storage.delete_study(study_id)


if __name__ == "__main__":
    create_and_delete_test()
    # delete_study_test()
    set_study_attrs_test()
    get_study_attrs_test()
    create_trial_test()
