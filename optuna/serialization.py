import pickle

import optuna
from optuna._experimental import experimental


@experimental("2.8.0")
def save_study(study: optuna.study.Study, path: str) -> None:
    tmp_study = optuna.create_study(directions=study.directions)

    _copy_study(study, tmp_study)

    with open(path, "wb") as f:
        pickle.dump(tmp_study, f)


@experimental("2.8.0")
def load_study(study: optuna.study.Study, path: str) -> None:
    with open(path, "rb") as f:
        tmp_study = pickle.load(f)

    _copy_study(tmp_study, study)


def _copy_study(src: optuna.study.Study, dst: optuna.study.Study) -> None:
    _copy_study_system_attrs(src, dst)
    _copy_study_user_attrs(src, dst)
    _copy_study_trials(src, dst)


def _copy_study_system_attrs(src: optuna.study.Study, dst: optuna.study.Study) -> None:
    for key, value in src.system_attrs.items():
        dst.set_system_attr(key, value)


def _copy_study_user_attrs(src: optuna.study.Study, dst: optuna.study.Study) -> None:
    for key, value in src.user_attrs.items():
        dst.set_user_attr(key, value)


def _copy_study_trials(src: optuna.study.Study, dst: optuna.study.Study) -> None:
    for trial in src.trials:
        dst.add_trial(trial)
