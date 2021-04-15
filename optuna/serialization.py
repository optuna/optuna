import pickle

import optuna


def save_study(study: optuna.study.Study, path: str) -> None:
    study_in_memory = optuna.create_study(directions=study.directions)
    for trial in study.trials:
        study_in_memory.add_trial(trial)

    with open(path, "wb") as f:
        pickle.dump(study_in_memory, f)


def load_study(study: optuna.study.Study, path: str) -> None:
    with open(path, "rb") as f:
        study_in_memory = pickle.load(f)

    for trial in study_in_memory.trials:
        study.add_trial(trial)
