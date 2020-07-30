"""
Optuna example that optimizes a classifier configuration for IMDB movie review dataset.
This script is based on the example of allentune (https://github.com/allenai/allentune).

In this example, we optimize the validation accuracy of
sentiment classification using an AllenNLP jsonnet config file.
Since it is too time-consuming to use the training dataset,	we here use the validation dataset instead.

"""

import os.path
import pkg_resources
import shutil

import allennlp

import optuna
from optuna.integration.allennlp import dump_best_config
from optuna.integration import AllenNLPExecutor


# This path trick is used since this example is also
# run from the root of this repository by CI.
EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(EXAMPLE_DIR, "classifier.jsonnet")
MODEL_DIR = "result"
BEST_CONFIG_PATH = "best_classifier.json"


def objective(trial):
    trial.suggest_float("DROPOUT", 0.0, 0.5)
    trial.suggest_int("EMBEDDING_DIM", 20, 50)
    trial.suggest_int("MAX_FILTER_SIZE", 3, 6)
    trial.suggest_int("NUM_FILTERS", 16, 32)
    trial.suggest_int("HIDDEN_SIZE", 16, 32)

    serialization_dir = os.path.join(MODEL_DIR, "test_{}".format(trial.number))
    executor = AllenNLPExecutor(trial, CONFIG_PATH, serialization_dir)

    return executor.run()


if __name__ == "__main__":
    if pkg_resources.parse_version(allennlp.__version__) < pkg_resources.parse_version("1.0.0"):
        raise RuntimeError("AllenNLP>=1.0.0 is required for this example.")

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50, timeout=600)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    dump_best_config(CONFIG_PATH, BEST_CONFIG_PATH, study)
    print("\nCreated optimized AllenNLP config to `{}`.".format(BEST_CONFIG_PATH))

    shutil.rmtree(MODEL_DIR)
