"""
Optuna example that optimizes a classifier configuration for IMDB movie review dataset.
This script is based on the example of allentune (https://github.com/allenai/allentune).

In this example, we optimize the validation accuracy of
sentiment classification using AllenNLP cli.
Since it is too time-consuming to use the training dataset,
we here use the validation dataset instead.

We have the following two ways to execute this example:

(1) Execute this code directly.
    $ python allennlp_cli.py


(2) Execute through CLI.
    $ STUDY_NAME=`optuna create-study --direction maximize --storage sqlite:///example.db`
    $ optuna study optimize allennlp_cli.py objective --n-trials=100 --study $STUDY_NAME \
      --storage sqlite:///example.db

"""

import os.path
import shutil

import optuna
from optuna.integration import AllenNLPExecutor


# This path trick is used since this example is also
# run from the root of this repository by CI.
EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = "result"


def objective(trial):

    trial.suggest_uniform("DROPOUT", 0.0, 0.5)
    trial.suggest_int("EMBEDDING_DIM", 20, 50)
    trial.suggest_int("MAX_FILTER_SIZE", 3, 6)
    trial.suggest_int("NUM_FILTERS", 16, 32)
    trial.suggest_int("HIDDEN_SIZE", 16, 32)

    config_path = os.path.join(EXAMPLE_DIR, "classifier.jsonnet")
    serialization_dir = os.path.join(MODEL_DIR, "test_{}".format(trial.number))
    executor = AllenNLPExecutor(trial, config_path, serialization_dir)
    return executor.run()


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50, timeout=600)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    shutil.rmtree(MODEL_DIR)
