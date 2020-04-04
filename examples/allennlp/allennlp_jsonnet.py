"""
Optuna example that optimizes a classifier configuration for IMDB movie review dataset.
This script is based on the example of allentune (https://github.com/allenai/allentune).

In this example, we optimize the validation accuracy of
sentiment classification using AllenNLP cli.
Since it is too time-consuming to use the entire dataset, we here use a small subset of it.

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


MODEL_DIR = "result"


def objective(trial):
    trial.suggest_uniform("LEARNING_RATE", 1e-2, 1e-1)
    trial.suggest_uniform("DROPOUT", 0.0, 0.5)
    trial.suggest_int("MAX_FILTER_SIZE", 3, 6)
    trial.suggest_int("NUM_FILTERS", 16, 128)
    trial.suggest_int("NUM_OUTPUT_LAYERS", 1, 3)
    trial.suggest_int("HIDDEN_SIZE", 16, 128)

    serialization_dir = os.path.join(MODEL_DIR, "test_{}".format(trial.number))
    executor = AllenNLPExecutor(trial, "classifier.jsonnet", serialization_dir, use_poetry=True)
    return executor.run()


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    shutil.rmtree(MODEL_DIR)
