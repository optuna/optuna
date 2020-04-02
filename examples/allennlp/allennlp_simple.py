"""
Optuna example that optimizes a classifier configuration for IMDB movie review dataset.
This script is based on the example of allentune (https://github.com/allenai/allentune).

In this example, we optimize the validation accuracy of sentiment classification using AllenNLP.
Since it is too time-consuming to use the entire dataset, we here use a small subset of it.

We have the following two ways to execute this example:

(1) Execute this code directly.
    $ python allennlp_simple.py


(2) Execute through CLI.
    $ STUDY_NAME=`optuna create-study --direction maximize --storage sqlite:///example.db`
    $ optuna study optimize allennlp_simple.py objective --n-trials=100 --study $STUDY_NAME \
      --storage sqlite:///example.db

"""

import json
import os.path
import subprocess

import optuna


def objective(trial):
    learning_rate = trial.suggest_uniform('learnig_rate', 1e-2, 1e-1)
    dropout = trial.suggest_uniform('dropout', 0.0, 0.5)
    max_filter_size = trial.suggest_int("max_filter_size", 3, 6)
    num_filters = trial.suggest_int("num_filters", 16, 128)
    num_output_layers = trial.suggest_int("num_output_layers", 1, 3)
    hidden_size = trial.suggest_int("hidden_size", 16, 128)

    env = {
        "LEARNING_RATE": str(learning_rate),
        "DROPOUT": str(dropout),
        "MAX_FILTER_SIZE": str(max_filter_size),
        "NUM_FILTERS": str(num_filters),
        "NUM_OUTPUT_LAYERS": str(num_output_layers),
        "HIDDEN_SIZE": str(hidden_size),
        "PATH": "/usr/bin"
    }

    allennlp_command = "/home/ubuntu/.poetry/bin/poetry run allennlp train"
    serialization_dir = f"test/test_{trial.number}"
    subprocess.run(f"{allennlp_command} -s {serialization_dir} classifier.jsonnet", stdout=True, env=env, shell=True)
    metrics = json.load(open(os.path.join(serialization_dir, "metrics.json")))["best_validation_accuracy"]
    return metrics


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
