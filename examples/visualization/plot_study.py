"""
Optuna example that visualizes the optimization result of multi-layer perceptrons.

In this example, we optimize the validation accuracy of object recognition using
scikit-learn and Fashion-MNIST. We optimize a neural network. As it is too time
consuming to use the whole Fashion-MNIST dataset, we here use a small subset of it.

We have can execute this example as follows.

    $ python plot_study.py

**Note:** If a parameter contains missing values, a trial with missing values is not plotted.
"""


from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

import optuna
from optuna.visualization import plot_contour
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_slice


def build_objective():
    fmnist = fetch_openml(name="Fashion-MNIST", version=1)
    classes = list(set(fmnist.target))

    # For demonstrational purpose, only use a subset of the dataset.
    n_samples = 4000
    data = fmnist.data[:n_samples]
    target = fmnist.target[:n_samples]

    x_train, x_test, y_train, y_test = train_test_split(data, target)

    def objective(trial):

        clf = MLPClassifier(
            hidden_layer_sizes=tuple(
                [trial.suggest_int("n_units_l{}".format(i), 32, 64) for i in range(3)]
            ),
            learning_rate_init=trial.suggest_loguniform("lr_init", 1e-5, 1e-1),
        )

        for step in range(100):
            clf.partial_fit(x_train, y_train, classes=classes)
            value = clf.score(x_test, y_test)

            # Report intermediate objective value.
            trial.report(value, step)

            # Handle pruning based on the intermediate value.
            if trial.should_prune(step):
                raise optuna.exceptions.TrialPruned()

        return value

    return objective


if __name__ == "__main__":

    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    objective = build_objective()
    study.optimize(objective, n_trials=100, timeout=600)

    # Visualize the Optimization History
    plot_optimization_history(study).show()

    # Visualize the Learning Curves of the Trials
    plot_intermediate_values(study).show()

    # Visualize High-dimensional Parameter Relationships
    plot_parallel_coordinate(study).show()

    # Select Parameters to Visualize
    plot_parallel_coordinate(study, params=["lr_init", "n_units_l0"]).show()

    # Visualize Hyperparameter Relationships
    plot_contour(study).show()

    # Select Parameters to Visualize
    plot_contour(study, params=["n_units_l0", "n_units_l1"]).show()

    # Visualize Individual Hyperparameters
    plot_slice(study).show()

    # Select Parameters to Visualize
    plot_slice(study, params=["n_units_l0", "n_units_l1"]).show()
