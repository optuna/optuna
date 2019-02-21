"""
Optuna example that optimizes a simple quadratic function.

In this example, we optimize a simple quadratic function. We also demonstrate how to continue an
optimization and to use timeouts.

We have the following two ways to execute this example:

(1) Execute this code directly.
    $ python quadratic_simple.py


(2) Execute through CLI.
    $ STUDY_NAME=`optuna create-study --storage sqlite:///example.db`
    $ optuna study optimize quadratic_simple.py objective --n-trials=100 --study $STUDY_NAME \
      --storage sqlite:///example.db

"""

import optuna


# Define a simple 2-dimensional objective function whose minimum value is -1 when (x, y) = (0, -1).
def objective(trial):
    x = trial.suggest_uniform('x', -100, 100)
    y = trial.suggest_categorical('y', [-1, 0, 1])
    return x**2 + y


if __name__ == '__main__':
    # Let us minimize the objective function above.
    print('Running 10 trials...')
    study = optuna.create_study()
    study.optimize(objective, n_trials=10)
    print('Best value: {} (params: {})\n'.format(study.best_value, study.best_params))

    # We can continue the optimization as follows.
    print('Running 20 additional trials...')
    study.optimize(objective, n_trials=20)
    print('Best value: {} (params: {})\n'.format(study.best_value, study.best_params))

    # We can specify the timeout instead of a number of trials.
    print('Running additional trials in 2 seconds...')
    study.optimize(objective, timeout=2.0)
    print('Best value: {} (params: {})\n'.format(study.best_value, study.best_params))
