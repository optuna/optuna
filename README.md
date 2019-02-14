<div align="center"><img src="docs/image/optuna-logo.png" width="800"/></div>

# Optuna: A hyperparameter optimization framework

[![pypi](https://img.shields.io/pypi/v/optuna.svg)](https://pypi.python.org/pypi/optuna)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/pfnet/optuna)
[![CircleCI](https://circleci.com/gh/pfnet/optuna.svg?style=svg)](https://circleci.com/gh/pfnet/optuna)
[![Read the Docs](https://readthedocs.org/projects/optuna/badge/?version=stable)](https://optuna.readthedocs.io/en/stable/)

[**Website**](https://optuna.org/)
| [**Docs**](https://optuna.readthedocs.io/en/stable/)
| [**Install Guide**](https://optuna.readthedocs.io/en/stable/installation.html)
| [**Tutorial**](https://optuna.readthedocs.io/en/stable/tutorial/index.html)

*Optuna* is an automatic hyperparameter optimization software framework, particularly designed
for machine learning. It features an imperative, *define-by-run* style user API. Thanks to our
*define-by-run* API, the code written with Optuna enjoys high modularity, and the user of
Optuna can dynamically construct the search spaces for the hyperparameters.


## Key Features

Optuna has modern functionalities as follows:

- Parallel distributed optimization
- Pruning of unpromising trials
- Web dashboard


## Basic Concepts

We use the terms *study* and *trial* as follows:

- Study: optimization based on an objective function
- Trial: a single execution of the objective function

Please refer to sample code below. The goal of a *study* is to find out the optimal set of
hyperparameter values (e.g., `classifier` and `svm_c`) through multiple *trials* (e.g.,
`n_trials=100`). Optuna is a framework designed for the automation and the acceleration of the
optimization *studies*.


```python
import ...

# Define an objective function to be minimized.
def objective(trial):

    # Invoke suggest methods of a Trial object to generate hyperparameters.
    classifier_name = trial.suggest_categorical('classifier', ['SVC', 'RandomForest'])
    if classifier_name == 'SVC':
        svc_c = trial.suggest_loguniform('svc_c', 1e-10, 1e10)
        classifier_obj = sklearn.svm.SVC(C=svc_c)
    else:
        rf_max_depth = trial.suggest_int('rf_max_depth', 2, 32)
        classifier_obj = sklearn.ensemble.RandomForestClassifier(max_depth=rf_max_depth)

    iris = sklearn.datasets.load_iris()
    x, y = iris.data, iris.target
    score = sklearn.model_selection.cross_val_score(classifier_obj, x, y)
    accuracy = score.mean()
    
    return 1.0 - accuracy  # A objective value linked with the Trial object.

study = optuna.create_study()  # Create a new study.
study.optimize(objective, n_trials=100)  # Invoke optimization of the objective function.
```


## Installation

To install Optuna, use `pip` as follows:

```
$ pip install optuna
```

Optuna supports Python 2.7 and Python 3.5 or newer.


## Contribution

Any contributions to Optuna are welcome! When you send a pull request, please follow the
[contribution guide](./CONTRIBUTING.md).


## License

MIT License (see [LICENSE](./LICENSE)).
