# Optuna: A hyperparameter optimization framework

*Optuna* is an automatic hyperparameter optimization software framework, particularly designed
for machine learning. It features an imperative, *define-by-run* style user API. Thanks to our
*define-by-run* API, the code written with *Optuna* enjoys high modularity, and the user of
*Optuna* can dynamically construct the search spaces for the hyperparameters. *Optuna* also
features modern functionalities including parallel distributed optimization, premature pruning
of unpromising trials, and web dashboard.

## Basic Concepts

We use the term *study* to refer to the optimization based on an objective function. A single
execution of the objective function is a *trial*. The goal of a *study* is to find out the optimal
set of hyperparameter values through multiple *trials*. Optuna is a framework designed for the
automation and the acceleration of the optimization *studies*.

Please refer to following sample code:

```python
import ...

# A: define an objective function
def objective(trial):

    # C: invoke suggest APIs to generate hyperparameters
    classifier_name = trial.suggest_categorical('classifier', ['SVC', 'RandomForest'])
    if classifier_name == 'SVC':
        svc_c = trial.suggest_loguniform('svc_c', 1e-10, 1e10)
        classifier_obj = sklearn.svm.SVC(C=svc_c)
    else:
        rf_max_depth = trial.suggest_int('rf_max_depth', 2, 32)
        classifier_obj = sklearn.ensemble.RandomForestClassifier(max_depth=rf_max_depth)

    iris = sklearn.datasets.load_iris()
    x, y = iris.data , iris.target
    score = sklearn.model_selection.cross_val_score(classifier_obj , x, y)
    accuracy = score.mean()
    return accuracy

# B: invoke optimize API
optuna.optimize(objective , n_trials=100, direction='maximize')
```

All a user needs to do in Optuna is to
1. define an objective function (code block `A`), and to
1. invoke the `optimize API` that takes the objective function as an input (code block `B`).

`suggest API` like `trial.suggest_categorical` and `trial.suggest_loguniform` is to be invoked
inside an objective function in order to generate the hyperparameters for the *trial*
(code block `C`). Upon the invocation of `suggest API`, a hyperparameter is statistically sampled
based on the history of previously evaluated *trials*.

Optuna supports multiple sampling algorithms such as random search and [Tree-structured Parzen
Estimator](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf).

Please see [examples](https://github.com/pfnet/optuna/tree/master/examples) to check Optuna
applications to various machine learning library such as
[scikit-learn](https://scikit-learn.org/stable/), [Chainer](https://chainer.org/),
[XGBoost](https://xgboost.readthedocs.io/en/latest/) and
[LightGBM](https://lightgbm.readthedocs.io/en/latest/).

## Installation

To install Optuna, use `pip` as follows:

```
$ pip install git+https://github.com/pfnet/optuna.git
```

Optuna supports Python 2.7 and Python 3.4 or newer.

## Contribution

Any contributions to Optuna are welcome! When you send a pull request, please follow the
[contribution guide](https://github.com/pfnet/optuna/tree/master/CONTRIBUTING.md).

## License

MIT License (see [LICENSE](https://github.com/pfnet/optuna/tree/master/LICENSE)).