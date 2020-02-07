<div align="center"><img src="https://raw.githubusercontent.com/optuna/optuna/master/docs/image/optuna-logo.png" width="800"/></div>

# Optuna: A hyperparameter optimization framework

[![Python](https://img.shields.io/badge/python-3.5%20%7C%203.6%20%7C%203.7%20%7C%203.8-blue)](https://www.python.org)
[![pypi](https://img.shields.io/pypi/v/optuna.svg)](https://pypi.python.org/pypi/optuna)
[![conda](https://img.shields.io/conda/vn/conda-forge/optuna.svg)](https://anaconda.org/conda-forge/optuna)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/optuna/optuna)
[![CircleCI](https://circleci.com/gh/optuna/optuna.svg?style=svg)](https://circleci.com/gh/optuna/optuna)
[![Read the Docs](https://readthedocs.org/projects/optuna/badge/?version=stable)](https://optuna.readthedocs.io/en/stable/)
[![Codecov](https://codecov.io/gh/optuna/optuna/branch/master/graph/badge.svg)](https://codecov.io/gh/optuna/optuna/branch/master)
[![Gitter chat](https://badges.gitter.im/optuna/gitter.svg)](https://gitter.im/optuna/optuna)

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
- Lightweight, versatile, and platform agnostic architecture


## Basic Concepts

We use the terms *study* and *trial* as follows:

- Study: optimization based on an objective function
- Trial: a single execution of the objective function

Please refer to sample code below. The goal of a *study* is to find out the optimal set of
hyperparameter values (e.g., `classifier` and `svm_c`) through multiple *trials* (e.g.,
`n_trials=100`). Optuna is a framework designed for the automation and the acceleration of the
optimization *studies*.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/optuna/optuna/blob/master/examples/quickstart.ipynb)

```python
import ...

# Define an objective function to be minimized.
def objective(trial):

    # Invoke suggest methods of a Trial object to generate hyperparameters.
    regressor_name = trial.suggest_categorical('classifier', ['SVR', 'RandomForest'])
    if regressor_name == 'SVR':
        svr_c = trial.suggest_loguniform('svr_c', 1e-10, 1e10)
        regressor_obj = sklearn.svm.SVR(C=svr_c)
    else:
        rf_max_depth = trial.suggest_int('rf_max_depth', 2, 32)
        regressor_obj = sklearn.ensemble.RandomForestRegressor(max_depth=rf_max_depth)

    X, y = sklearn.datasets.load_boston(return_X_y=True)
    X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X, y, random_state=0)

    regressor_obj.fit(X_train, y_train)
    y_pred = regressor_obj.predict(X_val)

    error = sklearn.metrics.mean_squared_error(y_val, y_pred)

    return error  # A objective value linked with the Trial object.

study = optuna.create_study()  # Create a new study.
study.optimize(objective, n_trials=100)  # Invoke optimization of the objective function.
```


## Integrations

[Integrations modules](https://optuna.readthedocs.io/en/stable/tutorial/pruning.html), which allow pruning, or early stopping, of unpromising trials are available for the following libraries:

* [XGBoost](./examples/pruning/xgboost_integration.py)
* [LightGBM](./examples/pruning/lightgbm_integration.py)
* [Chainer](./examples/pruning/chainer_integration.py)
* [Keras](./examples/pruning/keras_integration.py)
* [TensorFlow](./examples/pruning/tensorflow_estimator_integration.py)
* [tf.keras](./examples/pruning/tfkeras_integration.py)
* [MXNet](./examples/pruning/mxnet_integration.py)
* [PyTorch Ignite](./examples/pytorch_ignite_simple.py)
* [PyTorch Lightning](./examples/pytorch_lightning_simple.py)
* [FastAI](./examples/fastai_simple.py)

## Installation

Optuna is available at [the Python Package Index](https://pypi.org/project/optuna/) and on [Anaconda Cloud](https://anaconda.org/conda-forge/optuna).

```
# PyPI
$ pip install optuna
```

```
# Anaconda Cloud
$ conda install -c conda-forge optuna
```

Optuna supports Python 3.5 or newer.


## Communication

- [Github Issues] for bug reports, feature requests and questions.
- [Gitter] for interactive chat with developers.
- [Stack Overflow] for questions.

[GitHub issues]: https://github.com/optuna/optuna/issues
[Gitter]: https://gitter.im/optuna/optuna
[Stack Overflow]: https://stackoverflow.com/questions/tagged/optuna


## Contribution

Any contributions to Optuna are welcome! When you send a pull request, please follow the
[contribution guide](./CONTRIBUTING.md).


## License

MIT License (see [LICENSE](./LICENSE)).


## Reference

Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta, and Masanori Koyama. 2019.
Optuna: A Next-generation Hyperparameter Optimization Framework. In KDD ([arXiv](https://arxiv.org/abs/1907.10902)).
