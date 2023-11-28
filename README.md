<div align="center"><img src="https://raw.githubusercontent.com/optuna/optuna/master/docs/image/optuna-logo.png" width="800"/></div>

# Optuna: A hyperparameter optimization framework

[![Python](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org)
[![pypi](https://img.shields.io/pypi/v/optuna.svg)](https://pypi.python.org/pypi/optuna)
[![conda](https://img.shields.io/conda/vn/conda-forge/optuna.svg)](https://anaconda.org/conda-forge/optuna)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/optuna/optuna)
[![Read the Docs](https://readthedocs.org/projects/optuna/badge/?version=stable)](https://optuna.readthedocs.io/en/stable/)
[![Codecov](https://codecov.io/gh/optuna/optuna/branch/master/graph/badge.svg)](https://codecov.io/gh/optuna/optuna)

[**Website**](https://optuna.org/)
| [**Docs**](https://optuna.readthedocs.io/en/stable/)
| [**Install Guide**](https://optuna.readthedocs.io/en/stable/installation.html)
| [**Tutorial**](https://optuna.readthedocs.io/en/stable/tutorial/index.html)
| [**Examples**](https://github.com/optuna/optuna-examples)

*Optuna* is an automatic hyperparameter optimization software framework, particularly designed
for machine learning. It features an imperative, *define-by-run* style user API. Thanks to our
*define-by-run* API, the code written with Optuna enjoys high modularity, and the user of
Optuna can dynamically construct the search spaces for the hyperparameters.

## Key Features

Optuna has modern functionalities as follows:

- [Lightweight, versatile, and platform agnostic architecture](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/001_first.html)
  - Handle a wide variety of tasks with a simple installation that has few requirements.
- [Pythonic search spaces](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/002_configurations.html)
  - Define search spaces using familiar Python syntax including conditionals and loops.
- [Efficient optimization algorithms](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html)
  - Adopt state-of-the-art algorithms for sampling hyperparameters and efficiently pruning unpromising trials.
- [Easy parallelization](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html)
  - Scale studies to tens or hundreds or workers with little or no changes to the code.
- [Quick visualization](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/005_visualization.html)
  - Inspect optimization histories from a variety of plotting functions.


## Basic Concepts

We use the terms *study* and *trial* as follows:

- Study: optimization based on an objective function
- Trial: a single execution of the objective function

Please refer to sample code below. The goal of a *study* is to find out the optimal set of
hyperparameter values (e.g., `regressor` and `svr_c`) through multiple *trials* (e.g.,
`n_trials=100`). Optuna is a framework designed for the automation and the acceleration of the
optimization *studies*.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/optuna/optuna-examples/blob/main/quickstart.ipynb)

```python
import ...

# Define an objective function to be minimized.
def objective(trial):

    # Invoke suggest methods of a Trial object to generate hyperparameters.
    regressor_name = trial.suggest_categorical('regressor', ['SVR', 'RandomForest'])
    if regressor_name == 'SVR':
        svr_c = trial.suggest_float('svr_c', 1e-10, 1e10, log=True)
        regressor_obj = sklearn.svm.SVR(C=svr_c)
    else:
        rf_max_depth = trial.suggest_int('rf_max_depth', 2, 32)
        regressor_obj = sklearn.ensemble.RandomForestRegressor(max_depth=rf_max_depth)

    X, y = sklearn.datasets.fetch_california_housing(return_X_y=True)
    X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X, y, random_state=0)

    regressor_obj.fit(X_train, y_train)
    y_pred = regressor_obj.predict(X_val)

    error = sklearn.metrics.mean_squared_error(y_val, y_pred)

    return error  # An objective value linked with the Trial object.

study = optuna.create_study()  # Create a new study.
study.optimize(objective, n_trials=100)  # Invoke optimization of the objective function.
```


## Installation

Optuna is available at [the Python Package Index](https://pypi.org/project/optuna/) and on [Anaconda Cloud](https://anaconda.org/conda-forge/optuna).

```bash
# PyPI
$ pip install optuna
```

```bash
# Anaconda Cloud
$ conda install -c conda-forge optuna
```

Optuna supports Python 3.7 or newer.

Also, we also provide Optuna docker images on [DockerHub](https://hub.docker.com/r/optuna/optuna).


## Examples

Examples can be found in [optuna/optuna-examples](https://github.com/optuna/optuna-examples).

## Integrations

Optuna has integration features with various third-party libraries. Integrations can be found in [optuna/optuna-integration](https://github.com/optuna/optuna-integration) and the document is available [here](https://optuna-integration.readthedocs.io/en/stable/index.html). Integrations support libraries such as the following:

* [Catalyst](https://github.com/optuna/optuna-examples/tree/main/pytorch/catalyst_simple.py)
* [Catboost](https://github.com/optuna/optuna-examples/tree/main/catboost/catboost_pruning.py)
* [Dask](https://github.com/optuna/optuna-examples/tree/main/dask/dask_simple.py)
* [fastai (v2)](https://github.com/optuna/optuna-examples/tree/main/fastai/fastaiv2_simple.py)
* [Keras](https://github.com/optuna/optuna-examples/tree/main/keras/keras_integration.py)
* [LightGBM](https://github.com/optuna/optuna-examples/tree/main/lightgbm/lightgbm_integration.py)
* [MLflow](https://github.com/optuna/optuna-examples/tree/main/mlflow/keras_mlflow.py)
* [MXNet](https://github.com/optuna/optuna-examples/tree/main/mxnet/mxnet_integration.py)
* [PyTorch](https://github.com/optuna/optuna-examples/tree/main/pytorch/pytorch_simple.py)
* [PyTorch Ignite](https://github.com/optuna/optuna-examples/tree/main/pytorch/pytorch_ignite_simple.py)
* [PyTorch Lightning](https://github.com/optuna/optuna-examples/tree/main/pytorch/pytorch_lightning_simple.py)
* [TensorBoard](https://github.com/optuna/optuna-examples/tree/main/tensorboard/tensorboard_simple.py)
* [TensorFlow](https://github.com/optuna/optuna-examples/tree/main/tensorflow/tensorflow_estimator_integration.py)
* [tf.keras](https://github.com/optuna/optuna-examples/tree/main/tfkeras/tfkeras_integration.py)
* [Weights & Biases](https://github.com/optuna/optuna-examples/tree/main/wandb/wandb_integration.py)
* [XGBoost](https://github.com/optuna/optuna-examples/tree/main/xgboost/xgboost_integration.py)


## Web Dashboard

[Optuna Dashboard](https://github.com/optuna/optuna-dashboard) is a real-time web dashboard for Optuna.
You can check the optimization history, hyperparameter importances, etc. in graphs and tables.
You don't need to create a Python script to call [Optuna's visualization](https://optuna.readthedocs.io/en/stable/reference/visualization/index.html) functions.
Feature requests and bug reports welcome!

![optuna-dashboard](https://user-images.githubusercontent.com/5564044/204975098-95c2cb8c-0fb5-4388-abc4-da32f56cb4e5.gif)

Install `optuna-dashboard` via pip:

```
$ pip install optuna-dashboard
$ optuna-dashboard sqlite:///db.sqlite3
...
Listening on http://localhost:8080/
Hit Ctrl-C to quit.
```


## Communication

- [GitHub Discussions] for questions.
- [GitHub Issues] for bug reports and feature requests.

[GitHub Discussions]: https://github.com/optuna/optuna/discussions
[GitHub issues]: https://github.com/optuna/optuna/issues


## Contribution

Any contributions to Optuna are more than welcome!

If you are new to Optuna, please check the [good first issues](https://github.com/optuna/optuna/labels/good%20first%20issue). They are relatively simple, well-defined and are often good starting points for you to get familiar with the contribution workflow and other developers.

If you already have contributed to Optuna, we recommend the other [contribution-welcome issues](https://github.com/optuna/optuna/labels/contribution-welcome).

For general guidelines how to contribute to the project, take a look at [CONTRIBUTING.md](./CONTRIBUTING.md).


## Reference

Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta, and Masanori Koyama. 2019.
Optuna: A Next-generation Hyperparameter Optimization Framework. In KDD ([arXiv](https://arxiv.org/abs/1907.10902)).
