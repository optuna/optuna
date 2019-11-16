Optuna Examples
================

This page contains a list of example codes written with Optuna.

### Simple Black-box Optimization

* [quadratic function](./quadratic_simple.py)

### Examples with ML Libraries

* [scikit-learn](./sklearn_simple.py)
* [Chainer](./chainer_simple.py)
* [ChainerMN](./chainermn_simple.py)
* [Dask-ML](./dask_ml_simple.py)
* [LightGBM](./lightgbm_simple.py)
* [CatBoost](./catboost_simple.py)
* [MXNet](./mxnet_simple.py)
* [PyTorch](./pytorch_simple.py)
* [PyTorch Ignite](./pytorch_ignite_simple.py)
* [PyTorch Lightning](./pytorch_lightning_simple.py)
* [XGBoost](./xgboost_simple.py)
* [Tensorflow](./tensorflow_estimator_simple.py)
* [Tensorflow(eager)](./tensorflow_eager_simple.py)
* [Keras](./keras_simple.py)

### An example where an objective function uses additional arguments

The following example demonstrates how to implement an objective function that uses additional arguments other than `trial`.
* [scikit-learn (callable class version)](./sklearn_additional_args.py)

### Examples of Pruning

The following example demonstrates how to implement pruning logic with Optuna.

* [simple pruning (scikit-learn)](./pruning/simple.py)

In addition, integration modules are available for the following libraries, providing simpler interfaces to utilize pruning.

* [pruning with Chainer integration module](./pruning/chainer_integration.py)
* [pruning with XGBoost integration module](./pruning/xgboost_integration.py)
* [pruning with LightGBM integration module](./pruning/lightgbm_integration.py)
* [pruning with ChainerMN integration module](./pruning/chainermn_integration.py)
* [pruning with Tensorflow integration module](./pruning/tensorflow_estimator_integration.py)
* [pruning with Keras integration module](./pruning/keras_integration.py)
* [pruning with MXNet integration module](./pruning/mxnet_integration.py)
* [pruning with PyTorch Ignite integration module](./pytorch_ignite_simple.py)

### Examples of User-Defined Sampler

* [SimulatedAnnealingSampler](./samplers/simulated_annealing_sampler.py)

### Examples of Visualization

* [plotting learning curves](https://nbviewer.jupyter.org/github/optuna/optuna/blob/master/examples/visualization/plot_intermediate_values.ipynb)
* [plotting optimization history](https://nbviewer.jupyter.org/github/optuna/optuna/blob/master/examples/visualization/plot_optimization_history.ipynb)
* [plotting relationship between a parameter and objective value (slice plot)](https://nbviewer.jupyter.org/github/optuna/optuna/blob/master/examples/visualization/plot_slice.ipynb)
* [plotting relationship between two parameters (contour plot)](https://nbviewer.jupyter.org/github/optuna/optuna/blob/master/examples/visualization/plot_contour.ipynb)
* [plotting parallel coordinate](https://nbviewer.jupyter.org/github/optuna/optuna/blob/master/examples/visualization/plot_parallel_coordinate.ipynb)


### Examples of Distributed Optimization

* [optimizing on kubernetes](./distributed/kubernetes/README.md)