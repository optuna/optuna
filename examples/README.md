Optuna Examples
================

This page contains a list of example codes written with Optuna.

### Simple Black-box Optimization

* [quadratic function](./quadratic_simple.py)

### Examples with ML Libraries

* [scikit-learn](./sklearn_simple.py)
* [Chainer](./chainer_simple.py)
* [ChainerMN](./chainermn_simple.py)
* [LightGBM](./lightgbm_simple.py)
* [MXNet](./mxnet_simple.py)
* [PyTorch](./pytorch_simple.py)
* [XGBoost](./xgboost_simple.py)
* [Tensorflow](./tensorflow_estimator_simple.py)
* [Tensorflow(eager)](./tensorflow_eager_simple.py)

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

### Examples of Custom Sampler

* [GPyOptSampler](./samplers/gpyopt_sampler.py)

### Examples of Visualization

* [plotting learning curves](./visualization/plot_intermediate_values.ipynb)
