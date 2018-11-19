Optuna Examples
================

This page contains a list of example codes written with Optuna.

### Simple Black-box Optimization

* [quadratic function](./quadratic_simple.py)

### Examples with ML Libraries

* [scikit-learn](./sklearn_simple.py)
* [Chainer](./chainer_simple.py)
* [ChainerMN](./chainermn_simple.py)
* [LighGBM](./lightgbm_simple.py)
* [XGBoost](./xgboost_simple.py)

### Examples of Pruning

The following example demonstrates how to implement pruning logic with Optuna.

* [simple pruning (scikit-learn)](./pruning/simple.py)

In addition, integration modules are available for the following libraries, providing simpler interfaces to utilize pruning.

* [pruning with Chainer integration module](./pruning/chainer_integration.py)
* [pruning with XGBoost integration module](./pruning/xgboost_integration.py)
