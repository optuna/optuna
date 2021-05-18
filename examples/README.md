Optuna Examples
================

This page contains a list of example codes written with Optuna. The example files are in [optuna/optuna-examples](https://github.com/optuna/optuna-examples/).

### Simple Black-box Optimization

* [Quadratic function](./quadratic_simple.py)

### Examples with ML Libraries

* [AllenNLP](https://github.com/optuna/optuna-examples/blob/main/allennlp/allennlp_simple.py)
* [AllenNLP (Jsonnet)](https://github.com/optuna/optuna-examples/blob/main/allennlp/allennlp_jsonnet.py)
* [Catalyst](https://github.com/optuna/optuna-examples/blob/main/pytorch/catalyst_simple.py)
* [CatBoost](https://github.com/optuna/optuna-examples/blob/main/catboost/catboost_simple.py)
* [Chainer](https://github.com/optuna/optuna-examples/blob/main/chainer/chainer_simple.py)
* [ChainerMN](https://github.com/optuna/optuna-examples/blob/main/chainer/chainermn_simple.py)
* [Dask-ML](https://github.com/optuna/optuna-examples/blob/main/dask_ml/dask_ml_simple.py)
* [FastAI V1](https://github.com/optuna/optuna-examples/blob/main/fastai/fastaiv1_simple.py)
* [FastAI V2](https://github.com/optuna/optuna-examples/blob/main/fastai/fastaiv2_simple.py)
* [Haiku](https://github.com/optuna/optuna-examples/blob/main/haiku/haiku_simple.py)
* [Gluon](https://github.com/optuna/optuna-examples/blob/main/mxnet/gluon_simple.py)
* [Keras](https://github.com/optuna/optuna-examples/blob/main/keras/keras_simple.py)
* [LightGBM](https://github.com/optuna/optuna-examples/blob/main/lightgbm/lightgbm_simple.py)
* [LightGBM Tuner](https://github.com/optuna/optuna-examples/blob/main/lightgbm/lightgbm_tuner_simple.py)
* [MXNet](https://github.com/optuna/optuna-examples/blob/main/mxnet/mxnet_simple.py)
* [PyTorch](https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_simple.py)
* [PyTorch Ignite](https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_ignite_simple.py)
* [PyTorch Lightning](https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_lightning_simple.py)
* [RAPIDS](https://github.com/optuna/optuna-examples/blob/main/rapids_simple.py)
* [Scikit-learn](https://github.com/optuna/optuna-examples/blob/main/sklearn/sklearn_simple.py)
* [Scikit-learn OptunaSearchCV](https://github.com/optuna/optuna-examples/blob/main/sklearn/sklearn_optuna_search_cv_simple.py)
* [Scikit-image](https://github.com/optuna/optuna-examples/blob/main/skimage/skimage_lbp_simple.py)
* [SKORCH](https://github.com/optuna/optuna-examples/blob/main/pytorch/skorch_simple.py)
* [Tensorflow](https://github.com/optuna/optuna-examples/blob/main/tensorflow/tensorflow_estimator_simple.py)
* [Tensorflow (eager)](https://github.com/optuna/optuna-examples/blob/main/tensorflow/tensorflow_eager_simple.py)
* [XGBoost](https://github.com/optuna/optuna-examples/blob/main/xgboost/xgboost_simple.py)

### An example where an objective function uses additional arguments

The following example demonstrates how to implement an objective function that uses additional arguments other than `trial`.
* [Scikit-learn (callable class version)](https://github.com/optuna/optuna-examples/tree/main/sklearn/sklearn_additional_args.py)

### Examples of Pruning

The following example demonstrates how to implement pruning logic with Optuna.

* [Simple pruning (scikit-learn)](./simple_pruning.py)

In addition, integration modules are available for the following libraries, providing simpler interfaces to utilize pruning.

* [Pruning with Catalyst integration module](https://github.com/optuna/optuna-examples/blob/main/pytorch/catalyst_simple.py)
* [Pruning with Catboost integration module](https://github.com/optuna/optuna-examples/blob/main/catboost/catboost_simple.py)
* [Pruning with Chainer integration module](https://github.com/optuna/optuna-examples/blob/main/chainer/chainer_integration.py)
* [Pruning with ChainerMN integration module](https://github.com/optuna/optuna-examples/blob/main/chainer/chainermn_integration.py)
* [Pruning with FastAI V1 integration module](https://github.com/optuna/optuna-examples/blob/main/fastai/fastaiv1_simple.py)
* [Pruning with FastAI V2 integration module](https://github.com/optuna/optuna-examples/blob/main/fastai/fastaiv2_simple.py)
* [Pruning with Keras integration module](https://github.com/optuna/optuna-examples/blob/main/keras/keras_integration.py)
* [Pruning with LightGBM integration module](https://github.com/optuna/optuna-examples/blob/main/lightgbm/lightgbm_integration.py)
* [Pruning with MXNet integration module](https://github.com/optuna/optuna-examples/blob/main/mxnet/mxnet_integration.py)
* [Pruning with PyTorch integration module](https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_simple.py)
* [Pruning with PyTorch Ignite integration module](https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_ignite_simple.py)
* [Pruning with PyTorch Lightning integration module](https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_lightning_simple.py)
* [Pruning with Tensorflow integration module](https://github.com/optuna/optuna-examples/blob/main/tensorflow/tensorflow_estimator_integration.py)
* [Pruning with XGBoost integration module](https://github.com/optuna/optuna-examples/blob/main/xgboost/xgboost_integration.py)
* [Pruning with XGBoost integration module (cross validation, XGBoost.cv)](https://github.com/optuna/optuna-examples/blob/main/xgboost/xgboost_cv_integration.py)

### Examples of User-Defined Sampler

* [SimulatedAnnealingSampler](./samplers/simulated_annealing_sampler.py)

### Examples of Multi-Objective Optimization

* [Optimization with BoTorch](./multi_objective/botorch_simple.py)
* [Optimization of MLP with PyTorch](./multi_objective/pytorch_simple.py)

### Examples of Visualization

* [Visualizing study](https://colab.research.google.com/github/optuna/optuna/blob/master/examples/visualization/plot_study.ipynb)

### An example to enqueue trials with given parameter values

* [Enqueuing trials with given parameters](./enqueue_trial.py)

### Examples of MLflow

* [Tracking optimization process with MLflow](https://github.com/optuna/optuna-examples/blob/main/mlflow/keras_mlflow.py)

### Examples of Hydra

* [Optimization with Hydra](https://github.com/optuna/optuna-examples/blob/main/hydra/simple.py)

### Examples of Distributed Optimization

* [Optimizing on Kubernetes](https://github.com/optuna/optuna-examples/blob/main/kubernetes/README.md)
* [Optimizing with Ray's joblib backend](https://github.com/optuna/optuna-examples/blob/main/ray/ray_joblib.py)

### Examples of Reinforcement Learning

* [Optimization of Hyperparameters for Stable-Baslines Agent](https://github.com/optuna/optuna-examples/blob/main/rl/sb3_simple.py)

### External projects using Optuna

* [Allegro Trains](https://github.com/allegroai/trains)
* [BBO-Rietveld: Automated crystal structure refinement](https://github.com/quantumbeam/BBO-Rietveld)
* [Catalyst](https://github.com/catalyst-team/catalyst)
* [CuPy](https://github.com/cupy/cupy)
* [Hydra's Optuna Sweeper plugin](https://hydra.cc/docs/next/plugins/optuna_sweeper/)
* [Mozilla Voice STT](https://github.com/mozilla/DeepSpeech)
* [neptune.ai](https://neptune.ai)
* [OptGBM: A scikit-learn compatible LightGBM estimator with Optuna](https://github.com/Y-oHr-N/OptGBM)
* [PyKEEN](https://github.com/pykeen/pykeen)
* [RL Baselines Zoo](https://github.com/DLR-RM/rl-baselines3-zoo)

PRs to add additional projects welcome!

### Running with Optuna's Docker images?
You can use our docker images with the tag ending with `-dev` to run most of the examples.
For example, you can run [PyTorch example](./multi_objective/pytorch_simple.py) via `docker run --rm optuna/optuna:py3.7-dev python examples/multi_objective/pytorch_simple.py`.
Also, you can try our visualization example in Jupyter Notebook by opening `locqalhost:8888` in your browser after executing this:

```bash
docker run -p 8888:8888 --rm optuna/optuna:py3.7-dev jupyter notebook --allow-root --no-browser --port 8888 --ip 0.0.0.0 --NotebookApp.token='' --NotebookApp.password=''
```
