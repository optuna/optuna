Optuna Examples
================

This page contains a list of example codes written with Optuna.

### Simple Black-box Optimization

* [Quadratic function](./quadratic_simple.py)

### Examples with ML Libraries

* [Scikit-learn](./sklearn_simple.py)
* [Scikit-image](./skimage_lbp_simple.py)
* [Chainer](./chainer_simple.py)
* [ChainerMN](./chainermn_simple.py)
* [Dask-ML](./dask_ml_simple.py)
* [LightGBM](./lightgbm_simple.py)
* [LightGBM Tuner](./lightgbm_tuner_simple.py)
* [CatBoost](./catboost_simple.py)
* [MXNet](./mxnet_simple.py)
* [PyTorch](./pytorch_simple.py)
* [PyTorch Ignite](./pytorch_ignite_simple.py)
* [PyTorch Lightning](./pytorch_lightning_simple.py)
* [XGBoost](./xgboost_simple.py)
* [Tensorflow](./tensorflow_estimator_simple.py)
* [Tensorflow (eager)](./tensorflow_eager_simple.py)
* [Keras](./keras_simple.py)
* [FastAI](./fastai_simple.py)
* [AllenNLP](./allennlp/allennlp_simple.py)
* [AllenNLP (Jsonnet)](./allennlp/allennlp_jsonnet.py)

### An example where an objective function uses additional arguments

The following example demonstrates how to implement an objective function that uses additional arguments other than `trial`.
* [Scikit-learn (callable class version)](./sklearn_additional_args.py)

### Examples of Pruning

The following example demonstrates how to implement pruning logic with Optuna.

* [Simple pruning (scikit-learn)](./pruning/simple.py)

In addition, integration modules are available for the following libraries, providing simpler interfaces to utilize pruning.

* [Pruning with Chainer integration module](./pruning/chainer_integration.py)
* [Pruning with XGBoost integration module](./pruning/xgboost_integration.py)
* [Pruning with XGBoost integration module (cross validation, XGBoost.cv)](./pruning/xgboost_cv_integration.py)
* [Pruning with LightGBM integration module](./pruning/lightgbm_integration.py)
* [Pruning with ChainerMN integration module](./pruning/chainermn_integration.py)
* [Pruning with Tensorflow integration module](./pruning/tensorflow_estimator_integration.py)
* [Pruning with Keras integration module](./pruning/keras_integration.py)
* [Pruning with MXNet integration module](./pruning/mxnet_integration.py)
* [Pruning with PyTorch Ignite integration module](./pytorch_ignite_simple.py)
* [Pruning with PyTorch Lightning integration module](./pytorch_lightning_simple.py)
* [Pruning with FastAI integration module](./fastai_simple.py)

### Examples of User-Defined Sampler

* [SimulatedAnnealingSampler](./samplers/simulated_annealing_sampler.py)

### Examples of Visualization

* [Visualizing study](https://colab.research.google.com/github/optuna/optuna/blob/master/examples/visualization/plot_study.ipynb)

### Examples of MLflow

* [Tracking optimization process with MLflow](./mlflow/keras_mlflow.py)

### Examples of Distributed Optimization

* [Optimizing on Kubernetes](./distributed/kubernetes/README.md)

### External projects using Optuna

* [RL Baselines Zoo](https://github.com/DLR-RM/rl-baselines3-zoo)
* [Allegro Trains](https://github.com/allegroai/trains)
* [neptune.ai](https://neptune.ai)
* [PyKEEN](https://github.com/pykeen/pykeen)
* [Mozilla Voice STT](https://github.com/mozilla/DeepSpeech)
* [OptGBM: A scikit-learn compatible LightGBM estimator with Optuna](https://github.com/Y-oHr-N/OptGBM)
* [Catalyst](https://github.com/catalyst-team/catalyst)

PRs to add additional projects welcome!

### Running with Optuna's Docker images?
You can use our docker images with the tag ending with `-dev` to run most of the examples.
For example, you can run [PyTorch Lightning Simple](./pytorch_ignite_simple.py) via `docker run --rm optuna/optuna:py3.7-dev python examples/pytorch_lightning_simple.py`.  
Also, you can try our visualization example in Jupyter Notebook by opening `localhost:8888` in your browser after executing this:

```bash
docker run -p 8888:8888 --rm optuna/optuna:py3.7-dev jupyter notebook --allow-root --no-browser --port 8888 --ip 0.0.0.0 --NotebookApp.token='' --NotebookApp.password=''
```
