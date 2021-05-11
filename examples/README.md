Optuna Examples
================

This page contains a list of example codes written with Optuna.

### Simple Black-box Optimization

* [Quadratic function](./quadratic_simple.py)

### Examples with ML Libraries

The examples with ML libraries was moved to [optuna/optuna-examples](https://github.com/optuna/optuna-examples/).

### An example where an objective function uses additional arguments

The following example demonstrates how to implement an objective function that uses additional arguments other than `trial`.
* [Scikit-learn (callable class version)](https://github.com/optuna/optuna-examples/tree/main/sklearn/sklearn_additional_args.py)

### Examples of Pruning

The following example demonstrates how to implement pruning logic with Optuna.

* [Simple pruning (scikit-learn)](./simple_pruning.py)

### Examples of User-Defined Sampler

* [SimulatedAnnealingSampler](./samplers/simulated_annealing_sampler.py)

### Examples of Multi-Objective Optimization

* [Optimization with BoTorch](./multi_objective/botorch_simple.py)
* [Optimization of MLP with PyTorch](./multi_objective/pytorch_simple.py)

### Examples of Visualization

* [Visualizing study](https://colab.research.google.com/github/optuna/optuna/blob/master/examples/visualization/plot_study.ipynb)

### An example to enqueue trials with given parameter values

* [Enqueuing trials with given parameters](./enqueue_trial.py)

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
