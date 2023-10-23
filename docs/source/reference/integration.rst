.. module:: optuna.integration

optuna.integration
==================

The :mod:`~optuna.integration` module contains classes used to integrate Optuna with external machine learning frameworks.

.. note::
   Optuna's integration modules for third-party libraries have started migrating from Optuna itself to a package called 
   `optuna-integration`. Please check the `repository <https://github.com/optuna/optuna-integration>`_ and 
   the `documentation <https://optuna-integration.readthedocs.io/en/latest/index.html>`_.

For most of the ML frameworks supported by Optuna, the corresponding Optuna integration class serves only to implement a callback object and functions, compliant with the framework's specific callback API, to be called with each intermediate step in the model training. The functionality implemented in these callbacks across the different ML frameworks includes:

(1) Reporting intermediate model scores back to the Optuna trial using :func:`optuna.trial.Trial.report`,
(2) According to the results of :func:`optuna.trial.Trial.should_prune`, pruning the current model by raising :func:`optuna.TrialPruned`, and
(3) Reporting intermediate Optuna data such as the current trial number back to the framework, as done in :class:`~optuna.integration.MLflowCallback`.

For scikit-learn, an integrated :class:`~optuna.integration.OptunaSearchCV` estimator is available that combines scikit-learn BaseEstimator functionality with access to a class-level ``Study`` object.

BoTorch
-------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   optuna.integration.BoTorchSampler
   optuna.integration.botorch.logei_candidates_func
   optuna.integration.botorch.qei_candidates_func
   optuna.integration.botorch.qnei_candidates_func
   optuna.integration.botorch.qehvi_candidates_func
   optuna.integration.botorch.qnehvi_candidates_func
   optuna.integration.botorch.qparego_candidates_func

CatBoost
--------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   optuna.integration.CatBoostPruningCallback

Dask
----

.. autosummary::
   :toctree: generated/
   :nosignatures:

   optuna.integration.DaskStorage

fast.ai
-------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   optuna.integration.FastAIV1PruningCallback
   optuna.integration.FastAIV2PruningCallback
   optuna.integration.FastAIPruningCallback

LightGBM
--------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   optuna.integration.LightGBMPruningCallback
   optuna.integration.lightgbm.train
   optuna.integration.lightgbm.LightGBMTuner
   optuna.integration.lightgbm.LightGBMTunerCV

MLflow
------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   optuna.integration.MLflowCallback

Weights & Biases
----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   optuna.integration.WeightsAndBiasesCallback

pycma
-----

.. autosummary::
   :toctree: generated/
   :nosignatures:

   optuna.integration.PyCmaSampler
   optuna.integration.CmaEsSampler

PyTorch
-------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   optuna.integration.PyTorchIgnitePruningHandler
   optuna.integration.PyTorchLightningPruningCallback
   optuna.integration.TorchDistributedTrial

scikit-learn
------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   optuna.integration.OptunaSearchCV

scikit-optimize
---------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   optuna.integration.SkoptSampler

TensorFlow
----------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   optuna.integration.TensorBoardCallback

XGBoost
-------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   optuna.integration.XGBoostPruningCallback

Dependencies of each integration
--------------------------------

We summarize the necessary dependencies for each integration.

+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------+
| Integration                                                                                                                                                                       | Dependencies                       |
+===================================================================================================================================================================================+====================================+
| `AllenNLP <https://github.com/optuna/optuna/tree/master/optuna/integration/allennlp>`_                                                                                            | allennlp, torch, psutil, jsonnet   |
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------+
| `BoTorch <https://github.com/optuna/optuna/blob/master/optuna/integration/botorch.py>`_                                                                                           | botorch, gpytorch, torch           |
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------+
| `Catalyst <https://github.com/optuna/optuna/blob/master/optuna/integration/catalyst.py>`_                                                                                         | catalyst                           |
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------+
| `CatBoost <https://github.com/optuna/optuna/blob/master/optuna/integration/catboost.py>`_                                                                                         | catboost                           |
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------+
| `ChainerMN <https://github.com/optuna/optuna/blob/master/optuna/integration/chainermn.py>`_                                                                                       | chainermn                          |
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------+
| `Chainer <https://github.com/optuna/optuna/blob/master/optuna/integration/chainer.py>`_                                                                                           | chainer                            |
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------+
| `pycma <https://github.com/optuna/optuna/blob/master/optuna/integration/cma.py>`_                                                                                                 | cma                                |
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------+
| `Dask <https://github.com/optuna/optuna/blob/master/optuna/integration/dask.py>`_                                                                                                 | distributed                        |
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------+
| FastAI (`v1 <https://github.com/optuna/optuna/blob/master/optuna/integration/fastaiv1.py>`_, `v2 <https://github.com/optuna/optuna/blob/master/optuna/integration/fastaiv2.py>`_) | fastai                             |
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------+
| `Keras <https://github.com/optuna/optuna/blob/master/optuna/integration/keras.py>`_                                                                                               | keras                              |
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------+
| `LightGBMTuner <https://github.com/optuna/optuna/blob/master/optuna/integration/lightgbm.py>`_                                                                                    | lightgbm, scikit-learn             |
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------+
| `LightGBMPruningCallback <https://github.com/optuna/optuna/blob/master/optuna/integration/lightgbm.py>`_                                                                          | lightgbm                           |
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------+
| `MLflow <https://github.com/optuna/optuna/blob/master/optuna/integration/mlflow.py>`_                                                                                             | mlflow                             |
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------+
| `MXNet <https://github.com/optuna/optuna/blob/master/optuna/integration/mxnet.py>`_                                                                                               | mxnet                              |
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------+
| PyTorch `Distributed <https://github.com/optuna/optuna/blob/master/optuna/integration/pytorch_distributed.py>`_                                                                   | torch                              |
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------+
| PyTorch (`Ignite <https://github.com/optuna/optuna/blob/master/optuna/integration/pytorch_ignite.py>`_)                                                                           | pytorch-ignite                     |
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------+
| PyTorch (`Lightning <https://github.com/optuna/optuna/blob/master/optuna/integration/pytorch_lightning.py>`_)                                                                     | pytorch-lightning                  |
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------+
| `SHAP <https://github.com/optuna/optuna/blob/master/optuna/integration/shap.py>`_                                                                                                 | scikit-learn, shap                 |
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------+
| `Scikit-learn <https://github.com/optuna/optuna/blob/master/optuna/integration/sklearn.py>`_                                                                                      | pandas, scipy, scikit-learn        |
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------+
| `Scikit-optimize <https://github.com/optuna/optuna/blob/master/optuna/integration/skopt.py>`_                                                                                     | scikit-optimize                    |
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------+
| `SKorch <https://github.com/optuna/optuna/blob/master/optuna/integration/skorch.py>`_                                                                                             | skorch                             |
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------+
| `TensorBoard <https://github.com/optuna/optuna/blob/master/optuna/integration/tensorboard.py>`_                                                                                   | tensorboard, tensorflow            |
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------+
| `TensorFlow <https://github.com/optuna/optuna/blob/master/optuna/integration/tensorflow.py>`_                                                                                     | tensorflow, tensorflow-estimator   |
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------+
| `TensorFlow + Keras <https://github.com/optuna/optuna/blob/master/optuna/integration/tfkeras.py>`_                                                                                | tensorflow                         |
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------+
| `Weights & Biases <https://github.com/optuna/optuna/blob/master/optuna/integration/wandb.py>`_                                                                                    | wandb                              |
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------+
| `XGBoost <https://github.com/optuna/optuna/blob/master/optuna/integration/xgboost.py>`_                                                                                           | xgboost                            |
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------+
