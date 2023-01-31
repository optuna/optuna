.. module:: optuna.integration

optuna.integration
==================

The :mod:`~optuna.integration` module contains classes used to integrate Optuna with external machine learning frameworks.

For most of the ML frameworks supported by Optuna, the corresponding Optuna integration class serves only to implement a callback object and functions, compliant with the framework's specific callback API, to be called with each intermediate step in the model training. The functionality implemented in these callbacks across the different ML frameworks includes:

(1) Reporting intermediate model scores back to the Optuna trial using :func:`optuna.trial.Trial.report`,
(2) According to the results of :func:`optuna.trial.Trial.should_prune`, pruning the current model by raising :func:`optuna.TrialPruned`, and
(3) Reporting intermediate Optuna data such as the current trial number back to the framework, as done in :class:`~optuna.integration.MLflowCallback`.

For scikit-learn, an integrated :class:`~optuna.integration.OptunaSearchCV` estimator is available that combines scikit-learn BaseEstimator functionality with access to a class-level ``Study`` object.

AllenNLP
--------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   optuna.integration.AllenNLPExecutor
   optuna.integration.allennlp.dump_best_config
   optuna.integration.AllenNLPPruningCallback

BoTorch
-------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   optuna.integration.BoTorchSampler
   optuna.integration.botorch.qei_candidates_func
   optuna.integration.botorch.qehvi_candidates_func
   optuna.integration.botorch.qnehvi_candidates_func
   optuna.integration.botorch.qparego_candidates_func

Catalyst
--------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   optuna.integration.CatalystPruningCallback

CatBoost
--------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   optuna.integration.CatBoostPruningCallback

Chainer
-------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   optuna.integration.ChainerPruningExtension
   optuna.integration.ChainerMNStudy

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

Keras
-----

.. autosummary::
   :toctree: generated/
   :nosignatures:

   optuna.integration.KerasPruningCallback

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

MXNet
-----

.. autosummary::
   :toctree: generated/
   :nosignatures:

   optuna.integration.MXNetPruningCallback

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

SHAP
---------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   optuna.integration.ShapleyImportanceEvaluator

skorch
------

.. autosummary::
   :toctree: generated/
   :nosignatures:

    optuna.integration.SkorchPruningCallback

TensorFlow
----------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   optuna.integration.TensorBoardCallback
   optuna.integration.TensorFlowPruningHook
   optuna.integration.TFKerasPruningCallback

XGBoost
-------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   optuna.integration.XGBoostPruningCallback
