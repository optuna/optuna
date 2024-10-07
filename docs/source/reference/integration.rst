.. module:: optuna.integration

optuna.integration
==================

The :mod:`~optuna.integration` module contains classes used to integrate Optuna with external machine learning frameworks.

.. note::
   Optuna's integration modules for third-party libraries have started migrating from Optuna itself to a package called
   `optuna-integration`. Please check the `repository <https://github.com/optuna/optuna-integration>`__ and
   the `documentation <https://optuna-integration.readthedocs.io/en/stable/index.html>`__.

For most of the ML frameworks supported by Optuna, the corresponding Optuna integration class serves only to implement a callback object and functions, compliant with the framework's specific callback API, to be called with each intermediate step in the model training. The functionality implemented in these callbacks across the different ML frameworks includes:

(1) Reporting intermediate model scores back to the Optuna trial using :func:`optuna.trial.Trial.report`,
(2) According to the results of :func:`optuna.trial.Trial.should_prune`, pruning the current model by raising :func:`optuna.TrialPruned`, and
(3) Reporting intermediate Optuna data such as the current trial number back to the framework, as done in :class:`~optuna.integration.MLflowCallback`.

For scikit-learn, an integrated :class:`~optuna.integration.OptunaSearchCV` estimator is available that combines scikit-learn BaseEstimator functionality with access to a class-level ``Study`` object.

Dependencies of each integration
--------------------------------

We summarize the necessary dependencies for each integration.

+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------+
| Integration                                                                                                                                                                       | Dependencies                       |
+===================================================================================================================================================================================+====================================+
| `AllenNLP <https://github.com/optuna/optuna/tree/master/optuna/integration/allennlp>`__                                                                                           | allennlp, torch, psutil, jsonnet   |
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------+
| `BoTorch <https://github.com/optuna/optuna/blob/master/optuna/integration/botorch.py>`__                                                                                          | botorch, gpytorch, torch           |
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------+
| `CatBoost <https://github.com/optuna/optuna/blob/master/optuna/integration/catboost.py>`__                                                                                        | catboost                           |
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------+
| `ChainerMN <https://github.com/optuna/optuna/blob/master/optuna/integration/chainermn.py>`__                                                                                      | chainermn                          |
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------+
| `Chainer <https://github.com/optuna/optuna/blob/master/optuna/integration/chainer.py>`__                                                                                          | chainer                            |
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------+
| `pycma <https://github.com/optuna/optuna/blob/master/optuna/integration/cma.py>`__                                                                                                | cma                                |
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------+
| `Dask <https://github.com/optuna/optuna/blob/master/optuna/integration/dask.py>`__                                                                                                | distributed                        |
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------+
| `FastAI <https://github.com/optuna/optuna/blob/master/optuna/integration/fastaiv2.py>`__                                                                                          | fastai                             |
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------+
| `Keras <https://github.com/optuna/optuna/blob/master/optuna/integration/keras.py>`__                                                                                              | keras                              |
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------+
| `LightGBMTuner <https://github.com/optuna/optuna/blob/master/optuna/integration/lightgbm.py>`__                                                                                   | lightgbm, scikit-learn             |
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------+
| `LightGBMPruningCallback <https://github.com/optuna/optuna/blob/master/optuna/integration/lightgbm.py>`__                                                                         | lightgbm                           |
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------+
| `MLflow <https://github.com/optuna/optuna/blob/master/optuna/integration/mlflow.py>`__                                                                                            | mlflow                             |
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------+
| `MXNet <https://github.com/optuna/optuna/blob/master/optuna/integration/mxnet.py>`__                                                                                              | mxnet                              |
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------+
| PyTorch `Distributed <https://github.com/optuna/optuna/blob/master/optuna/integration/pytorch_distributed.py>`__                                                                  | torch                              |
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------+
| PyTorch (`Ignite <https://github.com/optuna/optuna/blob/master/optuna/integration/pytorch_ignite.py>`__)                                                                          | pytorch-ignite                     |
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------+
| PyTorch (`Lightning <https://github.com/optuna/optuna/blob/master/optuna/integration/pytorch_lightning.py>`__)                                                                    | pytorch-lightning                  |
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------+
| `SHAP <https://github.com/optuna/optuna/blob/master/optuna/integration/shap.py>`__                                                                                                | scikit-learn, shap                 |
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------+
| `Scikit-learn <https://github.com/optuna/optuna/blob/master/optuna/integration/sklearn.py>`__                                                                                     | pandas, scipy, scikit-learn        |
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------+
| `SKorch <https://github.com/optuna/optuna/blob/master/optuna/integration/skorch.py>`__                                                                                            | skorch                             |
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------+
| `TensorBoard <https://github.com/optuna/optuna/blob/master/optuna/integration/tensorboard.py>`__                                                                                  | tensorboard, tensorflow            |
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------+
| `TensorFlow <https://github.com/optuna/optuna/blob/master/optuna/integration/tensorflow.py>`__                                                                                    | tensorflow, tensorflow-estimator   |
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------+
| `TensorFlow + Keras <https://github.com/optuna/optuna/blob/master/optuna/integration/tfkeras.py>`__                                                                               | tensorflow                         |
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------+
| `Weights & Biases <https://github.com/optuna/optuna/blob/master/optuna/integration/wandb.py>`__                                                                                   | wandb                              |
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------+
| `XGBoost <https://github.com/optuna/optuna/blob/master/optuna/integration/xgboost.py>`__                                                                                          | xgboost                            |
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------+
