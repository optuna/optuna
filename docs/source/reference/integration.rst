.. module:: optuna.integration

Integration
===========

.. autoclass:: ChainerPruningExtension
    :members:

.. autoclass:: ChainerMNStudy
    :members:

.. autoclass:: CmaEsSampler
    :members:
    :exclude-members: infer_relative_search_space, sample_relative, sample_independent

.. autoclass:: FastAIPruningCallback
    :members:
    :exclude-members: on_epoch_end

.. autoclass:: PyTorchIgnitePruningHandler
    :members:

.. autoclass:: KerasPruningCallback
    :members:

.. autoclass:: LightGBMPruningCallback
    :members:

.. autofunction:: optuna.integration.lightgbm.train

.. autoclass:: optuna.integration.lightgbm.LightGBMTuner
    :members:
    :exclude-members: sample_train_set

.. autoclass:: MXNetPruningCallback
    :members:

.. autoclass:: PyTorchLightningPruningCallback
    :members:

.. autoclass:: SkoptSampler
    :members:
    :exclude-members: infer_relative_search_space, sample_relative, sample_independent

.. autoclass:: TensorFlowPruningHook
    :members:

.. autoclass:: TFKerasPruningCallback
    :members:

.. autoclass:: XGBoostPruningCallback
    :members:

.. autoclass:: OptunaSearchCV
    :members:
