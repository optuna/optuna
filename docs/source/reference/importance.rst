.. module:: optuna.importance

optuna.importance
=================

The :mod:`~optuna.importance` module provides functionality for evaluating hyperparameter
importances based on completed trials in a given study.
The utility function :func:`~optuna.importance.get_param_importances` takes a
:class:`~optuna.study.Study` and optional evaluator as two of its inputs.
The evaluator must derive from :class:`~optuna.importance.BaseImportanceEvaluator`, and
:class:`~optuna.importance.PedAnovaImportanceEvaluator` is used by default when not passed in.
Users implementing custom evaluators should refer to
:class:`~optuna.importance.PedAnovaImportanceEvaluator`,
:class:`~optuna.importance.FanovaImportanceEvaluator`, or
:class:`~optuna.importance.MeanDecreaseImpurityImportanceEvaluator`,
as a guide, paying close attention to the format of the return value from the evaluator's
``evaluate`` function.

.. note::

   For detailed usage, please refer to :func:`~optuna.importance.get_param_importances`.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   get_param_importances
   FanovaImportanceEvaluator
   MeanDecreaseImpurityImportanceEvaluator
   PedAnovaImportanceEvaluator
