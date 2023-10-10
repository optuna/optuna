.. module:: optuna.terminator

optuna.terminator
=================

The :mod:`~optuna.terminator` module implements a mechanism for automatically terminating the optimization process, accompanied by a callback class for the termination and evaluators for the estimated room for improvement in the optimization and statistical error of the objective function. The terminator stops the optimization process when the estimated potential improvement is smaller than the statistical error.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   optuna.terminator.BaseTerminator
   optuna.terminator.Terminator
   optuna.terminator.BaseImprovementEvaluator
   optuna.terminator.RegretBoundEvaluator
   optuna.terminator.BestValueStagnationEvaluator
   optuna.terminator.BaseErrorEvaluator
   optuna.terminator.CrossValidationErrorEvaluator
   optuna.terminator.StaticErrorEvaluator
   optuna.terminator.TerminatorCallback
   optuna.terminator.report_cross_validation_scores