.. module:: optuna.terminator

optuna.terminator
=================

The :mod:`~optuna.terminator` module implements a mechanism for automatically terminating the optimization process, accompanied by a callback class for the termination and evaluators for the estimated room for improvement in the optimization and statistical error of the objective function. The terminator determines to terminate the optimization process when the improvement is estimated to be smaller than the statistical error.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   optuna.terminator.terminator.Terminator
   optuna.terminator.callback.TerminatorCallback