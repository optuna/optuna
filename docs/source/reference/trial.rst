optuna.trial
============

The :mod:`optuna.trial` module contains :class:`~optuna.trial.Trial` related classes and functions. A :class:`~optuna.trial.Trial` instance represents a process of evaluating an objective function. This instance is passed to an objective function and provides interfaces to get parameter suggestion, manage the trial's state, and set/get user-defined attributes of the trial, so that optuna users can define a custome objective function through the interfaces. Basically, optuna users only treats it in their custom objective functions.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   optuna.trial.Trial
   optuna.trial.FixedTrial
   optuna.trial.FrozenTrial
   optuna.trial.TrialState
   optuna.trial.create_trial
