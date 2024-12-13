.. module:: optuna.trial

optuna.trial
============

The :mod:`~optuna.trial` module contains :class:`~optuna.trial.Trial` related classes and functions.

A :class:`~optuna.trial.Trial` instance represents a process of evaluating an objective function. This instance is passed to an objective function and provides interfaces to get parameter suggestion, manage the trial's state, and set/get user-defined attributes of the trial, so that Optuna users can define a custom objective function through the interfaces. Basically, Optuna users only use it in their custom objective functions.



.. autoclass:: optuna.trial.Trial

   .. rubric:: Methods

   .. autosummary::
   
      ~Trial.report
      ~Trial.set_system_attr
      ~Trial.set_user_attr
      ~Trial.should_prune
      ~Trial.suggest_categorical
      ~Trial.suggest_discrete_uniform
      ~Trial.suggest_float
      ~Trial.suggest_int
      ~Trial.suggest_loguniform
      ~Trial.suggest_uniform
   
   .. rubric:: Attributes

   .. autosummary::
   
      ~Trial.datetime_start
      ~Trial.distributions
      ~Trial.number
      ~Trial.params
      ~Trial.relative_params
      ~Trial.system_attrs
      ~Trial.user_attrs


.. autoclass:: optuna.trial.FixedTrial

   .. rubric:: Methods

   .. autosummary::
   
      ~FixedTrial.report
      ~FixedTrial.set_system_attr
      ~FixedTrial.set_user_attr
      ~FixedTrial.should_prune
      ~FixedTrial.suggest_categorical
      ~FixedTrial.suggest_discrete_uniform
      ~FixedTrial.suggest_float
      ~FixedTrial.suggest_int
      ~FixedTrial.suggest_loguniform
      ~FixedTrial.suggest_uniform
   
   
   .. rubric:: Attributes

   .. autosummary::
   
      ~FixedTrial.datetime_start
      ~FixedTrial.distributions
      ~FixedTrial.number
      ~FixedTrial.params
      ~FixedTrial.system_attrs
      ~FixedTrial.user_attrs


.. autoclass:: optuna.trial.FrozenTrial

   .. rubric:: Methods

   .. autosummary::
   
      ~FrozenTrial.report
      ~FrozenTrial.set_system_attr
      ~FrozenTrial.set_user_attr
      ~FrozenTrial.should_prune
      ~FrozenTrial.suggest_categorical
      ~FrozenTrial.suggest_discrete_uniform
      ~FrozenTrial.suggest_float
      ~FrozenTrial.suggest_int
      ~FrozenTrial.suggest_loguniform
      ~FrozenTrial.suggest_uniform
   
   .. rubric:: Attributes

   .. autosummary::
   
      ~FrozenTrial.datetime_start
      ~FrozenTrial.distributions
      ~FrozenTrial.duration
      ~FrozenTrial.last_step
      ~FrozenTrial.number
      ~FrozenTrial.params
      ~FrozenTrial.system_attrs
      ~FrozenTrial.user_attrs
      ~FrozenTrial.value
      ~FrozenTrial.values


.. autoclass:: optuna.trial.TrialState
   
   .. rubric:: Methods

   .. autosummary::
   
      ~TrialState.is_finished
   
   .. rubric:: Attributes

   .. autosummary::
   
      ~TrialState.RUNNING
      ~TrialState.COMPLETE
      ~TrialState.PRUNED
      ~TrialState.FAIL
      ~TrialState.WAITING


.. autofunction:: optuna.trial.create_trial

