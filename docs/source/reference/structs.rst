.. module:: optuna.structs

optuna.structs
==============

This module is deprecated, with former functionality moved to :class:`optuna.trial` and :class:`optuna.study`.

.. autoclass:: TrialState
    :members:

    .. deprecated:: 1.4.0
            This class is deprecated. Please use :class:`~optuna.trial.TrialState` instead.

.. autoclass:: StudyDirection
    :members:

    .. deprecated:: 1.4.0
            This class is deprecated. Please use :class:`~optuna.study.StudyDirection` instead.

.. autoclass:: FrozenTrial
    :members:
    :exclude-members: system_attrs, trial_id

.. autoclass:: StudySummary
    :members:
