.. module:: optuna.study

optuna.study
============

The :mod:`~optuna.study` module implements the :class:`~optuna.study.Study` object and related functions. A public constructor is available for the :class:`~optuna.study.Study` class, but direct use of this constructor is not recommended. Instead, library users should create and load a :class:`~optuna.study.Study` using :func:`~optuna.study.create_study` and :func:`~optuna.study.load_study` respectively.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Study
   create_study
   load_study
   delete_study
   copy_study
   get_all_study_names
   get_all_study_summaries
   MaxTrialsCallback
   StudyDirection
   StudySummary
