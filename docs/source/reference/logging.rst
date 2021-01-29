.. module:: optuna.logging

optuna.logging
==============

The :mod:`~optuna.logging` module implements logging using the Python ``logging`` package. Library users may be especially interested in setting verbosity levels using :func:`~optuna.logging.set_verbosity` to one of ``optuna.logging.CRITICAL`` (aka ``optuna.logging.FATAL``), ``optuna.logging.ERROR``, ``optuna.logging.WARNING`` (aka ``optuna.logging.WARN``), ``optuna.logging.INFO``, or ``optuna.logging.DEBUG``.


.. autosummary::
   :toctree: generated/
   :nosignatures:

   optuna.logging.get_verbosity
   optuna.logging.set_verbosity
   optuna.logging.disable_default_handler
   optuna.logging.enable_default_handler
   optuna.logging.disable_propagation
   optuna.logging.enable_propagation
