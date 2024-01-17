.. module:: optuna.artifacts

optuna.artifacts
================

The :mod:`~optuna.artifacts` module provides the way to manage artifacts (output files) in Optuna.

.. autoclass:: optuna.artifacts.FileSystemArtifactStore
   :no-members:

.. autoclass:: optuna.artifacts.Boto3ArtifactStore
   :no-members:

.. autoclass:: optuna.artifacts.GCSArtifactStore
   :no-members:

.. autoclass:: optuna.artifacts.Backoff
   :no-members:

.. autofunction:: optuna.artifacts.upload_artifact
