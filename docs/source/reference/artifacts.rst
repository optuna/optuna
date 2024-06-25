.. module:: optuna.artifacts

optuna.artifacts
================

The :mod:`~optuna.artifacts` module provides the way to manage artifacts (output files) in Optuna.
:ref:`The tutorial<artifact_tutorial>` is also available.

.. note::
   The methods defined in each ``ArtifactStore`` are not intended to be directly accessed by library users.

.. note::
   As ``ArtifactStore`` does not officially provide user API for artifact removal, please refer to :ref:`FAQ<remove for artifact store>` for the hack.

.. autoclass:: optuna.artifacts.FileSystemArtifactStore
   :no-members:

.. autoclass:: optuna.artifacts.Boto3ArtifactStore
   :no-members:

.. autoclass:: optuna.artifacts.GCSArtifactStore
   :no-members:

.. autoclass:: optuna.artifacts.Backoff
   :no-members:

.. autofunction:: optuna.artifacts.upload_artifact

.. autofunction:: optuna.artifacts.get_all_artifact_meta

.. autofunction:: optuna.artifacts.download_artifact
