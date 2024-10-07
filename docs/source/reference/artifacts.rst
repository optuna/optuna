.. module:: optuna.artifacts

optuna.artifacts
================

The :mod:`~optuna.artifacts` module provides the way to manage artifacts (output files) in Optuna.
Please also check :ref:`artifact_tutorial` and `our article <https://medium.com/optuna/file-management-during-llm-large-language-model-trainings-by-optuna-v4-0-0-artifact-store-5bdd5112f3c7>`__.
The storages covered by :mod:`~optuna.artifacts` are the following:

+-------------------------+----------------------------------------+
| Class Name              |           Supported Storage            |
+=========================+========================================+
| FileSystemArtifactStore | Local File System, Network File System |
+-------------------------+----------------------------------------+
| Boto3ArtifactStore      | Amazon S3 Compatible Object Storage    |
+-------------------------+----------------------------------------+
| GCSArtifactStore        | Google Cloud Storage                   |
+-------------------------+----------------------------------------+

.. note::
   The methods defined in each ``ArtifactStore`` are not intended to be directly accessed by library users.

.. note::
   As ``ArtifactStore`` does not officially provide user API for artifact removal, please refer to :ref:`remove_for_artifact_store` for the hack.

.. autoclass:: optuna.artifacts.FileSystemArtifactStore
   :no-members:

.. autoclass:: optuna.artifacts.Boto3ArtifactStore
   :no-members:

.. autoclass:: optuna.artifacts.GCSArtifactStore
   :no-members:

.. autoclass:: optuna.artifacts.Backoff
   :no-members:

.. autoclass:: optuna.artifacts.ArtifactMeta
   :no-members:

.. autofunction:: optuna.artifacts.upload_artifact

.. autofunction:: optuna.artifacts.get_all_artifact_meta

.. autofunction:: optuna.artifacts.download_artifact
