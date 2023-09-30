from optuna.exceptions import OptunaError


class ArtifactNotFound(OptunaError):
    """Exception raised when an artifact is not found.

    It is typically raised while calling
    :meth:`~optuna.artifact.protocol.ArtifactBackend.open_reader` or
    :meth:`~optuna.artifact.protocol.ArtifactBackend.remove` methods.
    """

    ...
