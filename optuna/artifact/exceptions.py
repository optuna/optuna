from optuna.exceptions import OptunaError


class ArtifactNotFound(OptunaError):
    """Exception raised when an artifact is not found.

    It is typically raised while calling
    :meth:`~optuna_dashboard.artifact.protocol.ArtifactBackend.open` or
    :meth:`~optuna_dashboard.artifact.protocol.ArtifactBackend.remove` methods.
    """

    ...
