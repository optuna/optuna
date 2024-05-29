from __future__ import annotations

import shutil

from optuna._experimental import experimental_func
from optuna.artifacts._protocol import ArtifactStore


@experimental_func("4.0.0")
def download_artifact(artifact_store: ArtifactStore, artifact_id: str, file_path: str) -> None:
    """Download an artifact from the artifact store.

    Args:
        artifact_store:
            An artifact store.
        artifact_id:
            The identifier of the artifact to download.
        file_path:
            A path to save the downloaded artifact.
    """
    with artifact_store.open_reader(artifact_id) as reader, open(file_path, "wb") as writer:
        shutil.copyfileobj(reader, writer)
