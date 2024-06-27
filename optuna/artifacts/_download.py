from __future__ import annotations

import os
import shutil

from optuna._experimental import experimental_func
from optuna.artifacts._protocol import ArtifactStore


@experimental_func("4.0.0")
def download_artifact(file_path: str, artifact_store: ArtifactStore, artifact_id: str) -> None:
    """Download an artifact from the artifact store.

    Args:
        artifact_id:
            The identifier of the artifact to download.
        artifact_store:
            An artifact store.
        file_path:
            A path to save the downloaded artifact.
    """
    if os.path.exists(file_path):
        raise FileExistsError(f"File already exists: {file_path}")

    with artifact_store.open_reader(artifact_id) as reader, open(file_path, "wb") as writer:
        shutil.copyfileobj(reader, writer)
