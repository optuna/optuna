from __future__ import annotations

import os
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

    .. note:
        Optuna does not provide any API to save artifact store information linked to each artifact,
        so we kindly ask users to save the artifact store information individually.

    """
    if os.path.exists(file_path):
        raise FileExistsError(f"File already exists: {file_path}")

    with artifact_store.open_reader(artifact_id) as reader, open(file_path, "wb") as writer:
        shutil.copyfileobj(reader, writer)
