from __future__ import annotations

import os
from pathlib import Path
import shutil
from typing import TYPE_CHECKING

from optuna._experimental import experimental_class
from optuna.artifacts.exceptions import ArtifactNotFound


if TYPE_CHECKING:
    from typing import BinaryIO


@experimental_class("3.3.0")
class FileSystemArtifactStore:
    """An artifact backend for file systems.

    Args:
        base_path:
            The base path to a directory to store artifacts.
    """

    def __init__(self, base_path: str | Path) -> None:
        if isinstance(base_path, str):
            base_path = Path(base_path)
        # TODO(Shinichi): Check if the base_path is valid directory.
        self._base_path = base_path

    def open_reader(self, artifact_id: str) -> BinaryIO:
        filepath = os.path.join(self._base_path, artifact_id)
        try:
            f = open(filepath, "rb")
        except FileNotFoundError as e:
            raise ArtifactNotFound("not found") from e
        return f

    def write(self, artifact_id: str, content_body: BinaryIO) -> None:
        filepath = os.path.join(self._base_path, artifact_id)
        with open(filepath, "wb") as f:
            shutil.copyfileobj(content_body, f)

    def remove(self, artifact_id: str) -> None:
        filepath = os.path.join(self._base_path, artifact_id)
        try:
            os.remove(filepath)
        except FileNotFoundError as e:
            raise ArtifactNotFound("not found") from e


if TYPE_CHECKING:
    # A mypy-runtime assertion to ensure that LocalArtifactBackend
    # implements all abstract methods in ArtifactBackendProtocol.
    from ._protocol import ArtifactStore

    _: ArtifactStore = FileSystemArtifactStore("")
