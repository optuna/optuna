from __future__ import annotations

import copy
import io
import shutil
import threading
from typing import TYPE_CHECKING

from optuna.artifacts.exceptions import ArtifactNotFound


if TYPE_CHECKING:
    from typing import BinaryIO


class FailArtifactStore:
    def open_reader(self, artifact_id: str) -> BinaryIO:
        raise Exception("something error raised")

    def write(self, artifact_id: str, content_body: BinaryIO) -> None:
        raise Exception("something error raised")

    def remove(self, artifact_id: str) -> None:
        raise Exception("something error raised")


class InMemoryArtifactStore:
    def __init__(self) -> None:
        self._data: dict[str, io.BytesIO] = {}
        self._lock = threading.Lock()

    def open_reader(self, artifact_id: str) -> BinaryIO:
        with self._lock:
            data = self._data.get(artifact_id)
            if data is None:
                raise ArtifactNotFound("not found")
            return copy.deepcopy(data)

    def write(self, artifact_id: str, content_body: BinaryIO) -> None:
        buf = io.BytesIO()
        shutil.copyfileobj(content_body, buf)
        buf.seek(0)
        with self._lock:
            self._data[artifact_id] = buf

    def remove(self, artifact_id: str) -> None:
        with self._lock:
            if artifact_id not in self._data:
                raise ArtifactNotFound("not found")
            del self._data[artifact_id]


if TYPE_CHECKING:
    from optuna.artifacts._protocol import ArtifactStore

    _fail: ArtifactStore = FailArtifactStore()
    _inmemory: ArtifactStore = InMemoryArtifactStore()
