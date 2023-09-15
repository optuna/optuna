import io
from pathlib import Path

import pytest

from optuna.artifacts import FileSystemArtifactStore
from optuna.artifacts.exceptions import ArtifactNotFound


def test_upload_download(tmp_path: Path) -> None:
    artifact_id = "dummy-uuid"
    dummy_content = b"Hello World"
    backend = FileSystemArtifactStore(tmp_path)
    backend.write(artifact_id, io.BytesIO(dummy_content))
    with backend.open_reader(artifact_id) as f:
        actual = f.read()
    assert actual == dummy_content


def test_remove(tmp_path: Path) -> None:
    artifact_id = "dummy-uuid"
    dummy_content = b"Hello World"
    backend = FileSystemArtifactStore(tmp_path)

    backend.write(artifact_id, io.BytesIO(dummy_content))
    objects = list(tmp_path.glob("*"))
    assert len([obj for obj in objects if obj.name == artifact_id]) == 1

    backend.remove(artifact_id)
    objects = list(tmp_path.glob("*"))
    assert len([obj for obj in objects if obj.name == artifact_id]) == 0


def test_file_not_found(tmp_path: str) -> None:
    backend = FileSystemArtifactStore(tmp_path)

    with pytest.raises(ArtifactNotFound):
        backend.open_reader("not-found-id")

    with pytest.raises(ArtifactNotFound):
        backend.remove("not-found-id")
