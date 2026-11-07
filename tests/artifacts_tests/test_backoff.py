import io
from unittest.mock import patch
import uuid

import pytest

from optuna.artifacts import Backoff
from optuna.artifacts.exceptions import ArtifactNotFound

from .stubs import FailArtifactStore
from .stubs import InMemoryArtifactStore


def test_backoff_time() -> None:
    backend = Backoff(
        backend=FailArtifactStore(),
        min_delay=0.1,
        multiplier=10,
        max_delay=10,
    )
    assert backend._get_sleep_secs(0) == 0.1
    assert backend._get_sleep_secs(1) == 1
    assert backend._get_sleep_secs(2) == 10


def test_read_and_write() -> None:
    artifact_id = f"test-{uuid.uuid4()}"
    dummy_content = b"Hello World"

    backend = Backoff(
        backend=InMemoryArtifactStore(),
        min_delay=0.1,
        multiplier=10,
        max_delay=10,
    )
    backend.write(artifact_id, io.BytesIO(dummy_content))
    with backend.open_reader(artifact_id) as f:
        actual = f.read()
    assert actual == dummy_content


def test_remove() -> None:
    artifact_id = f"test-{uuid.uuid4()}"
    artifact_store = InMemoryArtifactStore()
    backend = Backoff(backend=artifact_store, max_retries=3)
    backend.write(artifact_id, io.BytesIO(b"content"))

    with patch.object(artifact_store, "remove", wraps=artifact_store.remove) as remove_mock:
        backend.remove(artifact_id)

    remove_mock.assert_called_once_with(artifact_id)

    with pytest.raises(ArtifactNotFound):
        backend.open_reader(artifact_id)


@pytest.mark.parametrize("initial_position", [0, 4])
@pytest.mark.parametrize("read_size", [2, -1])
@pytest.mark.parametrize("failures", [1, 2])
def test_write_retry_preserves_stream_position(
    initial_position: int, read_size: int, failures: int
) -> None:
    artifact_store = InMemoryArtifactStore()
    backend = Backoff(artifact_store, max_retries=failures + 1)
    content = b"skipartifact content"
    content_body = io.BytesIO(content)
    content_body.seek(initial_position)
    original_write = artifact_store.write
    attempts = 0

    def flaky_write(artifact_id: str, body: io.BytesIO) -> None:
        nonlocal attempts
        attempts += 1
        if attempts <= failures:
            body.read(read_size)
            raise OSError("temporary upload failure")
        original_write(artifact_id, body)

    with (
        patch.object(artifact_store, "write", side_effect=flaky_write) as write_mock,
        patch("optuna.artifacts._backoff.time.sleep") as sleep_mock,
    ):
        backend.write("artifact", content_body)

    with artifact_store.open_reader("artifact") as reader:
        assert reader.read() == content[initial_position:]
    assert write_mock.call_count == failures + 1
    assert sleep_mock.call_count == failures


def test_write_nonseekable_stream_without_retry() -> None:
    class NonSeekableStream(io.BytesIO):
        def tell(self) -> int:
            raise io.UnsupportedOperation("not seekable")

        def seek(self, offset: int, whence: int = 0) -> int:
            raise io.UnsupportedOperation("not seekable")

    artifact_store = InMemoryArtifactStore()
    backend = Backoff(artifact_store)
    backend.write("artifact", NonSeekableStream(b"content"))
    with artifact_store.open_reader("artifact") as reader:
        assert reader.read() == b"content"
