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
