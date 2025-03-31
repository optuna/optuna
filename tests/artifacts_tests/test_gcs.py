from __future__ import annotations

import contextlib
import io
import os
from typing import TYPE_CHECKING
from unittest.mock import patch

import google.cloud.storage
import pytest

from optuna.artifacts import GCSArtifactStore
from optuna.artifacts.exceptions import ArtifactNotFound


if TYPE_CHECKING:
    from collections.abc import Iterator


_MOCK_BUCKET_CONTENT: dict[str, bytes] = dict()


class MockBucket:
    def get_blob(self, blob_name: str) -> "MockBlob" | None:
        if blob_name in _MOCK_BUCKET_CONTENT:
            return MockBlob(blob_name)
        else:
            return None

    def blob(self, blob_name: str) -> "MockBlob":
        return MockBlob(blob_name)

    def delete_blob(self, blob_name: str) -> None:
        del _MOCK_BUCKET_CONTENT[blob_name]

    def list_blobs(self) -> Iterator["MockBlob"]:
        for blob_name in _MOCK_BUCKET_CONTENT.keys():
            yield MockBlob(blob_name)


class MockBlob:
    def __init__(self, blob_name: str) -> None:
        self.blob_name = blob_name

    def download_as_bytes(self) -> bytes:
        return _MOCK_BUCKET_CONTENT[self.blob_name]

    def upload_from_string(self, data: bytes) -> None:
        _MOCK_BUCKET_CONTENT[self.blob_name] = data


@contextlib.contextmanager
def init_mock_client() -> Iterator[None]:
    # In case we fail to patch `google.cloud.storage.Client`, we deliberately set an invalid
    # credential path so that we do not accidentally access GCS.
    # Note that this is not a perfect measure; it can become ineffective in future when the
    # mechanism for finding the default credential is changed in the Cloud Storage API.
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/dev/null"

    with patch("google.cloud.storage.Client") as MockClient:
        instance = MockClient.return_value

        def bucket(name: str) -> MockBucket:
            assert name == "mock-bucket"
            return MockBucket()

        instance.bucket.side_effect = bucket
        yield


@pytest.mark.parametrize("explicit_client", [False, True])
def test_upload_download(explicit_client: bool) -> None:
    with init_mock_client():
        bucket_name = "mock-bucket"
        if explicit_client:
            backend = GCSArtifactStore(bucket_name, google.cloud.storage.Client())
        else:
            backend = GCSArtifactStore(bucket_name)

        artifact_id = "dummy-uuid"
        dummy_content = b"Hello World"
        buf = io.BytesIO(dummy_content)

        backend.write(artifact_id, buf)

        client = google.cloud.storage.Client()
        assert len(list(client.bucket(bucket_name).list_blobs())) == 1

        blob = client.bucket(bucket_name).blob(artifact_id)
        assert blob.download_as_bytes() == dummy_content

        with backend.open_reader(artifact_id) as f:
            actual = f.read()
        assert actual == dummy_content


def test_remove() -> None:
    with init_mock_client():
        bucket_name = "mock-bucket"
        backend = GCSArtifactStore(bucket_name)
        client = google.cloud.storage.Client()

        artifact_id = "dummy-uuid"
        backend.write(artifact_id, io.BytesIO(b"Hello"))
        assert len(list(client.bucket(bucket_name).list_blobs())) == 1

        backend.remove(artifact_id)
        assert len(list(client.bucket(bucket_name).list_blobs())) == 0


def test_file_not_found_exception() -> None:
    with init_mock_client():
        bucket_name = "mock-bucket"
        backend = GCSArtifactStore(bucket_name)

        with pytest.raises(ArtifactNotFound):
            backend.open_reader("not-found-id")
