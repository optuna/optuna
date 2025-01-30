from __future__ import annotations

import io
from typing import TYPE_CHECKING

import boto3
from moto import mock_aws
import pytest

from optuna.artifacts import Boto3ArtifactStore
from optuna.artifacts.exceptions import ArtifactNotFound


if TYPE_CHECKING:
    from collections.abc import Iterator

    from mypy_boto3_s3 import S3Client
    from typing_extensions import Annotated

    # TODO(Shinichi) import Annotated from typing after python 3.8 support is dropped.


@pytest.fixture()
def init_mock_client() -> Iterator[tuple[str, S3Client]]:
    with mock_aws():
        # Runs before each test
        bucket_name = "moto-bucket"
        s3_client = boto3.client("s3")
        s3_client.create_bucket(Bucket=bucket_name)

        yield bucket_name, s3_client

        # Runs after each test
        objects = s3_client.list_objects(Bucket=bucket_name).get("Contents", [])
        if objects:
            s3_client.delete_objects(
                Bucket=bucket_name,
                Delete={"Objects": [{"Key": obj["Key"] for obj in objects}], "Quiet": True},
            )
        s3_client.delete_bucket(Bucket=bucket_name)


@pytest.mark.parametrize("avoid_buf_copy", [True, False])
def test_upload_download(
    init_mock_client: Annotated[tuple[str, S3Client], pytest.fixture],
    avoid_buf_copy: bool,
) -> None:
    bucket_name, s3_client = init_mock_client
    backend = Boto3ArtifactStore(bucket_name, avoid_buf_copy=avoid_buf_copy)

    artifact_id = "dummy-uuid"
    dummy_content = b"Hello World"
    buf = io.BytesIO(dummy_content)

    backend.write(artifact_id, buf)
    assert len(s3_client.list_objects(Bucket=bucket_name)["Contents"]) == 1

    obj = s3_client.get_object(Bucket=bucket_name, Key=artifact_id)
    assert obj["Body"].read() == dummy_content

    with backend.open_reader(artifact_id) as f:
        actual = f.read()
    assert actual == dummy_content
    if avoid_buf_copy is False:
        assert buf.closed is False


def test_remove(init_mock_client: Annotated[tuple[str, S3Client], pytest.fixture]) -> None:
    bucket_name, s3_client = init_mock_client
    backend = Boto3ArtifactStore(bucket_name)

    artifact_id = "dummy-uuid"
    backend.write(artifact_id, io.BytesIO(b"Hello"))
    objects = s3_client.list_objects(Bucket=bucket_name)["Contents"]
    assert len([obj for obj in objects if obj["Key"] == artifact_id]) == 1

    backend.remove(artifact_id)
    objects = s3_client.list_objects(Bucket=bucket_name).get("Contents", [])
    assert len([obj for obj in objects if obj["Key"] == artifact_id]) == 0


def test_file_not_found_exception(
    init_mock_client: Annotated[tuple[str, S3Client], pytest.fixture],
) -> None:
    bucket_name, _ = init_mock_client
    backend = Boto3ArtifactStore(bucket_name)

    with pytest.raises(ArtifactNotFound):
        backend.open_reader("not-found-id")
