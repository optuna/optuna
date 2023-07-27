from __future__ import annotations

from collections.abc import Generator
import io

import boto3
from moto import mock_s3
import pytest

from optuna.artifacts import Boto3ArtifactStore
from optuna.artifacts.exceptions import ArtifactNotFound


class TestBoto3ArtifactStore:
    @pytest.fixture(autouse=True)
    def init_mock_client(self) -> Generator:
        with mock_s3():
            # Runs before each test
            self.bucket_name = "moto-bucket"
            self.s3_client = boto3.client("s3")
            self.s3_client.create_bucket(Bucket=self.bucket_name)

            yield  # Runs test

            # Runs after each test
            objects = self.s3_client.list_objects(Bucket=self.bucket_name).get("Contents", [])
            if objects:
                self.s3_client.delete_objects(
                    Bucket=self.bucket_name,
                    Delete={"Objects": [{"Key": obj["Key"] for obj in objects}], "Quiet": True},
                )
            self.s3_client.delete_bucket(Bucket=self.bucket_name)

    def test_upload_download(self) -> None:
        artifact_id = "dummy-uuid"
        dummy_content = b"Hello World"
        buf = io.BytesIO(dummy_content)

        backend = Boto3ArtifactStore(self.bucket_name)
        backend.write(artifact_id, buf)
        assert len(self.s3_client.list_objects(Bucket=self.bucket_name)["Contents"]) == 1
        obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=artifact_id)
        assert obj["Body"].read() == dummy_content

        with backend.open_reader(artifact_id) as f:
            actual = f.read()
        assert actual == dummy_content
        assert buf.closed is False

    def test_remove(self) -> None:
        artifact_id = "dummy-uuid"
        backend = Boto3ArtifactStore(self.bucket_name)
        backend.write(artifact_id, io.BytesIO(b"Hello"))
        objects = self.s3_client.list_objects(Bucket=self.bucket_name)["Contents"]
        assert len([obj for obj in objects if obj["Key"] == artifact_id]) == 1

        backend.remove(artifact_id)
        objects = self.s3_client.list_objects(Bucket=self.bucket_name).get("Contents", [])
        assert len([obj for obj in objects if obj["Key"] == artifact_id]) == 0

    def test_file_not_found_exception(self) -> None:
        backend = Boto3ArtifactStore(self.bucket_name)
        with pytest.raises(ArtifactNotFound):
            backend.open_reader("not-found-id")
