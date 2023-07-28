from __future__ import annotations

import io
import shutil
from typing import TYPE_CHECKING

import boto3
from botocore.exceptions import ClientError

from optuna.artifacts.exceptions import ArtifactNotFound


if TYPE_CHECKING:
    from typing import BinaryIO

    from mypy_boto3_s3 import S3Client


class Boto3ArtifactStore:
    """An artifact backend for Boto3."""

    def __init__(
        self, bucket_name: str, client: S3Client | None = None, *, avoid_buf_copy: bool = False
    ) -> None:
        self.bucket = bucket_name
        self.client = client or boto3.client("s3")
        # This flag is added to avoid that upload_fileobj() method of Boto3 client may close the
        # source file object. See https://github.com/boto/boto3/issues/929.
        self._avoid_buf_copy = avoid_buf_copy

    def open_reader(self, artifact_id: str) -> BinaryIO:
        try:
            obj = self.client.get_object(Bucket=self.bucket, Key=artifact_id)
        except ClientError as e:
            if _is_not_found_error(e):
                raise ArtifactNotFound("not found") from e
            raise
        body = obj.get("Body")
        assert body is not None
        return body

    def write(self, artifact_id: str, content_body: BinaryIO) -> None:
        fsrc: BinaryIO = content_body
        if not self._avoid_buf_copy:
            buf = io.BytesIO()
            shutil.copyfileobj(content_body, buf)
            buf.seek(0)
            fsrc = buf
        self.client.upload_fileobj(fsrc, self.bucket, artifact_id)

    def remove(self, artifact_id: str) -> None:
        try:
            self.client.delete_object(Bucket=self.bucket, Key=artifact_id)
        except ClientError as e:
            if _is_not_found_error(e):
                raise ArtifactNotFound("not found") from e
            raise


def _is_not_found_error(e: ClientError) -> bool:
    error_code = e.response.get("Error", {}).get("Code")
    http_status_code = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return error_code == "NoSuchKey" or http_status_code == 404


if TYPE_CHECKING:
    # A mypy-runtime assertion to ensure that Boto3ArtifactStore implements all abstract methods
    # in ArtifactStore.
    from ._protocol import ArtifactStore

    _: ArtifactStore = Boto3ArtifactStore("")
