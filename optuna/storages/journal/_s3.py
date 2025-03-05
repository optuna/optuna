from __future__ import annotations

import json
import logging
import random
import time
from typing import Any

from optuna._experimental import experimental_class
from optuna._imports import try_import
from optuna.storages.journal._base import BaseJournalBackend


with try_import() as _imports:
    import boto3

logger = logging.getLogger(__name__)

_MAX_RETRY = 50
_INITIAL_BACKOFF_RANGE = (0.09, 0.11)
_EXPONENTIAL_BACKOFF_RANGE = (1.0, 1.5)


@experimental_class("4.3.0")
class JournalS3Backend(BaseJournalBackend):
    """S3 storage class for Journal log backend.

    Args:
        s3_path: The S3 path for an object to store the logs. This should have "s3://" prefix.
    """

    def __init__(self, s3_path: str) -> None:
        _imports.check()

        if not s3_path.startswith("s3://") or "/" not in s3_path.removeprefix("s3://"):
            raise ValueError(f"Invalid S3 path: {s3_path}")
        self._bucket, self._key = s3_path.removeprefix("s3://").split("/", 1)

        self._s3 = boto3.client("s3")

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        del state["_s3"]
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._s3 = boto3.client("s3")

    def read_logs(self, log_number_from: int) -> list[dict[str, Any]]:
        """Read logs with a log number greater than or equal to ``log_number_from``."""
        try:
            response = self._s3.get_object(Bucket=self._bucket, Key=self._key)
        except self._s3.exceptions.NoSuchKey:
            return []
        logs = json.loads(response["Body"].read())
        return logs[log_number_from:]

    def append_logs(self, logs: list[dict[str, Any]]) -> None:
        """Append logs to the backend."""
        # Initial wait before apply.
        current_wait = random.uniform(*_INITIAL_BACKOFF_RANGE)
        for _ in range(_MAX_RETRY):
            try:
                response = self._s3.get_object(Bucket=self._bucket, Key=self._key)
                existing_logs = json.loads(response["Body"].read())
                logger.debug(f"Current log length: {len(existing_logs)}")
                condition_kwargs = {"IfMatch": response["ETag"]}
            except self._s3.exceptions.NoSuchKey:
                logger.debug(
                    f"The object 's3://{self._bucket}/{self._key}' does not exist. "
                    "Trying to create a new object."
                )
                existing_logs = []
                condition_kwargs = {"IfNoneMatch": "*"}
            try:
                response = self._s3.put_object(
                    Bucket=self._bucket,
                    Key=self._key,
                    Body=json.dumps(existing_logs + logs, indent=2).encode(),
                    **condition_kwargs,
                )
                logger.debug(
                    f"Previous log length {len(existing_logs)} -> "
                    f"Current log length {len(existing_logs + logs)}"
                )
                return
            except self._s3.exceptions.ClientError as e:
                if e.response["Error"]["Code"] in (
                    "PreconditionFailed",
                    "ConditionalRequestConflict",
                ):
                    if existing_logs:
                        logger.info(
                            "Failed to append logs to S3 because the object was updated. "
                            f"Waiting for {current_wait} before retrying..."
                        )
                    else:
                        logger.info(
                            "Failed to create logs to S3 because the object was created. "
                            f"Waiting for {current_wait} before retrying..."
                        )
                    # retry from reading.
                    time.sleep(current_wait)
                    current_wait *= random.uniform(*_EXPONENTIAL_BACKOFF_RANGE)
                    continue
                raise e
        logger.error("Failed to append logs to S3.")
        raise RuntimeError("Failed to append logs to S3.")
