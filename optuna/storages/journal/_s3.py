import json
import logging
from typing import Any

from optuna._experimental import experimental_class
from optuna._imports import try_import
from optuna.storages.journal._base import BaseJournalBackend
from optuna.storages.journal._base import BaseJournalSnapshot


with try_import() as _imports:
    import boto3

logger = logging.Logger(__name__)

@experimental_class("3.1.0")
class JournalS3Backend(BaseJournalBackend, BaseJournalSnapshot):
    """S3 storage class for Journal log backend.

    Args:
        bucket:
            Name of the S3 bucket.
        prefix:
            Prefix for the log and snapshot objects in the bucket.
        num_logs_per_object:
            Number of logs per S3 object. The default value is 10000.
            If continuing an existing log, the value must be consistent with the existing value.
    """
    
    def __init__(self, bucket: str, prefix: str, num_logs_per_object: int = 10000) -> None:
        _imports.check()

        self._bucket = bucket
        if prefix and prefix.endswith("/"):
            self.prefix = prefix[:-1]
        else:
            self._prefix = prefix
        self._num_logs_per_object = num_logs_per_object

        self._s3 = boto3.client("s3")

        # conditionally create a config object and assert it is consistent.
        num_logs_per_object_key = f"{self._prefix}/num_logs_per_object"
        try:
            self._s3.put_object(
                Bucket=self._bucket,
                Key=num_logs_per_object_key, Body=str(num_logs_per_object).encode(),
                            IfNoneMatch="*")
        except self._s3.exceptions.ClientError as e:
            if e.response['Error']['Code'] != 'PreconditionFailed':
                logging.error(f"Error creating object '{num_logs_per_object_key}': {e}")
                raise
            logger.info(f"Object '{self._prefix}/num_logs_per_object' already exists. Checking consistency.")
            response = self._s3.get_object(Bucket=self._bucket, Key=num_logs_per_object_key)
            if int(response["Body"].read().decode()) != num_logs_per_object:
                raise ValueError(f"num_logs_per_object in S3 object '{num_logs_per_object_key}' is inconsistent with the provided value.")
    
    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        del state["_s3"]
        return state
    
    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._s3 = boto3.client("s3")

    def _log_look_up(self, log_number: int) -> tuple[str, int]:
        """Returns the object key and the index of the log in the object."""
        return f"{self._prefix}/logs/{log_number // self._num_logs_per_object}.json", log_number % self._num_logs_per_object

    def _list_log_object_keys(self) -> list[str]:
        """Returns a list of log objects from S3, with pagination, ordered from old to new."""
        log_object_keys = []
        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._bucket, Prefix=f"{self._prefix}/logs/"):
            if "Contents" in page:
                log_object_keys.extend([content["Key"] for content in page["Contents"]])

        # Not using `sorted`` to follow numerical order, not lexicographical order.
        ordered = [f"{self._prefix}/logs/{i}.json" for i in range(len(log_object_keys))]
        if set(log_object_keys) != set(ordered):
            raise ValueError(f"Log object keys ({log_object_keys}) are inconsistent with the expected keys "
                             f"{ordered}.")
        return ordered

    def read_logs(self, log_number_from: int) -> list[dict[str, Any]]:
        """Read logs with a log number greater than or equal to ``log_number_from``."""
        log_cache = {}
        # Get the current list first, so that others can write logs while reading.
        # The last file may be appended, but it is fine and the read still ends there.
        log_objects = self._list_log_object_keys()
        current_log_number = log_number_from
        logs = []
        while True:
            log_object_key, log_index = self._log_look_up(current_log_number)
            if log_object_key not in log_cache:
                if log_object_key not in log_objects:
                    break
                response = self._s3.get_object(Bucket=self._bucket, Key=log_object_key)
                log_cache[log_object_key] = json.loads(response["Body"].read())
            if log_index >= len(log_cache[log_object_key]):
                break
            logs.append(log_cache[log_object_key][log_index])
            current_log_number += 1
        return logs
    
    def _write_logs(self, key: str, logs: list[dict[str, Any]], if_none_match: str) -> bool:
        """Write logs to the S3 object with the given key. Returns False if precondition fails."""
        try:
            self._s3.put_object(Bucket=self._bucket, Key=key, Body=json.dumps(logs).encode(),
                            IfNoneMatch=if_none_match)
        except self._s3.exceptions.ClientError as e:
            if e.response['Error']['Code'] != 'PreconditionFailed':
                logger.warning(f"Other process has updated the log {key}. Retrying.")
                return False
            raise

    def append_logs(self, logs):
        while logs:
            log_objects = self._list_log_object_keys()
            if log_objects:
                last_log_key = log_objects[-1]
                response = self._s3.get_object(Bucket=self._bucket, Key=last_log_key)
                last_logs = json.loads(response["Body"].read())
                if len(last_logs) < self._num_logs_per_object:
                    # try appending to the last log
                    num_logs_to_append = min(len(logs), self._num_logs_per_object - len(last_logs))
                    last_logs.extend(logs[:num_logs_to_append])
                    if not self._write_logs(last_log_key, last_logs, response["ETag"]):
                        # retry from reading.
                        continue
                    logs = logs[num_logs_to_append:]
                    if not logs:
                        break
            # at this point, all log objects should be full.
            existing_logs = len(log_objects) * self._num_logs_per_object
            new_log_key, _ = self._log_look_up(existing_logs + 1)
            # create a new log object
            num_logs_to_append = min(len(logs), self._num_logs_per_object)
            logs_to_write = logs[:num_logs_to_append]
            if not self._write_logs(new_log_key, logs_to_write, "*"):
                # retry from reading.
                continue
            logs = logs[num_logs_to_append:]
            
    def save_snapshot(self, snapshot: bytes) -> None:
        """Save snapshot to the backend."""
        self._s3.put_object(Bucket=self._bucket, Key=f"{self._prefix}/snapshot", Body=snapshot)
    
    def load_snapshot(self) -> bytes | None:
        """Load snapshot from the backend."""
        try:
            response = self._s3.get_object(Bucket=self._bucket, Key=f"{self._prefix}/snapshot")
            return response["Body"].read()
        except self._s3.exceptions.NoSuchKey:
            return None
        except self._s3.exceptions.ClientError as e:
            logger.error(f"Error loading snapshot from S3: {e}")
            raise
