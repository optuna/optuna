from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
import json
import mimetypes
import os
import uuid

from optuna.artifacts._protocol import ArtifactStore
from optuna.storages import BaseStorage
from optuna.trial._frozen import FrozenTrial
from optuna.trial._trial import Trial


ARTIFACTS_ATTR_PREFIX = "artifacts:"
DEFAULT_MIME_TYPE = "application/octet-stream"


@dataclass
class ArtifactMeta:
    artifact_id: str
    filename: str
    mimetype: str
    encoding: str | None


def upload_artifact(
    trial: Trial | FrozenTrial,
    file_path: str,
    artifact_store: ArtifactStore,
    *,
    storage: BaseStorage | None = None,
) -> str:
    """Upload an artifact to the artifact store.

    Args:
        trial:
            A :class:`~optuna.trial.Trial` object, a :class:`~optuna.trial.FrozenTrial` object, or
            an ID of the trial.
        file_path:
            A path to the file to be uploaded.
        artifact_store:
            An artifact store.
        storage:
            A storage object. If trial is not a :class:`~optuna.trial.Trial` object, this argument
            is required.

    Returns:
        An artifact ID.
    """

    filename = os.path.basename(file_path)

    if isinstance(trial, Trial) and storage is None:
        storage = trial.storage

    trial_id: int

    if storage is None:
        raise ValueError("storage is required for FrozenTrial or int.")
    if isinstance(trial, Trial) or isinstance(trial, FrozenTrial):
        trial_id = trial._trial_id
    elif isinstance(trial, int):
        trial_id = trial
    else:
        raise ValueError("trial must be Trial, FrozenTrial or int.")

    artifact_id = str(uuid.uuid4())

    guess_mimetype, guess_encoding = mimetypes.guess_type(filename)

    artifact = ArtifactMeta(
        artifact_id=artifact_id,
        filename=filename,
        mimetype=guess_mimetype or DEFAULT_MIME_TYPE,
        encoding=guess_encoding,
    )
    attr_key = ARTIFACTS_ATTR_PREFIX + artifact_id
    storage.set_trial_system_attr(trial_id, attr_key, json.dumps(asdict(artifact)))

    with open(file_path, "rb") as f:
        artifact_store.write(artifact_id, f)
    return artifact_id
