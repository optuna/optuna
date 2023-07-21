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
    trial: Trial | FrozenTrial | int,
    file_path: str,
    artifact_store: ArtifactStore,  # prefer with warning
    *,
    storage: BaseStorage | None = None,
) -> str:
    filename = os.path.basename(file_path)

    if isinstance(trial, Trial) and storage is None:
        storage = trial.study._storage

    trial_id: int

    if storage is not None:
        if isinstance(trial, Trial):
            trial_id = trial._trial_id
            storage = trial.storage
        elif isinstance(trial, FrozenTrial):
            trial_id = trial._trial_id
        elif isinstance(trial, int):
            trial_id = trial
        else:
            raise ValueError("trial must be Trial, FrozenTrial or int")
    else:
        raise ValueError("storage is required for FrozenTrial")

    artifact_id = str(uuid.uuid4())

    guess_mimetype, guess_encoding = mimetypes.guess_type(filename)

    artifact = ArtifactMeta(
        artifact_id=artifact_id,
        filename=filename,
        mimetype=guess_mimetype or DEFAULT_MIME_TYPE,
        encoding=guess_encoding,
    )
    attr_key = ARTIFACTS_ATTR_PREFIX + f"{trial_id}:" + artifact_id
    storage.set_trial_system_attr(trial_id, attr_key, json.dumps(asdict(artifact)))

    with open(file_path, "rb") as f:
        artifact_store.write(artifact_id, f)
    return artifact_id
