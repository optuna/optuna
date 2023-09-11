from __future__ import annotations

import json
import pathlib

import pytest

import optuna
from optuna.artifacts import FileSystemArtifactStore
from optuna.artifacts import upload_artifact
from optuna.artifacts._protocol import ArtifactStore
from optuna.artifacts._upload import ArtifactMeta


@pytest.fixture(params=["FileSystem"])
def artifact_store(tmp_path: pathlib.PurePath, request: pytest.FixtureRequest) -> ArtifactStore:
    if request.param == "FileSystem":
        return FileSystemArtifactStore(str(tmp_path))
    assert False, f"Unknown artifact store: {request.param}"


def test_upload_trial_artifact(tmp_path: pathlib.PurePath, artifact_store: ArtifactStore) -> None:
    file_path = str(tmp_path / "dummy.txt")

    with open(file_path, "w") as f:
        f.write("foo")

    storage = optuna.storages.InMemoryStorage()
    study = optuna.create_study(storage=storage)

    trial = study.ask()

    upload_artifact(trial, file_path, artifact_store)

    frozen_trial = study._storage.get_trial(trial._trial_id)

    with pytest.raises(ValueError):
        upload_artifact(frozen_trial, file_path, artifact_store)

    upload_artifact(frozen_trial, file_path, artifact_store, storage=trial.study._storage)

    system_attrs = storage.get_trial_system_attrs(frozen_trial._trial_id)
    artifact_items = [
        ArtifactMeta(**json.loads(val))
        for key, val in system_attrs.items()
        if key.startswith("artifacts:")
    ]

    assert len(artifact_items) == 2
    assert artifact_items[0].artifact_id != artifact_items[1].artifact_id
    assert artifact_items[0].filename == "dummy.txt"
    assert artifact_items[0].mimetype == "text/plain"
    assert artifact_items[0].encoding is None


def test_upload_study_artifact(tmp_path: pathlib.PurePath, artifact_store: ArtifactStore) -> None:
    file_path = str(tmp_path / "dummy.txt")

    with open(file_path, "w") as f:
        f.write("foo")

    storage = optuna.storages.InMemoryStorage()
    study = optuna.create_study(storage=storage)
    artifact_id = upload_artifact(study, file_path, artifact_store)

    system_attrs = storage.get_study_system_attrs(study._study_id)
    artifact_items = [
        ArtifactMeta(**json.loads(val))
        for key, val in system_attrs.items()
        if key.startswith("artifacts:")
    ]

    assert len(artifact_items) == 1
    assert artifact_items[0].artifact_id == artifact_id
    assert artifact_items[0].filename == "dummy.txt"
    assert artifact_items[0].mimetype == "text/plain"
    assert artifact_items[0].encoding is None


def test_upload_artifact_with_mimetype(
    tmp_path: pathlib.PurePath, artifact_store: ArtifactStore
) -> None:
    file_path = str(tmp_path / "dummy.obj")

    with open(file_path, "w") as f:
        f.write("foo")

    study = optuna.create_study()

    trial = study.ask()

    upload_artifact(trial, file_path, artifact_store, mimetype="model/obj", encoding="utf-8")

    frozen_trial = study._storage.get_trial(trial._trial_id)

    with pytest.raises(ValueError):
        upload_artifact(frozen_trial, file_path, artifact_store)

    upload_artifact(frozen_trial, file_path, artifact_store, storage=trial.study._storage)

    system_attrs = trial.study._storage.get_trial(frozen_trial._trial_id).system_attrs
    artifact_items = [
        ArtifactMeta(**json.loads(val))
        for key, val in system_attrs.items()
        if key.startswith("artifacts:")
    ]

    assert len(artifact_items) == 2
    assert artifact_items[0].artifact_id != artifact_items[1].artifact_id
    assert artifact_items[0].filename == "dummy.obj"
    assert artifact_items[0].mimetype == "model/obj"
    assert artifact_items[0].encoding == "utf-8"
