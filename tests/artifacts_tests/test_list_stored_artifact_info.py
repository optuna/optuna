from __future__ import annotations

import pathlib

import pytest

import optuna
from optuna.artifacts import FileSystemArtifactStore
from optuna.artifacts import list_stored_artifact_info
from optuna.artifacts import upload_artifact
from optuna.artifacts._protocol import ArtifactStore


@pytest.fixture(params=["FileSystem"])
def artifact_store(tmp_path: pathlib.PurePath, request: pytest.FixtureRequest) -> ArtifactStore:
    if request.param == "FileSystem":
        return FileSystemArtifactStore(str(tmp_path))
    assert False, f"Unknown artifact store: {request.param}"


def test_list_artifact_info_stored_in_trial(
    tmp_path: pathlib.PurePath, artifact_store: ArtifactStore
) -> None:
    file_path = str(tmp_path / "dummy.txt")

    with open(file_path, "w") as f:
        f.write("foo")

    storage = optuna.storages.InMemoryStorage()
    study = optuna.create_study(storage=storage)
    trial = study.ask()
    artifact_id = upload_artifact(trial, file_path, artifact_store)
    frozen_trial = study._storage.get_trial(trial._trial_id)
    upload_artifact(frozen_trial, file_path, artifact_store, storage=trial.study._storage)
    artifact_info = list_stored_artifact_info(frozen_trial)
    assert len(artifact_info) == 1
    assert artifact_info[0]["artifact_id"] == artifact_id
    assert artifact_info[0]["original_file_path"] == file_path


def test_list_artifact_info_stored_in_study(
    tmp_path: pathlib.PurePath, artifact_store: ArtifactStore
) -> None:
    file_path = str(tmp_path / "dummy.txt")

    with open(file_path, "w") as f:
        f.write("foo")

    storage = optuna.storages.InMemoryStorage()
    study = optuna.create_study(storage=storage)
    artifact_id = upload_artifact(study, file_path, artifact_store)
    artifact_info = list_stored_artifact_info(study)
    assert len(artifact_info) == 1
    assert artifact_info[0]["artifact_id"] == artifact_id
    assert artifact_info[0]["original_file_path"] == file_path
