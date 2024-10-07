from __future__ import annotations

import pathlib

import pytest

import optuna
from optuna.artifacts import FileSystemArtifactStore
from optuna.artifacts import get_all_artifact_meta
from optuna.artifacts import upload_artifact
from optuna.artifacts._protocol import ArtifactStore


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
    upload_artifact(study_or_trial=trial, file_path=file_path, artifact_store=artifact_store)
    frozen_trial = study._storage.get_trial(trial._trial_id)
    with pytest.raises(ValueError):
        upload_artifact(
            study_or_trial=frozen_trial, file_path=file_path, artifact_store=artifact_store
        )

    upload_artifact(
        study_or_trial=frozen_trial,
        file_path=file_path,
        artifact_store=artifact_store,
        storage=trial.study._storage,
    )
    artifact_items = get_all_artifact_meta(frozen_trial, storage=storage)
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
    artifact_id = upload_artifact(
        study_or_trial=study, file_path=file_path, artifact_store=artifact_store
    )
    artifact_items = get_all_artifact_meta(study)
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
    upload_artifact(
        study_or_trial=trial,
        file_path=file_path,
        artifact_store=artifact_store,
        mimetype="model/obj",
        encoding="utf-8",
    )
    frozen_trial = study._storage.get_trial(trial._trial_id)
    with pytest.raises(ValueError):
        upload_artifact(
            study_or_trial=frozen_trial, file_path=file_path, artifact_store=artifact_store
        )
    upload_artifact(
        study_or_trial=frozen_trial,
        file_path=file_path,
        artifact_store=artifact_store,
        storage=trial.study._storage,
    )
    artifact_items = get_all_artifact_meta(frozen_trial, storage=study._storage)
    assert len(artifact_items) == 2
    assert artifact_items[0].artifact_id != artifact_items[1].artifact_id
    assert artifact_items[0].filename == "dummy.obj"
    assert artifact_items[0].mimetype == "model/obj"
    assert artifact_items[0].encoding == "utf-8"


def test_upload_artifact_with_positional_args(
    tmp_path: pathlib.PurePath, artifact_store: ArtifactStore
) -> None:

    storage = optuna.storages.InMemoryStorage()
    study = optuna.create_study(storage=storage)
    trial = study.ask()

    def _validate(artifact_id: str) -> None:
        artifact_items = get_all_artifact_meta(trial, storage=storage)
        assert artifact_items[-1].artifact_id == artifact_id
        assert artifact_items[-1].filename == "dummy.txt"
        assert artifact_items[-1].mimetype == "text/plain"
        assert artifact_items[-1].encoding is None

    file_path = str(tmp_path / "dummy.txt")
    with open(file_path, "w") as f:
        f.write("foo")

    with pytest.warns(FutureWarning):
        artifact_id = upload_artifact(trial, file_path, artifact_store)  # type: ignore
    _validate(artifact_id=artifact_id)
    with pytest.warns(FutureWarning):
        artifact_id = upload_artifact(
            trial, file_path, artifact_store=artifact_store  # type: ignore
        )
    _validate(artifact_id=artifact_id)
    with pytest.warns(FutureWarning):
        artifact_id = upload_artifact(
            trial, file_path=file_path, artifact_store=artifact_store  # type: ignore
        )
    _validate(artifact_id=artifact_id)
