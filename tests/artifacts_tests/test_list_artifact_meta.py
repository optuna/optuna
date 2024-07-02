from __future__ import annotations

import pathlib

import pytest

import optuna
from optuna.artifacts import FileSystemArtifactStore
from optuna.artifacts import get_all_artifact_meta
from optuna.artifacts import upload_artifact
from optuna.artifacts._protocol import ArtifactStore
from optuna.storages import BaseStorage
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import Trial


parametrize_artifact_meta = pytest.mark.parametrize(
    "filename,mimetype,expected_mimetype,encoding",
    [("dummy.txt", None, "text/plain", None), ("dummy.obj", "model/obj", "model/obj", "utf-8")],
)


@pytest.fixture(params=["FileSystem"])
def artifact_store(tmp_path: pathlib.PurePath, request: pytest.FixtureRequest) -> ArtifactStore:
    if request.param == "FileSystem":
        return FileSystemArtifactStore(str(tmp_path))
    assert False, f"Unknown artifact store: {request.param}"


def _check_uploaded_artifact_meta(
    study_or_trial: Study | Trial | FrozenTrial,
    storage: BaseStorage,
    artifact_store: ArtifactStore,
    filename: str,
    file_path: str,
    mimetype: str | None,
    expected_mimetype: str,
    encoding: str | None,
) -> None:
    artifact_id = upload_artifact(
        study_or_trial=study_or_trial,
        file_path=file_path,
        artifact_store=artifact_store,
        storage=storage,
        mimetype=mimetype,
        encoding=encoding,
    )
    artifact_meta_list = get_all_artifact_meta(study_or_trial, storage=storage)
    assert len(artifact_meta_list) == 1
    assert artifact_meta_list[0].artifact_id == artifact_id
    assert artifact_meta_list[0].filename == filename
    assert artifact_meta_list[0].mimetype == expected_mimetype
    assert artifact_meta_list[0].encoding == encoding


@parametrize_artifact_meta
def test_get_all_artifact_meta_in_trial(
    tmp_path: pathlib.PurePath,
    artifact_store: ArtifactStore,
    filename: str,
    mimetype: str | None,
    expected_mimetype: str,
    encoding: str | None,
) -> None:
    file_path = str(tmp_path / filename)

    with open(file_path, "w") as f:
        f.write("foo")

    storage = optuna.storages.InMemoryStorage()
    study = optuna.create_study(storage=storage)
    trial = study.ask()
    _check_uploaded_artifact_meta(
        study_or_trial=trial,
        storage=storage,
        artifact_store=artifact_store,
        filename=filename,
        file_path=file_path,
        mimetype=mimetype,
        expected_mimetype=expected_mimetype,
        encoding=encoding,
    )


@parametrize_artifact_meta
def test_get_all_artifact_meta_in_frozen_trial(
    tmp_path: pathlib.PurePath,
    artifact_store: ArtifactStore,
    filename: str,
    mimetype: str | None,
    expected_mimetype: str,
    encoding: str | None,
) -> None:
    file_path = str(tmp_path / filename)

    with open(file_path, "w") as f:
        f.write("foo")

    storage = optuna.storages.InMemoryStorage()
    study = optuna.create_study(storage=storage)
    trial = study.ask()
    frozen_trial = study._storage.get_trial(trial._trial_id)
    _check_uploaded_artifact_meta(
        study_or_trial=frozen_trial,
        storage=storage,
        artifact_store=artifact_store,
        filename=filename,
        file_path=file_path,
        mimetype=mimetype,
        expected_mimetype=expected_mimetype,
        encoding=encoding,
    )


@parametrize_artifact_meta
def test_get_all_artifact_meta_in_study(
    tmp_path: pathlib.PurePath,
    artifact_store: ArtifactStore,
    filename: str,
    mimetype: str | None,
    expected_mimetype: str,
    encoding: str | None,
) -> None:
    file_path = str(tmp_path / filename)

    with open(file_path, "w") as f:
        f.write("foo")

    storage = optuna.storages.InMemoryStorage()
    study = optuna.create_study(storage=storage)
    _check_uploaded_artifact_meta(
        study_or_trial=study,
        storage=storage,
        artifact_store=artifact_store,
        filename=filename,
        file_path=file_path,
        mimetype=mimetype,
        expected_mimetype=expected_mimetype,
        encoding=encoding,
    )
