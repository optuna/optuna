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


@pytest.fixture(params=["FileSystem"])
def artifact_store(tmp_path: pathlib.PurePath, request: pytest.FixtureRequest) -> ArtifactStore:
    if request.param == "FileSystem":
        return FileSystemArtifactStore(str(tmp_path))
    assert False, f"Unknown artifact store: {request.param}"


def _check_artifact_meta(
    artifact_store: ArtifactStore,
    study_or_trial: Trial | FrozenTrial | Study,
    file_path: str,
    filename: str,
    mimetype: str | None,
    encoding: str | None,
    storage: BaseStorage | None,
    n_linked_artifact_meta: int,
) -> None:
    artifact_id = upload_artifact(
        study_or_trial,
        file_path,
        artifact_store,
        storage=storage,
        mimetype=mimetype,
        encoding=encoding,
    )
    mimetype = "text/plain" if mimetype is None else mimetype
    if isinstance(study_or_trial, FrozenTrial):
        artifact_meta_list = get_all_artifact_meta(study_or_trial, storage=storage)
    else:
        artifact_meta_list = get_all_artifact_meta(study_or_trial)

    assert len(artifact_meta_list) == n_linked_artifact_meta
    assert artifact_meta_list[-1].artifact_id == artifact_id
    assert artifact_meta_list[-1].filename == filename
    assert artifact_meta_list[-1].mimetype == mimetype
    assert artifact_meta_list[-1].encoding == encoding


@pytest.mark.parametrize(
    "filename,mimetype,encoding", [("dummy.txt", None, None), ("dummy.obj", "model/obj", "utf-8")]
)
def test_get_all_artifact_meta(
    tmp_path: pathlib.PurePath,
    artifact_store: ArtifactStore,
    filename: str,
    mimetype: str | None,
    encoding: str | None,
) -> None:
    file_path = str(tmp_path / filename)

    with open(file_path, "w") as f:
        f.write("foo")

    storage = optuna.storages.InMemoryStorage()
    study = optuna.create_study(storage=storage)
    trial = study.ask()
    frozen_trial = study._storage.get_trial(trial._trial_id)
    for study_or_trial, n_linked_artifact_meta in zip([study, trial, frozen_trial], [1, 1, 2]):
        assert isinstance(study_or_trial, (Study, Trial, FrozenTrial))  # MyPy redefinition.
        _check_artifact_meta(
            artifact_store,
            study_or_trial,
            file_path,
            filename,
            mimetype=mimetype,
            encoding=encoding,
            storage=storage,
            n_linked_artifact_meta=n_linked_artifact_meta,
        )
