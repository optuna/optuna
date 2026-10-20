from __future__ import annotations

import pathlib

import pytest

import optuna
from optuna.artifacts import FileSystemArtifactStore
from optuna.artifacts import get_all_artifact_meta
from optuna.artifacts import upload_artifact
from optuna.artifacts._protocol import ArtifactStore
from optuna.testing.storages import StorageSupplier


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


@pytest.mark.parametrize("storage_mode", ["inmemory", "sqlite", "journal"])
@pytest.mark.parametrize("target", ["study", "trial", "frozen_trial"])
def test_upload_artifact_missing_source(
    tmp_path: pathlib.PurePath,
    artifact_store: ArtifactStore,
    storage_mode: str,
    target: str,
) -> None:
    file_path = str(tmp_path / "dummy.txt")
    with open(file_path, "w") as f:
        f.write("foo")

    with StorageSupplier(storage_mode) as storage:
        study = optuna.create_study(storage=storage)
        trial = study.ask()
        targets: dict[str, optuna.Study | optuna.Trial | optuna.trial.FrozenTrial] = {
            "study": study,
            "trial": trial,
            "frozen_trial": study.get_trials()[0],
        }
        study_or_trial = targets[target]
        upload_artifact(
            study_or_trial=study_or_trial,
            file_path=file_path,
            artifact_store=artifact_store,
            storage=storage,
        )
        original_metadata = get_all_artifact_meta(study_or_trial, storage=storage)
        assert len(original_metadata) == 1

        with pytest.raises(FileNotFoundError):
            upload_artifact(
                study_or_trial=study_or_trial,
                file_path=str(tmp_path / "missing.txt"),
                artifact_store=artifact_store,
                storage=storage,
            )
        assert get_all_artifact_meta(study_or_trial, storage=storage) == original_metadata

        artifact_id = upload_artifact(
            study_or_trial=study_or_trial,
            file_path=file_path,
            artifact_store=artifact_store,
            storage=storage,
        )
        artifact_items = get_all_artifact_meta(study_or_trial, storage=storage)
        assert len(artifact_items) == 2
        assert original_metadata[0] in artifact_items
        assert artifact_id in [artifact.artifact_id for artifact in artifact_items]
        with artifact_store.open_reader(artifact_id) as f:
            assert f.read() == b"foo"


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
            trial,  # type: ignore
            file_path,
            artifact_store=artifact_store,
        )
    _validate(artifact_id=artifact_id)
    with pytest.warns(FutureWarning):
        artifact_id = upload_artifact(
            trial,  # type: ignore
            file_path=file_path,
            artifact_store=artifact_store,
        )
    _validate(artifact_id=artifact_id)
