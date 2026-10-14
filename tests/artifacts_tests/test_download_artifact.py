from __future__ import annotations

import pathlib
from typing import BinaryIO
from unittest.mock import patch

import pytest

import optuna
from optuna.artifacts import download_artifact
from optuna.artifacts import FileSystemArtifactStore
from optuna.artifacts import upload_artifact
from optuna.artifacts._protocol import ArtifactStore


@pytest.fixture(params=["FileSystem"])
def artifact_store(tmp_path: pathlib.PurePath, request: pytest.FixtureRequest) -> ArtifactStore:
    if request.param == "FileSystem":
        return FileSystemArtifactStore(str(tmp_path))
    assert False, f"Unknown artifact store: {request.param}"


def test_download_artifact(tmp_path: pathlib.PurePath, artifact_store: ArtifactStore) -> None:
    study = optuna.create_study()
    artifact_ids: list[str] = []

    def objective(trial: optuna.Trial) -> float:
        x = trial.suggest_int("x", 0, 100)
        y = trial.suggest_int("y", 0, 100)
        dummy_file = str(tmp_path / f"dummy_{trial.number}.txt")
        with open(dummy_file, "w") as f:
            f.write(f"{x} {y}")
        artifact_ids.append(
            upload_artifact(
                study_or_trial=trial, file_path=dummy_file, artifact_store=artifact_store
            )
        )
        return x**2 + y**2

    study.optimize(objective, n_trials=5)

    for i, artifact_id in enumerate(artifact_ids):
        dummy_downloaded_file = str(tmp_path / f"dummy_downloaded_{i}.txt")
        download_artifact(
            file_path=dummy_downloaded_file, artifact_store=artifact_store, artifact_id=artifact_id
        )
        with open(dummy_downloaded_file, "r") as f:
            assert f.read() == f"{study.trials[i].params['x']} {study.trials[i].params['y']}"


def test_download_artifact_preserves_destination_created_during_open_reader(
    tmp_path: pathlib.Path, artifact_store: ArtifactStore
) -> None:
    source = tmp_path / "source.txt"
    source.write_bytes(b"artifact contents")
    study = optuna.create_study()
    artifact_id = upload_artifact(
        study_or_trial=study, file_path=str(source), artifact_store=artifact_store
    )
    destination = tmp_path / "destination.txt"
    original_open_reader = artifact_store.open_reader

    def open_reader(artifact_id: str) -> BinaryIO:
        destination.write_bytes(b"other task's contents")
        return original_open_reader(artifact_id)

    with patch.object(artifact_store, "open_reader", side_effect=open_reader):
        with pytest.raises(FileExistsError):
            download_artifact(
                artifact_store=artifact_store, artifact_id=artifact_id, file_path=str(destination)
            )

    assert destination.read_bytes() == b"other task's contents"
