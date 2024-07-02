from __future__ import annotations

import pathlib

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
