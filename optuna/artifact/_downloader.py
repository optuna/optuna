import shutil


def download_artifacts(artifact: dict[str, str], file_path: str):
    # study.artifact.open()
    shutil.move(artifact["filepath"], file_path)
