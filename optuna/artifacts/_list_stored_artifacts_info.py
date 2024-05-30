from __future__ import annotations

import json

from optuna._experimental import experimental_func
from optuna.artifacts._upload import ARTIFACTS_ATTR_PREFIX
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import Trial


@experimental_func("4.0.0")
def list_stored_artifact_info(study_or_trial: Trial | FrozenTrial | Study) -> list[dict[str, str]]:
    """List the associated artifact information of the provided trial or study.

    Args:
        study_or_trial:
            A :class:`~optuna.trial.Trial` object, a :class:`~optuna.trial.FrozenTrial`, or
            a :class:`~optuna.study.Study` object.

    Returns:
        The list of artifact information in the trial or study.
        Each artifact information is a dict including ``artifact_id`` and ``original_file_path``.
        Note that If :class:`~optuna.study.Study` is provided, we return the information of the
        artifacts uploaded to ``study``, but not to all the trials in the study.
    """
    system_attrs = study_or_trial.system_attrs
    artifact_info_list: list[dict[str, str]] = []
    for attr_key, attr_json_string in system_attrs.items():
        if not attr_key.startswith(ARTIFACTS_ATTR_PREFIX):
            continue

        attr_content = json.loads(attr_json_string)
        artifact_info = {
            "artifact_id": attr_content["artifact_id"],
            "original_file_path": attr_content["file_path"],
        }
        artifact_info_list.append(artifact_info)

    return artifact_info_list
