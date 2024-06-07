from __future__ import annotations

import json

from optuna._experimental import experimental_func
from optuna.artifacts._upload import ArtifactMeta
from optuna.artifacts._upload import ARTIFACTS_ATTR_PREFIX
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import Trial


@experimental_func("4.0.0")
def get_all_artifact_meta(study_or_trial: Trial | FrozenTrial | Study) -> list[ArtifactMeta]:
    """List the associated artifact information of the provided trial or study.

    Args:
        study_or_trial:
            A :class:`~optuna.trial.Trial` object, a :class:`~optuna.trial.FrozenTrial`, or
            a :class:`~optuna.study.Study` object.

    Returns:
        The list of artifact meta in the trial or study.
        Each artifact meta includes ``artifact_id``, ``filename``, ``mimetype``, and ``encoding``.
        Note that If :class:`~optuna.study.Study` is provided, we return the information of the
        artifacts uploaded to ``study``, but not to all the trials in the study.
    """
    system_attrs = study_or_trial.system_attrs
    artifact_meta_list: list[ArtifactMeta] = []
    for attr_key, attr_json_string in system_attrs.items():
        if not attr_key.startswith(ARTIFACTS_ATTR_PREFIX):
            continue

        attr_content = json.loads(attr_json_string)
        artifact_meta = ArtifactMeta(
            artifact_id=attr_content["artifact_id"],
            filename=attr_content["filename"],
            mimetype=attr_content["mimetype"],
            encoding=attr_content["encoding"],
        )
        artifact_meta_list.append(artifact_meta)

    return artifact_meta_list
