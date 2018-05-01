from typing import Any
from typing import Dict
from typing import NamedTuple

from pfnopt.study_task import StudyTask


StudySummary = NamedTuple(
    'StudySummary',
    [('study_id', int),
     ('study_uuid', str),
     ('user_attrs', Dict[str, Any]),
     ('n_trials', int),
     ('task', StudyTask)])
