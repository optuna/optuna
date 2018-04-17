import enum
from typing import Any
from typing import Dict
from typing import NamedTuple
from typing import Optional


class StudyTask(enum.Enum):

    MINIMIZE = 0
    MAXIMIZE = 1


StudySystemAttributes = NamedTuple(
    'StudySystemAttributes',
    [('name', Optional[str]),
     ('task', Optional[StudyTask])])


StudySummary = NamedTuple(
    'StudySummary',
    [('study_id', int),
     ('study_uuid', str),
     ('system_attrs', StudySystemAttributes),
     ('user_attrs', Dict[str, Any]),
     ('n_trials', int)])
