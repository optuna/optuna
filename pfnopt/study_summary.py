from datetime import datetime
from typing import Any
from typing import Dict
from typing import NamedTuple
from typing import Optional

from pfnopt.frozen_trial import FrozenTrial
from pfnopt.study_task import StudyTask


StudySummary = NamedTuple(
    'StudySummary',
    [('study_id', int),
     ('study_uuid', str),
     ('task', StudyTask),
     ('best_trial', Optional[FrozenTrial]),
     ('user_attrs', Dict[str, Any]),
     ('n_trials', int),
     ('datetime_start', Optional[datetime])])
