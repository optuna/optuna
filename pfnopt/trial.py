import enum
from typing import Any
from typing import Dict
from typing import NamedTuple
from typing import Optional

import datetime


class State(enum.Enum):

    RUNNING = 0
    COMPLETE = 1
    PRUNED = 2
    FAIL = 3


class Trial(object):

    def __init__(self, trial_id):
        # type: (int) -> None

        self.trial_id = trial_id  # type: int
        self.state = State.RUNNING  # type: State
        self.params = {}  # type: Dict[str, Any]
        self.system_attrs = \
            SystemAttributes(datetime_start=None, datetime_complete=None)  # type: SystemAttributes
        self.user_attrs = {}  # type: Dict[str, Any]
        self.value = None  # type: Optional[float]
        self.intermediate_values = {}  # type: Dict[int, float]

        # TODO(Akiba): remove this
        self.params_in_internal_repr = {}  # type: Dict[str, float]


SystemAttributes = NamedTuple(
    'SystemAttributes',
    [('datetime_start', Optional[datetime.datetime]),
     ('datetime_complete', Optional[datetime.datetime])])
