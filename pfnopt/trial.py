from datetime import datetime
import enum
from typing import Any  # NOQA
from typing import Dict  # NOQA
from typing import NamedTuple
from typing import Optional


class State(enum.Enum):

    RUNNING = 0
    COMPLETE = 1
    PRUNED = 2
    FAIL = 3

    def is_finished(self):
        # type: () -> bool

        return self == State.COMPLETE or self == State.PRUNED


Trial = NamedTuple(
    'Trial',
    [('trial_id', int),
     ('state', State),
     ('params', Dict[str, Any]),
     ('user_attrs', Dict[str, Any]),
     ('value', Optional[float]),
     ('intermediate_values', Dict[int, float]),
     ('params_in_internal_repr', Dict[str, float]),
     ('datetime_start', Optional[datetime]),
     ('datetime_complete', Optional[datetime])])
