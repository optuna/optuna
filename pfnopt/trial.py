import enum
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
        self.trial_id = trial_id
        self.state = State.RUNNING
        self.params = {}
        self.params_in_internal_repr = {}  # TODO(Akiba): eliminate me
        self.system_attrs = \
            SystemAttributes(datetime_start=None, datetime_complete=None)  # type: SystemAttributes
        self.user_attrs = {}
        self.value = None
        self.intermediate_values = {}


class SystemAttributes(
    NamedTuple(
        '_BaseSystemAttributes',
        [('datetime_start', Optional[datetime.datetime]),
         ('datetime_complete', Optional[datetime.datetime])])):
    pass
