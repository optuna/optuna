from datetime import datetime
import enum
import json
from typing import Any  # NOQA
from typing import Dict  # NOQA
from typing import NamedTuple
from typing import Optional


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
    [('datetime_start', Optional[datetime]),
     ('datetime_complete', Optional[datetime])])


def system_attrs_to_json(system_attrs):
    # type: (SystemAttributes) -> str

    def convert(attr):
        if isinstance(attr, datetime):
            return attr.strftime('%Y%m%d%H%M%S')
        else:
            return attr

    return json.dumps(system_attrs._asdict(), default=convert)


def json_to_system_attrs(system_attrs_json):
    # type: (str) -> SystemAttributes
    system_attrs_dict = json.loads(system_attrs_json)

    for k, v in system_attrs_dict.items():
        if k in {'datetime_start', 'datetime_complete'}:
            system_attrs_dict[k] = None if v is None else datetime.strptime(v, '%Y%m%d%H%M%S')
        else:
            system_attrs_dict[k] = v

    return SystemAttributes(**system_attrs_dict)
