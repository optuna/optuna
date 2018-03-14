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


SystemAttributes = NamedTuple(
    'SystemAttributes',
    [('datetime_start', Optional[datetime]),
     ('datetime_complete', Optional[datetime])])


Trial = NamedTuple(
    'Trial',
    [('trial_id', int),
     ('state', State),
     ('params', Dict[str, Any]),
     ('system_attrs', SystemAttributes),
     ('user_attrs', Dict[str, Any]),
     ('value', Optional[float]),
     ('intermediate_values', Dict[int, float]),
     ('params_in_internal_repr', Dict[str, float])])


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
