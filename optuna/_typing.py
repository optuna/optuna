from typing import Dict
from typing import List
from typing import Tuple
from typing import Union


JSONPrimitiveType = Union[str, int, float, bool, None]

JSONSerializable = Union[
    Dict[JSONPrimitiveType, "JSONSerializable"],
    List["JSONSerializable"],
    Tuple["JSONSerializable", ...],
]

__all__ = ["JSONSerializable"]
