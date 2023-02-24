from typing import Dict
from typing import List
from typing import Tuple
from typing import Union


JSONSerializable = Union[
    Dict[Union[str, int, float, bool, None], "JSONSerializable"],
    List["JSONSerializable"],
    Tuple["JSONSerializable", ...],
    str,
    int,
    float,
    bool,
    None,
]



__all__ = ["JSONSerializable"]
