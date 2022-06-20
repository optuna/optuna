import datetime
from typing import Any
from typing import Dict
from typing import Optional
from typing import Sequence

from optuna import logging
from optuna.study._study_direction import StudyDirection


_logger = logging.get_logger(__name__)


class FrozenStudy:
    def __init__(
        self,
        study_name: str,
        direction: Optional[StudyDirection],
        user_attrs: Dict[str, Any],
        system_attrs: Dict[str, Any],
        study_id: int,
        directions: Optional[Sequence[StudyDirection]] = None,
    ):
        self.study_name = study_name
        if direction is None and directions is None:
            raise ValueError("Specify one of `direction` and `directions`.")
        elif directions is not None:
            self._directions = list(directions)
        elif direction is not None:
            self._directions = [direction]
        else:
            raise ValueError("Specify only one of `direction` and `directions`.")
        self.user_attrs = user_attrs
        self.system_attrs = system_attrs
        self._study_id = study_id

    def __eq__(self, other: Any) -> bool:

        if not isinstance(other, FrozenStudy):
            return NotImplemented

        return other.__dict__ == self.__dict__

    def __lt__(self, other: Any) -> bool:

        if not isinstance(other, FrozenStudy):
            return NotImplemented

        return self._study_id < other._study_id

    def __le__(self, other: Any) -> bool:

        if not isinstance(other, FrozenStudy):
            return NotImplemented

        return self._study_id <= other._study_id

    @property
    def direction(self) -> StudyDirection:

        if len(self._directions) > 1:
            raise RuntimeError(
                "This attribute is not available during multi-objective optimization."
            )

        return self._directions[0]

    @property
    def directions(self) -> Sequence[StudyDirection]:

        return self._directions
