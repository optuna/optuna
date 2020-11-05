import datetime
from typing import Any
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Union

from optuna.distributions import BaseDistribution
from optuna.trial._frozen.base import BaseFrozenTrial
from optuna.trial._state import TrialState


class MultiObjectiveFrozenTrial(BaseFrozenTrial):
    def __init__(
        self,
        number: int,
        state: TrialState,
        value: Optional[Sequence[float]],
        datetime_start: Optional[datetime.datetime],
        datetime_complete: Optional[datetime.datetime],
        params: Dict[str, Any],
        distributions: Dict[str, BaseDistribution],
        user_attrs: Dict[str, Any],
        system_attrs: Dict[str, Any],
        intermediate_values: Dict[int, Sequence[float]],
        trial_id: int,
    ) -> None:

        super(MultiObjectiveFrozenTrial, self).__init__(
            number,
            state,
            value,
            datetime_start,
            datetime_complete,
            params,
            distributions,
            user_attrs,
            system_attrs,
            intermediate_values,
            trial_id,
        )

    @property
    def value(self) -> Optional[Sequence[float]]:

        return tuple(self._value)

    @value.setter
    def value(self, v: Optional[float]) -> None:

        self._value = v

    @property
    def intermediate_values(self) -> Dict[int, Sequence[float]]:

        value: Dict[int, Union[float, Sequence[float]]] = {}
        for k, v in self._intermediate_values.items():
            value[k] = tuple(v)

        return value

    @intermediate_values.setter
    def intermediate_values(self, value: Dict[int, Sequence[float]]) -> None:

        self._intermediate_values = value
