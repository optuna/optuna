import datetime
from typing import Any
from typing import Dict
from typing import Optional

from optuna.distributions import BaseDistribution
from optuna.trial._frozen.base import BaseFrozenTrial
from optuna.trial._state import TrialState


class FrozenTrial(BaseFrozenTrial):
    def __init__(
        self,
        number: int,
        state: TrialState,
        value: Optional[float],
        datetime_start: Optional[datetime.datetime],
        datetime_complete: Optional[datetime.datetime],
        params: Dict[str, Any],
        distributions: Dict[str, BaseDistribution],
        user_attrs: Dict[str, Any],
        system_attrs: Dict[str, Any],
        intermediate_values: Dict[int, float],
        trial_id: int,
    ) -> None:

        super(FrozenTrial, self).__init__(
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
    def value(self) -> Optional[float]:

        return self._value

    @value.setter
    def value(self, v: Optional[float]) -> None:

        self._value = v

    @property
    def intermediate_values(self) -> Dict[int, float]:

        return self._intermediate_values

    @intermediate_values.setter
    def intermediate_values(self, value: Dict[int, float]) -> None:

        self._intermediate_values = value
