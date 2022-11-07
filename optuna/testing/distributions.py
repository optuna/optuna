from typing import Any
from typing import Dict

from optuna.distributions import BaseDistribution


class UnsupportedDistribution(BaseDistribution):
    def single(self) -> bool:

        return False

    def _contains(self, param_value_in_internal_repr: float) -> bool:

        return True

    def _asdict(self) -> Dict:

        return {}

    def to_internal_repr(self, param_value_in_external_repr: Any) -> float:
        return float(param_value_in_external_repr)
