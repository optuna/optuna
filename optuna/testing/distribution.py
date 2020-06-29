from optuna.distributions import BaseDistribution
from optuna import type_checking

from typing import Dict


class UnsupportedDistribution(BaseDistribution):
    def single(self) -> bool:

        return False

    def _contains(self, param_value_in_internal_repr: float) -> bool:

        return True

    def _asdict(self) -> Dict:

        return {}
