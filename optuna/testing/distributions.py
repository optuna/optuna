from typing import Dict

from optuna.distributions import BaseDistribution


class UnsupportedDistribution(BaseDistribution):
    def single(self) -> bool:

        return False

    def _contains(self, param_value_in_internal_repr: float) -> bool:

        return True

    def _asdict(self) -> Dict:

        return {}
