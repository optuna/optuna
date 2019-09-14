from optuna import type_checking
from optuna.distributions import BaseDistribution

if type_checking.TYPE_CHECKING:
    from typing import Dict  # NOQA


class UnsupportedDistribution(BaseDistribution):

    def single(self):
        # type: () -> bool

        return False

    def _contains(self, param_value_in_internal_repr):
        # type: (float) -> bool

        return True

    def _asdict(self):
        # type: () -> Dict

        return {}
