from optuna.distributions import BaseDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution


def _distribution_is_log(distribution: BaseDistribution) -> bool:
    if isinstance(distribution, FloatDistribution):
        return distribution.log

    if isinstance(distribution, IntDistribution):
        return distribution.log

    return False
