import decimal
import warnings

from optuna import logging


_logger = logging.get_logger(__name__)


def _adjust_discrete_uniform_high(name, low, high, q):
    # type: (str, float, float, float) -> float

    d_high = decimal.Decimal(str(high))
    d_low = decimal.Decimal(str(low))
    d_q = decimal.Decimal(str(q))

    d_r = d_high - d_low

    if d_r % d_q != decimal.Decimal("0"):
        high = float((d_r // d_q) * d_q + d_low)
        warnings.warn(
            "The range of parameter `{}` is not divisible by `q`, and is "
            "replaced by [{}, {}].".format(name, low, high)
        )

    return high
