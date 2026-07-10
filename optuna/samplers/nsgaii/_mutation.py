from __future__ import annotations

from typing import Any
from typing import TYPE_CHECKING

import numpy as np

from optuna._transform import _SearchSpaceTransform
from optuna.distributions import BaseDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.samplers.nsgaii._mutations._base import BaseMutation


if TYPE_CHECKING:
    from optuna.study import Study


_NUMERICAL_DISTRIBUTIONS = (
    FloatDistribution,
    IntDistribution,
)


def perform_mutation(
    mutation: BaseMutation,
    rng: np.random.RandomState,
    study: Study,
    distribution: BaseDistribution,
    value: Any,
) -> Any | None:
    if not isinstance(distribution, _NUMERICAL_DISTRIBUTIONS):
        return None

    transform = _SearchSpaceTransform({"": distribution})
    trans_value = transform.transform({"": value})
    trans_mutated_value = mutation.mutation(trans_value.item(), rng, study, transform.bounds[0])
    trans_mutated_value = np.clip(
        trans_mutated_value, transform.bounds[0, 0], transform.bounds[0, 1]
    )

    return transform.untransform(np.array([trans_mutated_value]))[""]
