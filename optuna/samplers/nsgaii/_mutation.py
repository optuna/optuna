from __future__ import annotations

from typing import Any
from typing import TYPE_CHECKING

import numpy as np

from optuna._transform import _SearchSpaceTransform
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.samplers.nsgaii._mutations._base import BaseMutation
from optuna.samplers.nsgaii._mutations._base import CategoricalMutation
from optuna.samplers.nsgaii._mutations._base import MixedMutation
from optuna.samplers.nsgaii._mutations._base import NumericalMutation


if TYPE_CHECKING:
    from optuna.study import Study


_NUMERICAL_DISTRIBUTIONS = (
    FloatDistribution,
    IntDistribution,
)

_MUTATION_FALLBACK = object()


def perform_mutation(
    mutation: BaseMutation,
    rng: np.random.RandomState,
    study: Study,
    distribution: BaseDistribution,
    value: Any,
) -> Any:
    if isinstance(mutation, MixedMutation):
        if isinstance(distribution, _NUMERICAL_DISTRIBUTIONS):
            if mutation.numerical is None:
                return _MUTATION_FALLBACK
            mutation = mutation.numerical
        elif isinstance(distribution, CategoricalDistribution):
            if mutation.categorical is None:
                return _MUTATION_FALLBACK
            mutation = mutation.categorical
        else:
            return _MUTATION_FALLBACK

    if not isinstance(distribution, _NUMERICAL_DISTRIBUTIONS):
        if isinstance(distribution, CategoricalDistribution) and isinstance(
            mutation, CategoricalMutation
        ):
            mutated_value = mutation.mutation(value, rng, study, distribution.choices)
            distribution.to_internal_repr(mutated_value)
            return mutated_value

        return _MUTATION_FALLBACK

    if not isinstance(mutation, NumericalMutation):
        return _MUTATION_FALLBACK

    transform = _SearchSpaceTransform({"": distribution})
    trans_value = transform.transform({"": value})
    trans_mutated_value = mutation.mutation(trans_value.item(), rng, study, transform.bounds[0])
    trans_mutated_value = np.clip(
        trans_mutated_value, transform.bounds[0, 0], transform.bounds[0, 1]
    )

    return transform.untransform(np.array([trans_mutated_value]))[""]
