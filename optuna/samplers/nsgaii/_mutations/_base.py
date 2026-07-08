from __future__ import annotations

import abc
from typing import TYPE_CHECKING

from optuna._experimental import experimental_class


if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np

    from optuna.distributions import CategoricalChoiceType
    from optuna.study import Study


@experimental_class("5.0.0")
class BaseMutation(abc.ABC):
    """Base class for mutations.

    A mutation operation is used by :class:`~optuna.samplers.NSGAIISampler`
    to mutate a parameter when creating a new individual.
    """

    def __str__(self) -> str:
        return self.__class__.__name__


@experimental_class("5.0.0")
class NumericalMutation(BaseMutation):
    """Base class for numerical mutations.

    A numerical mutation operation is used by :class:`~optuna.samplers.NSGAIISampler`
    to mutate a numerical parameter when creating a new individual.
    """

    @abc.abstractmethod
    def mutation(
        self,
        param: float,
        rng: np.random.RandomState,
        study: Study,
        search_space_bounds: np.ndarray,
    ) -> float:
        """Mutate the given parameter.

        Args:
            param:
                A parameter value in the transformed numerical search space.
            rng:
                An instance of ``numpy.random.RandomState``.
            study:
                Target study object.
            search_space_bounds:
                A ``numpy.ndarray`` with shape ``(2,)`` representing the numerical
                distribution bounds constructed from transformed search space.

        Returns:
            A mutated parameter value in the transformed numerical search space.
        """

        raise NotImplementedError


@experimental_class("5.0.0")
class CategoricalMutation(BaseMutation):
    """Base class for categorical mutations.

    A categorical mutation operation is used by :class:`~optuna.samplers.NSGAIISampler`
    to mutate a categorical parameter when creating a new individual.
    """

    @abc.abstractmethod
    def mutation(
        self,
        param: CategoricalChoiceType,
        rng: np.random.RandomState,
        study: Study,
        choices: Sequence[CategoricalChoiceType],
    ) -> CategoricalChoiceType:
        """Mutate the given parameter.

        Args:
            param:
                A categorical parameter value.
            rng:
                An instance of ``numpy.random.RandomState``.
            study:
                Target study object.
            choices:
                Parameter value candidates.

        Returns:
            A mutated categorical parameter value.
        """

        raise NotImplementedError


@experimental_class("5.0.0")
class MixedMutation(BaseMutation):
    """Composite mutation for numerical and categorical parameters.

    A mixed mutation operation is used by :class:`~optuna.samplers.NSGAIISampler`
    to apply different mutation operations depending on the distribution type.

    Args:
        numerical:
            A mutation operation for numerical parameters.
        categorical:
            A mutation operation for categorical parameters.
    """

    def __init__(
        self,
        *,
        numerical: NumericalMutation | None = None,
        categorical: CategoricalMutation | None = None,
    ) -> None:
        if numerical is None and categorical is None:
            raise ValueError("At least one mutation must be specified.")

        self._numerical = numerical
        self._categorical = categorical

    @property
    def numerical(self) -> NumericalMutation | None:
        return self._numerical

    @property
    def categorical(self) -> CategoricalMutation | None:
        return self._categorical
