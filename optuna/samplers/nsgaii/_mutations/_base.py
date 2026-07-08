from __future__ import annotations

import abc
from typing import TYPE_CHECKING

from build.lib.optuna._experimental import experimental_class


if TYPE_CHECKING:
    import numpy as np

    from optuna.study import Study


@experimental_class("5.0.0")
class BaseMutation(abc.ABC):
    """Base class for mutations.

    A mutation operation is used by :class:`~optuna.samplers.NSGAIISampler`
    to mutate a numerical parameter when creating a new individual.
    """

    def __str__(self) -> str:
        return self.__class__.__name__

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
