from __future__ import annotations

import numpy


class LazyRandomState:

    """Lazy Random State class.


    This is a class to initialize the random state just before use to prevent
    duplication of the same random state when deepcopy is applied to the instance of sampler.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng: numpy.random.RandomState | None = None
        if seed is not None:
            self.rng.seed(seed=seed)

    def _set_rng(self) -> None:
        self._rng = numpy.random.RandomState()

    @property
    def rng(self) -> numpy.random.RandomState:
        if self._rng is None:
            self._set_rng()
        assert self._rng is not None
        return self._rng
