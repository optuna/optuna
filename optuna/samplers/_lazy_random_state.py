from typing import Optional

import numpy


class LazyRandomState:

    """Lazy Random State class.


    This is a class to initialize just before using random state
    to solve the following issue.
    https://github.com/optuna/optuna/issues/4604
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng: Optional[numpy.random.RandomState] = None
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
