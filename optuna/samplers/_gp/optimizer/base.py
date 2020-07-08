import abc
from typing import Any
from typing import Callable
from typing import Optional

import numpy as np


class BaseOptimizer(object, metaclass=abc.ABCMeta):
    """Base class for acquisition optimizers"""

    @abc.abstractmethod
    def optimize(
        self, f: Callable[[Any], Any], df: Optional[Callable[[Any], Any]] = None
    ) -> np.ndarray:
        """Optimize (maximize) the given objective function f.

        Args:
            f:
                The objective function to be maximized.
            df:
                The gradient of the objective function.
        Returns:
            A parameter list which maximize the objective function.
        """

        raise NotImplementedError
