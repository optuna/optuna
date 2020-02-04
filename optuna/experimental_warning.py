import functools
from typing import Any
from typing import Callable
import warnings


def experimental_version(version: str) -> Any:

    def experimental(func: Callable[[Any], Any]) -> Callable[[Any], Any]:

        # TODO(crcrpar): Annotate this correctly.
        @functools.wraps(func)
        def new_func(*args: Any, **kwargs: Any) -> Any:
            """Wrapped function."""

            warnings.simplefilter('always', UserWarning)
            warnings.warn(
                "{} is experimental (from version: {}). "
                "The interface can change in the future".format(func.__name__, version))

            # TODO(crcrpar): Make this works
            if func.__doc__ is not None:
                func.__doc__ += """
.. note::
    Added in version {} as experimental feature. The interface can change in the future.
""".format(version)
            return func(*args, **kwargs)  # type: ignore

        return new_func

    return experimental


def experimental_class_version(version: str) -> Any:

    def experimental_class(cls: Any) -> Any:
        """Decorates a class as experimental.

        This decorator is supposed to be applied to the experimental class.
        """

        warnings.simplefilter('always', UserWarning)
        warnings.warn(
            "{} is experimental (from version: {}). The interface can change in the future".format(
                cls.__name__, version),
            category=UserWarning)
        # TODO(crcrpar): Make this works
        cls.__doc__ += """

.. note::
    Added in version {} as experimental feature. The interface can change in the future.
""".format(version)
        return cls

    return experimental_class
