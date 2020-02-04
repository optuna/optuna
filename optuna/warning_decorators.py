import functools
from typing import Any
from typing import Callable
import warnings


_EXPERIMENTAL_DOCSTRING_TEMPLATE = """

.. note::
    Added in v{} as experimental feature. The interface can change in the future.
    See the details in https://github.com/optuna/optuna/releases/tag/v{}.
"""


class ExperimentalWarning(UserWarning):
    """Experimental Warning class.

    This implementation exists here because the policy of `FutureWarning` has been changed
    since Python 3.7 was released.
    """

    pass


def _validate_version(version: str) -> None:

    if not isinstance(version, str) or len(version.split('.')) != 3:
        raise ValueError(
            'Invalid version specification. Must follow `x.y.z` format but `{}` is given'.format(
                version))


def experimental(version: str) -> Any:
    """Decorator for experimental functions or methods.

    Args:
        version: The version number.
    """

    _validate_version(version)

    def _experimental(func: Callable[[Any], Any]) -> Callable[[Any], Any]:

        docstring = _EXPERIMENTAL_DOCSTRING_TEMPLATE.format(version, version)
        if func.__doc__ is None:
            func.__doc__ = ''
        func.__doc__ += docstring

        # TODO(crcrpar): Annotate this correctly.
        @functools.wraps(func)
        def new_func(*args: Any, **kwargs: Any) -> Any:
            """Wrapped function."""

            warnings.simplefilter('always', UserWarning)
            warnings.warn(
                "{} is experimental (supported from v{}). "
                "The interface can change in the future.".format(func.__name__, version),
                ExperimentalWarning
            )

            return func(*args, **kwargs)  # type: ignore

        return new_func

    return _experimental


def experimental_class(version: str) -> Any:
    """Decorator for experimental class implementation.

    Args:
        version: The version number.
    """

    _validate_version(version)

    def _experimental_class(cls: Any) -> Any:
        """Decorates a class as experimental.

        This decorator is supposed to be applied to the experimental class.
        """

        _original_init = cls.__init__

        def wrapped_init(self, *args: Any, **kwargs: Any) -> None:
            warnings.simplefilter('always', UserWarning)
            warnings.warn(
                "{} is experimental (supported from v{}). "
                "The interface can change in the future.".format(cls.__name__, version),
                ExperimentalWarning
            )

            _original_init(self, *args, **kwargs)

        cls.__init__ = wrapped_init

        if cls.__doc__ is None:
            cls.__doc__ = ''
        cls.__doc__ += _EXPERIMENTAL_DOCSTRING_TEMPLATE.format(version, version)
        return cls

    return _experimental_class
