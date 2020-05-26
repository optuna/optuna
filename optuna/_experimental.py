import functools
import inspect
from typing import Any
from typing import Callable
import warnings

from optuna.exceptions import ExperimentalWarning


# White spaces of each line are necessary to beautifully rendered documentation.
# NOTE(crcrpar): When `experimental` decorator is applied to member methods, these lines require
# another four spaces.
_EXPERIMENTAL_DOCSTRING_TEMPLATE = """

    .. note::
        Added in v{ver} as an experimental feature. The interface may change in newer versions
        without prior notice. See https://github.com/optuna/optuna/releases/tag/v{ver}.
"""


def _validate_version(version: str) -> None:

    if not isinstance(version, str) or len(version.split(".")) != 3:
        raise ValueError(
            "Invalid version specification. Must follow `x.y.z` format but `{}` is given".format(
                version
            )
        )


def experimental(version: str, name: str = None) -> Any:
    """Decorate class or function as experimental.

    Args:
        version: The first version that supports the target feature.
        name: The name of the feature. Defaults to the function or class name. Optional.
    """

    _validate_version(version)

    def _experimental_wrapper(f: Any) -> Any:
        # f is either func or class.

        def _experimental_func(func: Callable[[Any], Any]) -> Callable[[Any], Any]:

            docstring = _EXPERIMENTAL_DOCSTRING_TEMPLATE.format(ver=version)
            if func.__doc__ is None:
                func.__doc__ = ""
            func.__doc__ += docstring

            # TODO(crcrpar): Annotate this correctly.
            @functools.wraps(func)
            def new_func(*args: Any, **kwargs: Any) -> Any:
                """Wrapped function."""

                warnings.warn(
                    "{} is experimental (supported from v{}). "
                    "The interface can change in the future.".format(
                        name if name is not None else func.__name__, version
                    ),
                    ExperimentalWarning,
                )

                return func(*args, **kwargs)  # type: ignore

            return new_func

        def _experimental_class(cls: Any) -> Any:
            """Decorates a class as experimental.

            This decorator is supposed to be applied to the experimental class.
            """

            _original_init = cls.__init__

            @functools.wraps(_original_init)
            def wrapped_init(self, *args, **kwargs) -> None:  # type: ignore
                warnings.warn(
                    "{} is experimental (supported from v{}). "
                    "The interface can change in the future.".format(
                        name if name is not None else cls.__name__, version
                    ),
                    ExperimentalWarning,
                )

                _original_init(self, *args, **kwargs)

            cls.__init__ = wrapped_init

            if cls.__doc__ is None:
                cls.__doc__ = ""
            cls.__doc__ += _EXPERIMENTAL_DOCSTRING_TEMPLATE.format(ver=version)
            return cls

        return _experimental_class(f) if inspect.isclass(f) else _experimental_func(f)

    return _experimental_wrapper
