import functools
import inspect
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


def _make_func_spec_str(func: Callable[..., Any]) -> str:

    name = func.__name__
    argspec = inspect.getfullargspec(func)

    n_defaults = len(argspec.defaults) if argspec.defaults is not None else 0

    if n_defaults > 0:
        args = ', '.join(argspec.args[:-n_defaults])
        with_default_values = ', '.join(
            [
                '{}={}'.format(a, d)
                for a, d in zip(argspec.args[-n_defaults:], argspec.defaults)  # type: ignore
            ]
        )
    else:
        args = ', '.join(argspec.args)
        with_default_values = ''

    if len(args) > 0 and len(with_default_values) > 0:
        args += ', '

    str_args_description = '(' + args + with_default_values + ')\n\n'
    return name + str_args_description


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
        func.__doc__ = _make_func_spec_str(func) + func.__doc__ + docstring

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

        def wrapped_init(self, *args, **kwargs) -> None:  # type: ignore
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
        cls.__doc__ = (
            _make_func_spec_str(_original_init) +
            cls.__doc__ +
            _EXPERIMENTAL_DOCSTRING_TEMPLATE.format(version, version)
        )
        return cls

    return _experimental_class
