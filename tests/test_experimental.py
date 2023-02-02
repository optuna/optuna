from typing import Any

import pytest

from optuna import _experimental
from optuna.exceptions import ExperimentalWarning


def _sample_func() -> int:
    return 10


class _Sample:
    def __init__(self, a: Any, b: Any, c: Any) -> None:
        pass

    def _method(self) -> None:
        """summary

        detail
        """
        pass

    def _method_expected(self) -> None:
        """summary

        detail

        .. note::
            Added in v1.1.0 as an experimental feature. The interface may change in newer versions
            without prior notice. See https://github.com/optuna/optuna/releases/tag/v1.1.0.
        """
        pass


@pytest.mark.parametrize("version", ["1.1", 100, None])
def test_experimental_raises_error_for_invalid_version(version: Any) -> None:
    with pytest.raises(ValueError):
        _experimental.experimental_func(version)

    with pytest.raises(ValueError):
        _experimental.experimental_class(version)


def test_experimental_func_decorator() -> None:
    version = "1.1.0"
    decorator_experimental = _experimental.experimental_func(version)
    assert callable(decorator_experimental)

    decorated_func = decorator_experimental(_sample_func)
    assert decorated_func.__name__ == _sample_func.__name__
    assert decorated_func.__doc__ == _experimental._EXPERIMENTAL_NOTE_TEMPLATE.format(ver=version)

    with pytest.warns(ExperimentalWarning):
        decorated_func()


def test_experimental_instance_method_decorator() -> None:
    version = "1.1.0"
    decorator_experimental = _experimental.experimental_func(version)
    assert callable(decorator_experimental)

    decorated_method = decorator_experimental(_Sample._method)
    assert decorated_method.__name__ == _Sample._method.__name__
    assert decorated_method.__doc__ == _Sample._method_expected.__doc__

    with pytest.warns(ExperimentalWarning):
        decorated_method(None)  # type: ignore


def test_experimental_class_decorator() -> None:
    version = "1.1.0"
    decorator_experimental = _experimental.experimental_class(version)
    assert callable(decorator_experimental)

    decorated_class = decorator_experimental(_Sample)
    assert decorated_class.__name__ == "_Sample"
    assert decorated_class.__init__.__name__ == "__init__"
    assert decorated_class.__doc__ == _experimental._EXPERIMENTAL_NOTE_TEMPLATE.format(ver=version)

    with pytest.warns(ExperimentalWarning):
        decorated_class("a", "b", "c")


def test_experimental_class_decorator_name() -> None:
    name = "foo"
    decorator_experimental = _experimental.experimental_class("1.1.0", name=name)
    decorated_sample = decorator_experimental(_Sample)

    with pytest.warns(ExperimentalWarning) as record:
        decorated_sample("a", "b", "c")

    assert isinstance(record.list[0].message, Warning)
    assert name in record.list[0].message.args[0]


def test_experimental_decorator_name() -> None:
    name = "bar"
    decorator_experimental = _experimental.experimental_func("1.1.0", name=name)
    decorated_sample_func = decorator_experimental(_sample_func)

    with pytest.warns(ExperimentalWarning) as record:
        decorated_sample_func()

    assert isinstance(record.list[0].message, Warning)
    assert name in record.list[0].message.args[0]
