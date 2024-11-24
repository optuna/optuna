from __future__ import annotations

from typing import Any

import pytest

from optuna import _deprecated


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

        .. warning::
            Deprecated in v1.1.0. This feature will be removed in the future. The removal of this
            feature is currently scheduled for v3.0.0, but this schedule is subject to change.
            See https://github.com/optuna/optuna/releases/tag/v1.1.0.
        """
        pass


@pytest.mark.parametrize("deprecated_version", ["1.1", 100, None, "2.0.0"])
@pytest.mark.parametrize("removed_version", ["1.1", 10, "1.0.0"])
def test_deprecation_raises_error_for_invalid_version(
    deprecated_version: Any, removed_version: Any
) -> None:
    with pytest.raises(ValueError):
        _deprecated.deprecated_func(deprecated_version, removed_version)

    with pytest.raises(ValueError):
        _deprecated.deprecated_class(deprecated_version, removed_version)


def test_deprecation_decorator() -> None:
    deprecated_version = "1.1.0"
    removed_version = "3.0.0"
    decorator_deprecation = _deprecated.deprecated_func(deprecated_version, removed_version)
    assert callable(decorator_deprecation)

    def _func() -> int:
        return 10

    decorated_func = decorator_deprecation(_func)
    assert decorated_func.__name__ == _func.__name__
    assert decorated_func.__doc__ == _deprecated._DEPRECATION_NOTE_TEMPLATE.format(
        d_ver=deprecated_version, r_ver=removed_version
    )

    with pytest.warns(FutureWarning):
        decorated_func()


def test_deprecation_instance_method_decorator() -> None:
    deprecated_version = "1.1.0"
    removed_version = "3.0.0"
    decorator_deprecation = _deprecated.deprecated_func(deprecated_version, removed_version)
    assert callable(decorator_deprecation)

    decorated_method = decorator_deprecation(_Sample._method)
    assert decorated_method.__name__ == _Sample._method.__name__
    assert decorated_method.__doc__ == _Sample._method_expected.__doc__

    with pytest.warns(FutureWarning):
        decorated_method(None)  # type: ignore


def test_deprecation_class_decorator() -> None:
    deprecated_version = "1.1.0"
    removed_version = "3.0.0"
    decorator_deprecation = _deprecated.deprecated_class(deprecated_version, removed_version)
    assert callable(decorator_deprecation)

    decorated_class = decorator_deprecation(_Sample)
    assert decorated_class.__name__ == "_Sample"
    assert decorated_class.__init__.__name__ == "__init__"
    assert decorated_class.__doc__ == _deprecated._DEPRECATION_NOTE_TEMPLATE.format(
        d_ver=deprecated_version, r_ver=removed_version
    )

    with pytest.warns(FutureWarning):
        decorated_class("a", "b", "c")


def test_deprecation_class_decorator_name() -> None:
    name = "foo"
    decorator_deprecation = _deprecated.deprecated_class("1.1.0", "3.0.0", name=name)
    decorated_sample = decorator_deprecation(_Sample)

    with pytest.warns(FutureWarning) as record:
        decorated_sample("a", "b", "c")

    assert isinstance(record.list[0].message, Warning)
    assert name in record.list[0].message.args[0]


def test_deprecation_decorator_name() -> None:
    def _func() -> int:
        return 10

    name = "bar"
    decorator_deprecation = _deprecated.deprecated_func("1.1.0", "3.0.0", name=name)
    decorated_sample_func = decorator_deprecation(_func)

    with pytest.warns(FutureWarning) as record:
        decorated_sample_func()

    assert isinstance(record.list[0].message, Warning)
    assert name in record.list[0].message.args[0]


@pytest.mark.parametrize("text", [None, "", "test", "test" * 100])
def test_deprecation_text_specified(text: str | None) -> None:
    def _func() -> int:
        return 10

    decorator_deprecation = _deprecated.deprecated_func("1.1.0", "3.0.0", text=text)
    decorated_func = decorator_deprecation(_func)
    expected_func_doc = _deprecated._DEPRECATION_NOTE_TEMPLATE.format(d_ver="1.1.0", r_ver="3.0.0")
    if text is None:
        pass
    elif len(text) > 0:
        expected_func_doc += "\n\n    " + text + "\n"
    else:
        expected_func_doc += "\n\n\n"
    assert decorated_func.__name__ == _func.__name__
    assert decorated_func.__doc__ == expected_func_doc

    with pytest.warns(FutureWarning) as record:
        decorated_func()
    assert isinstance(record.list[0].message, Warning)
    expected_warning_message = _deprecated._DEPRECATION_WARNING_TEMPLATE.format(
        name="_func", d_ver="1.1.0", r_ver="3.0.0"
    )
    if text is not None:
        expected_warning_message += " " + text
    assert record.list[0].message.args[0] == expected_warning_message


@pytest.mark.parametrize("text", [None, "", "test", "test" * 100])
def test_deprecation_class_text_specified(text: str | None) -> None:
    class _Class:
        def __init__(self, a: Any, b: Any, c: Any) -> None:
            pass

    decorator_deprecation = _deprecated.deprecated_class("1.1.0", "3.0.0", text=text)
    decorated_class = decorator_deprecation(_Class)
    expected_class_doc = _deprecated._DEPRECATION_NOTE_TEMPLATE.format(
        d_ver="1.1.0", r_ver="3.0.0"
    )
    if text is None:
        pass
    elif len(text) > 0:
        expected_class_doc += "\n\n    " + text + "\n"
    else:
        expected_class_doc += "\n\n\n"
    assert decorated_class.__name__ == _Class.__name__
    assert decorated_class.__doc__ == expected_class_doc

    with pytest.warns(FutureWarning) as record:
        decorated_class(None, None, None)
    assert isinstance(record.list[0].message, Warning)
    expected_warning_message = _deprecated._DEPRECATION_WARNING_TEMPLATE.format(
        name="_Class", d_ver="1.1.0", r_ver="3.0.0"
    )
    if text is not None:
        expected_warning_message += " " + text
    assert record.list[0].message.args[0] == expected_warning_message


def test_deprecation_decorator_default_removed_version() -> None:
    deprecated_version = "1.1.0"
    removed_version = "3.0.0"
    decorator_deprecation = _deprecated.deprecated_func(deprecated_version, removed_version)
    assert callable(decorator_deprecation)

    def _func() -> int:
        return 10

    decorated_func = decorator_deprecation(_func)
    assert decorated_func.__name__ == _func.__name__
    assert decorated_func.__doc__ == _deprecated._DEPRECATION_NOTE_TEMPLATE.format(
        d_ver=deprecated_version, r_ver=removed_version
    )

    with pytest.warns(FutureWarning):
        decorated_func()
