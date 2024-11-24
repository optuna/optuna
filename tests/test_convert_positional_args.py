from __future__ import annotations

import re

import pytest

from optuna._convert_positional_args import convert_positional_args


def _sample_func(*, a: int, b: int, c: int) -> int:
    return a + b + c


class _SimpleClass:
    @convert_positional_args(previous_positional_arg_names=["self", "a", "b"])
    def simple_method(self, a: int, *, b: int, c: int = 1) -> None:
        pass


def test_convert_positional_args_decorator() -> None:
    previous_positional_arg_names: list[str] = []
    decorator_converter = convert_positional_args(
        previous_positional_arg_names=previous_positional_arg_names
    )

    decorated_func = decorator_converter(_sample_func)
    assert decorated_func.__name__ == _sample_func.__name__


def test_convert_positional_args_future_warning_for_methods() -> None:
    simple_class = _SimpleClass()
    with pytest.warns(FutureWarning) as record:
        simple_class.simple_method(1, 2, c=3)  # type: ignore
        simple_class.simple_method(1, b=2, c=3)  # No warning.
        simple_class.simple_method(a=1, b=2, c=3)  # No warning.

    assert len(record) == 1
    for warn in record.list:
        assert isinstance(warn.message, FutureWarning)
        assert "simple_method" in str(warn.message)


def test_convert_positional_args_future_warning() -> None:
    previous_positional_arg_names: list[str] = ["a", "b"]
    decorator_converter = convert_positional_args(
        previous_positional_arg_names=previous_positional_arg_names
    )
    assert callable(decorator_converter)

    decorated_func = decorator_converter(_sample_func)
    with pytest.warns(FutureWarning) as record:
        decorated_func(1, 2, c=3)  # type: ignore
        decorated_func(1, b=2, c=3)  # type: ignore
        decorated_func(a=1, b=2, c=3)  # No warning.

    assert len(record) == 2
    for warn in record.list:
        assert isinstance(warn.message, FutureWarning)
        assert _sample_func.__name__ in str(warn.message)


def test_convert_positional_args_mypy_type_inference() -> None:
    previous_positional_arg_names: list[str] = []
    decorator_converter = convert_positional_args(
        previous_positional_arg_names=previous_positional_arg_names
    )
    assert callable(decorator_converter)

    class _Sample:
        def __init__(self) -> None:
            pass

        def method(self) -> bool:
            return True

    def _func_sample() -> _Sample:
        return _Sample()

    def _func_none() -> None:
        pass

    ret_none = decorator_converter(_func_none)()
    assert ret_none is None

    ret_sample = decorator_converter(_func_sample)()
    assert isinstance(ret_sample, _Sample)
    assert ret_sample.method()


@pytest.mark.parametrize(
    "previous_positional_arg_names, raise_error",
    [(["a", "b", "c", "d"], True), (["a", "d"], True), (["b", "a"], False)],
)
def test_convert_positional_args_invalid_previous_positional_arg_names(
    previous_positional_arg_names: list[str], raise_error: bool
) -> None:
    decorator_converter = convert_positional_args(
        previous_positional_arg_names=previous_positional_arg_names
    )
    assert callable(decorator_converter)

    if raise_error:
        with pytest.raises(AssertionError) as record:
            decorator_converter(_sample_func)
        res = re.findall(r"({.+?}|set\(\))", str(record.value))
        assert len(res) == 2
        assert eval(res[0]) == set(previous_positional_arg_names)
        assert eval(res[1]) == set(["a", "b", "c"])
    else:
        decorator_converter(_sample_func)


def test_convert_positional_args_invalid_positional_args() -> None:
    previous_positional_arg_names: list[str] = ["a", "b"]
    decorator_converter = convert_positional_args(
        previous_positional_arg_names=previous_positional_arg_names
    )
    assert callable(decorator_converter)

    decorated_func = decorator_converter(_sample_func)
    with pytest.warns(FutureWarning):
        with pytest.raises(TypeError) as record:
            decorated_func(1, 2, 3)  # type: ignore
        assert str(record.value) == "_sample_func() takes 2 positional arguments but 3 were given."

        with pytest.raises(TypeError) as record:
            decorated_func(1, 3, b=2)  # type: ignore
        assert str(record.value) == "_sample_func() got multiple values for arguments {'b'}."
