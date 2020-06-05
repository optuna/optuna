import pytest

from optuna._imports import try_import


def test_try_import_successful() -> None:
    with try_import() as imports:
        pass
    assert imports.is_successful()
    imports.check()


def test_try_import_failing() -> None:
    with try_import() as imports:
        raise ImportError
    assert not imports.is_successful()
    with pytest.raises(ImportError):
        imports.check()

    with try_import() as imports:
        raise SyntaxError
    assert not imports.is_successful()
    with pytest.raises(ImportError):
        imports.check()

    with pytest.raises(NotImplementedError):
        with try_import() as imports:
            raise NotImplementedError
    assert imports.is_successful()  # No imports failed.
    imports.check()
