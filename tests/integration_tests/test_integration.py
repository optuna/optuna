import pytest


pytestmark = pytest.mark.integration


def test_import() -> None:
    with pytest.raises(ImportError):
        from optuna.integration import unknown_module  # type: ignore # NOQA


def test_module_attributes() -> None:
    import optuna

    with pytest.raises(AttributeError):
        optuna.integration.unknown_attribute  # type: ignore
