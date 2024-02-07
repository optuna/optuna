import pytest


pytestmark = pytest.mark.integration


def test_import() -> None:
    from optuna.integration import lightgbm  # NOQA
    from optuna.integration import LightGBMPruningCallback  # NOQA
    from optuna.integration import xgboost  # NOQA
    from optuna.integration import XGBoostPruningCallback  # NOQA

    with pytest.raises(ImportError):
        from optuna.integration import unknown_module  # type: ignore # NOQA


def test_module_attributes() -> None:
    import optuna

    assert hasattr(optuna.integration, "lightgbm")
    assert hasattr(optuna.integration, "xgboost")
    assert hasattr(optuna.integration, "LightGBMPruningCallback")
    assert hasattr(optuna.integration, "XGBoostPruningCallback")

    with pytest.raises(AttributeError):
        optuna.integration.unknown_attribute  # type: ignore
