import pytest


def test_import():
    # type: () -> None

    from optuna.integration import chainer  # NOQA
    from optuna.integration import chainermn  # NOQA
    from optuna.integration import keras  # NOQA
    from optuna.integration import lightgbm  # NOQA
    from optuna.integration import mxnet  # NOQA
    from optuna.integration import tensorflow  # NOQA
    from optuna.integration import xgboost  # NOQA

    from optuna.integration import ChainerMNStudy  # NOQA
    from optuna.integration import ChainerPruningExtension  # NOQA
    from optuna.integration import KerasPruningCallback  # NOQA
    from optuna.integration import LightGBMPruningCallback  # NOQA
    from optuna.integration import MXNetPruningCallback  # NOQA
    from optuna.integration import TensorFlowPruningHook  # NOQA
    from optuna.integration import XGBoostPruningCallback  # NOQA

    with pytest.raises(ImportError):
        from optuna.integration import unknown_module  # type: ignore # NOQA


def test_module_attributes():
    # type: () -> None

    import optuna

    assert hasattr(optuna.integration, "chainer")
    assert hasattr(optuna.integration, "chainermn")
    assert hasattr(optuna.integration, "keras")
    assert hasattr(optuna.integration, "lightgbm")
    assert hasattr(optuna.integration, "mxnet")
    assert hasattr(optuna.integration, "tensorflow")
    assert hasattr(optuna.integration, "xgboost")
    assert hasattr(optuna.integration, "ChainerMNStudy")
    assert hasattr(optuna.integration, "ChainerPruningExtension")
    assert hasattr(optuna.integration, "KerasPruningCallback")
    assert hasattr(optuna.integration, "LightGBMPruningCallback")
    assert hasattr(optuna.integration, "MXNetPruningCallback")
    assert hasattr(optuna.integration, "TensorFlowPruningHook")
    assert hasattr(optuna.integration, "XGBoostPruningCallback")

    with pytest.raises(AttributeError):
        optuna.integration.unknown_attribute  # type: ignore
