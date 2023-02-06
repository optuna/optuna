import logging

import _pytest.capture
import _pytest.logging

import optuna.logging


def test_get_logger(caplog: _pytest.logging.LogCaptureFixture) -> None:
    # Log propagation is necessary for caplog to capture log outputs.
    optuna.logging.enable_propagation()

    logger = optuna.logging.get_logger("optuna.foo")
    with caplog.at_level(logging.INFO, logger="optuna.foo"):
        logger.info("hello")
    assert "hello" in caplog.text


def test_default_handler(capsys: _pytest.capture.CaptureFixture) -> None:
    # We need to reconstruct our default handler to properly capture stderr.
    optuna.logging._reset_library_root_logger()
    optuna.logging.set_verbosity(optuna.logging.INFO)

    library_root_logger = optuna.logging._get_library_root_logger()

    example_logger = optuna.logging.get_logger("optuna.bar")

    # Default handler enabled
    optuna.logging.enable_default_handler()
    assert library_root_logger.handlers
    example_logger.info("hey")
    _, err = capsys.readouterr()
    assert "hey" in err

    # Default handler disabled
    optuna.logging.disable_default_handler()
    assert not library_root_logger.handlers
    example_logger.info("yoyo")
    _, err = capsys.readouterr()
    assert "yoyo" not in err


def test_verbosity(capsys: _pytest.capture.CaptureFixture) -> None:
    # We need to reconstruct our default handler to properly capture stderr.
    optuna.logging._reset_library_root_logger()
    library_root_logger = optuna.logging._get_library_root_logger()
    example_logger = optuna.logging.get_logger("optuna.hoge")
    optuna.logging.enable_default_handler()

    # level INFO
    optuna.logging.set_verbosity(optuna.logging.INFO)
    assert library_root_logger.getEffectiveLevel() == logging.INFO
    example_logger.warning("hello-warning")
    example_logger.info("hello-info")
    example_logger.debug("hello-debug")
    _, err = capsys.readouterr()
    assert "hello-warning" in err
    assert "hello-info" in err
    assert "hello-debug" not in err

    # level WARNING
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    assert library_root_logger.getEffectiveLevel() == logging.WARNING
    example_logger.warning("bye-warning")
    example_logger.info("bye-info")
    example_logger.debug("bye-debug")
    _, err = capsys.readouterr()
    assert "bye-warning" in err
    assert "bye-info" not in err
    assert "bye-debug" not in err


def test_propagation(caplog: _pytest.logging.LogCaptureFixture) -> None:
    optuna.logging._reset_library_root_logger()
    logger = optuna.logging.get_logger("optuna.foo")

    # Propagation is disabled by default.
    with caplog.at_level(logging.INFO, logger="optuna"):
        logger.info("no-propagation")
    assert "no-propagation" not in caplog.text

    # Enable propagation.
    optuna.logging.enable_propagation()
    with caplog.at_level(logging.INFO, logger="optuna"):
        logger.info("enable-propagate")
    assert "enable-propagate" in caplog.text

    # Disable propagation.
    optuna.logging.disable_propagation()
    with caplog.at_level(logging.INFO, logger="optuna"):
        logger.info("disable-propagation")
    assert "disable-propagation" not in caplog.text
