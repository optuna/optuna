import logging
from typing import Any
from typing import Optional

from tqdm.auto import tqdm

from optuna._experimental import experimental
from optuna import logging as optuna_logging

_tqdm_handler = None  # type: Optional[_TqdmLoggingHandler]


# Reference: https://gist.github.com/hvy/8b80c2cedf02b15c24f85d1fa17ebe02
class _TqdmLoggingHandler(logging.StreamHandler):
    def emit(self, record: Any) -> None:
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)


class _ProgressBar(object):
    """Progress Bar implementation for `Study.optimize` on the top of `tqdm`.

    Args:
        is_valid:
            Whether to show progress bars in `Study.optimize`.
        n_trials:
            The number of trials.
        timeout:
            Stop study after the given number of second(s).
    """

    def __init__(
        self, is_valid: bool, n_trials: Optional[int] = None, timeout: Optional[float] = None,
    ) -> None:
        self._is_valid = is_valid
        self._n_trials = n_trials
        self._timeout = timeout

        if self._is_valid:
            self._init_valid()

    # TODO(hvy): Remove initialization indirection via this method when the progress bar is no
    # longer experimental.
    @experimental("1.2.0", name="Progress bar")
    def _init_valid(self) -> None:
        self._progress_bar = tqdm(range(self._n_trials) if self._n_trials is not None else None)
        global _tqdm_handler

        _tqdm_handler = _TqdmLoggingHandler()
        _tqdm_handler.setLevel(logging.INFO)
        _tqdm_handler.setFormatter(optuna_logging.create_default_formatter())
        optuna_logging.disable_default_handler()
        optuna_logging._get_library_root_logger().addHandler(_tqdm_handler)

    def update(self, elapsed_seconds: Optional[float]) -> None:
        """Update the progress bars if ``is_valid`` is ``True``.

        Args:
            elapsed_seconds:
                The time past since `Study.optimize` started.
        """
        if self._is_valid:
            self._progress_bar.update(1)
            if self._timeout is not None and elapsed_seconds is not None:
                self._progress_bar.set_postfix_str(
                    "{:.02f}/{} seconds".format(elapsed_seconds, self._timeout)
                )

    def close(self) -> None:
        """Close progress bars."""
        if self._is_valid:
            self._progress_bar.close()
            assert _tqdm_handler is not None
            optuna_logging._get_library_root_logger().removeHandler(_tqdm_handler)
            optuna_logging.enable_default_handler()
