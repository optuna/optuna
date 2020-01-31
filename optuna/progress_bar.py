import logging
from typing import Any
from typing import Optional

from tqdm.auto import tqdm

from optuna import logging as optuna_logging

_tqdm_handler = None  # type: Optional[TqdmLoggingHandler]


# Reference: https://gist.github.com/hvy/8b80c2cedf02b15c24f85d1fa17ebe02
class TqdmLoggingHandler(logging.StreamHandler):

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
        self,
        is_valid: bool,
        n_trials: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self._is_valid = is_valid
        self._n_trials = n_trials
        self._timeout = timeout

        if self._is_valid:
            self._progress_bar = tqdm(
                range(n_trials) if n_trials is not None else None,
                # position=1
            )
            global _tqdm_handler

            tqdm_handler = TqdmLoggingHandler()
            tqdm_handler.setLevel(logging.NOTSET)
            tqdm_handler.setFormatter(optuna_logging.create_default_formatter())
            optuna_logging.disable_default_handler()
            optuna_logging._get_library_root_logger().addHandler(tqdm_handler)

    def update(self, elapsed_seconds: Optional[float]) -> None:
        """Update the progress bars if ``is_valid`` is ``True``.

        Args:
            elapsed_seconds:
                The time past since `Study.optimize` started.
        """
        if self._is_valid:
            self._progress_bar.update(1)
            if elapsed_seconds is not None:
                self._progress_bar.set_postfix_str(
                    '{:.02f}/{} seconds'.format(elapsed_seconds, self._timeout))

    def set_description_str(self, msg: Optional[str]) -> None:
        """Update the best value message.

        Args:
            msg:
                The description of the latest best values.
        """
        # if self._is_valid:
        #     self._log_bar.set_description_str(msg)
        pass

    def close(self) -> None:
        """Close progress bars."""
        if self._is_valid:
            self._progress_bar.close()
            optuna_logging._get_library_root_logger().removeHandler(_tqdm_handler)
            optuna_logging.enable_default_handler()
