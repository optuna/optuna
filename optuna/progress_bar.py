import logging
from typing import Any
from typing import Optional

from tqdm.auto import tqdm

from optuna import logging as optuna_logging
from optuna._experimental import experimental_func


_tqdm_handler: Optional["_TqdmLoggingHandler"] = None


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


class _ProgressBar:
    """Progress Bar implementation for :func:`~optuna.study.Study.optimize` on the top of `tqdm`.

    Args:
        is_valid:
            Whether to show progress bars in :func:`~optuna.study.Study.optimize`.
        n_trials:
            The number of trials.
        timeout:
            Stop study after the given number of second(s).
    """

    def __init__(
        self, is_valid: bool, n_trials: Optional[int] = None, timeout: Optional[float] = None
    ) -> None:

        self._is_valid = is_valid and (n_trials or timeout) is not None
        self._n_trials = n_trials
        self._timeout = timeout
        self._last_elapsed_seconds = 0.0

        if self._is_valid:
            self._init_valid()

    # TODO(hvy): Remove initialization indirection via this method when the progress bar is no
    # longer experimental.
    @experimental_func("1.2.0", name="Progress bar")
    def _init_valid(self) -> None:

        if self._n_trials is not None:
            self._progress_bar = tqdm(total=self._n_trials)

        else:
            fmt = "{percentage:3.0f}%|{bar}| {elapsed}/{desc}"
            self._progress_bar = tqdm(total=self._timeout, bar_format=fmt)

            # Using description string instead postfix string
            # to display formatted timeout, since postfix carries
            # extra comma space auto-format.
            # https://github.com/tqdm/tqdm/issues/712
            total = tqdm.format_interval(self._timeout)
            self._progress_bar.set_description_str(total)

        global _tqdm_handler

        _tqdm_handler = _TqdmLoggingHandler()
        _tqdm_handler.setLevel(logging.INFO)
        _tqdm_handler.setFormatter(optuna_logging.create_default_formatter())
        optuna_logging.disable_default_handler()
        optuna_logging._get_library_root_logger().addHandler(_tqdm_handler)

    def update(self, elapsed_seconds: float) -> None:
        """Update the progress bars if ``is_valid`` is :obj:`True`.

        Args:
            elapsed_seconds:
                The time past since :func:`~optuna.study.Study.optimize` started.
        """

        if self._is_valid:
            if self._n_trials is not None:
                self._progress_bar.update(1)
                if self._timeout is not None:
                    self._progress_bar.set_postfix_str(
                        "{:.02f}/{} seconds".format(elapsed_seconds, self._timeout)
                    )

            elif self._timeout is not None:
                time_diff = elapsed_seconds - self._last_elapsed_seconds
                if elapsed_seconds > self._timeout:
                    # Clip elapsed time to avoid tqdm warnings.
                    time_diff -= elapsed_seconds - self._timeout

                self._progress_bar.update(time_diff)
                self._last_elapsed_seconds = elapsed_seconds

            else:
                assert False

    def close(self) -> None:
        """Close progress bars."""

        if self._is_valid:
            self._progress_bar.close()
            assert _tqdm_handler is not None
            optuna_logging._get_library_root_logger().removeHandler(_tqdm_handler)
            optuna_logging.enable_default_handler()
