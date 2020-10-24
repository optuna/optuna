import copy
import datetime
import gc
import math
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
import warnings

import joblib
from joblib import delayed
from joblib import Parallel

import optuna
from optuna import exceptions
from optuna import logging
from optuna import progress_bar as pbar_module
from optuna import storages
from optuna import trial as trial_module
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


_logger = logging.get_logger(__name__)


def _optimize(
    study: "optuna.Study",
    func: "optuna.study.ObjectiveFuncType",
    n_trials: Optional[int] = None,
    timeout: Optional[float] = None,
    n_jobs: int = 1,
    catch: Tuple[Type[Exception], ...] = (),
    callbacks: Optional[List[Callable[["optuna.Study", FrozenTrial], None]]] = None,
    gc_after_trial: bool = False,
    show_progress_bar: bool = False,
) -> None:
    if not isinstance(catch, tuple):
        raise TypeError(
            "The catch argument is of type '{}' but must be a tuple.".format(type(catch).__name__)
        )

    if not study._optimize_lock.acquire(False):
        raise RuntimeError("Nested invocation of `Study.optimize` method isn't allowed.")

    # TODO(crcrpar): Make progress bar work when n_jobs != 1.
    progress_bar = pbar_module._ProgressBar(show_progress_bar and n_jobs == 1, n_trials, timeout)

    study._stop_flag = False

    try:
        if n_jobs == 1:
            _optimize_sequential(
                study,
                func,
                n_trials,
                timeout,
                catch,
                callbacks,
                gc_after_trial,
                reseed_sampler_rng=False,
                time_start=None,
                progress_bar=progress_bar,
            )
        else:
            if show_progress_bar:
                warnings.warn("Progress bar only supports serial execution (`n_jobs=1`).")

            time_start = datetime.datetime.now()

            def _should_stop() -> bool:
                if study._stop_flag:
                    return True

                if timeout is not None:
                    # This is needed for mypy.
                    t: float = timeout
                    return (datetime.datetime.now() - time_start).total_seconds() > t

                return False

            if n_trials is not None:
                _iter = iter(range(n_trials))
            else:
                _iter = iter(_should_stop, True)

            with Parallel(n_jobs=n_jobs, prefer="threads") as parallel:
                if not isinstance(
                    parallel._backend, joblib.parallel.ThreadingBackend
                ) and isinstance(study._storage, storages.InMemoryStorage):
                    warnings.warn(
                        "The default storage cannot be shared by multiple processes. "
                        "Please use an RDB (RDBStorage) when you use joblib for "
                        "multi-processing. The usage of RDBStorage can be found in "
                        "https://optuna.readthedocs.io/en/stable/tutorial/rdb.html.",
                        UserWarning,
                    )

                parallel(
                    delayed(_optimize_sequential)(
                        study,
                        func,
                        1,
                        timeout,
                        catch,
                        callbacks,
                        gc_after_trial,
                        reseed_sampler_rng=True,
                        time_start=time_start,
                        progress_bar=None,
                    )
                    for _ in _iter
                )
    finally:
        study._optimize_lock.release()
        progress_bar.close()


def _optimize_sequential(
    study: "optuna.Study",
    func: "optuna.study.ObjectiveFuncType",
    n_trials: Optional[int],
    timeout: Optional[float],
    catch: Tuple[Type[Exception], ...],
    callbacks: Optional[List[Callable[["optuna.Study", FrozenTrial], None]]],
    gc_after_trial: bool,
    reseed_sampler_rng: bool,
    time_start: Optional[datetime.datetime],
    progress_bar: Optional[pbar_module._ProgressBar],
) -> None:
    if reseed_sampler_rng:
        study.sampler.reseed_rng()

    i_trial = 0

    if time_start is None:
        time_start = datetime.datetime.now()

    while True:
        if study._stop_flag:
            break

        if n_trials is not None:
            if i_trial >= n_trials:
                break
            i_trial += 1

        if timeout is not None:
            elapsed_seconds = (datetime.datetime.now() - time_start).total_seconds()
            if elapsed_seconds >= timeout:
                break

        try:
            trial = _run_trial(study, func, catch)
        except Exception:
            raise
        finally:
            # The following line mitigates memory problems that can be occurred in some
            # environments (e.g., services that use computing containers such as CircleCI).
            # Please refer to the following PR for further details:
            # https://github.com/optuna/optuna/pull/325.
            if gc_after_trial:
                gc.collect()

        if callbacks is not None:
            frozen_trial = copy.deepcopy(study._storage.get_trial(trial._trial_id))
            for callback in callbacks:
                callback(study, frozen_trial)

        if progress_bar is not None:
            progress_bar.update((datetime.datetime.now() - time_start).total_seconds())

    study._storage.remove_session()


def _run_trial(
    study: "optuna.Study",
    func: "optuna.study.ObjectiveFuncType",
    catch: Tuple[Type[Exception], ...],
) -> trial_module.Trial:
    trial = study._ask()

    trial_id = trial._trial_id
    trial_number = trial.number

    try:
        value = func(trial)
    except exceptions.TrialPruned as e:
        # Register the last intermediate value if present as the value of the trial.
        # TODO(hvy): Whether a pruned trials should have an actual value can be discussed.
        frozen_trial = study._storage.get_trial(trial_id)
        last_step = frozen_trial.last_step
        study._tell(
            trial,
            TrialState.PRUNED,
            None if last_step is None else frozen_trial.intermediate_values[last_step],
        )
        _logger.info("Trial {} pruned. {}".format(trial_number, str(e)))
        return trial
    except Exception as e:
        message = "Trial {} failed because of the following error: {}".format(
            trial_number, repr(e)
        )
        study._storage.set_trial_system_attr(trial_id, "fail_reason", message)
        study._tell(trial, TrialState.FAIL, None)
        _logger.warning(message, exc_info=True)
        if isinstance(e, catch):
            return trial
        raise

    try:
        value = float(value)
    except (
        ValueError,
        TypeError,
    ):
        message = (
            "Trial {} failed, because the returned value from the "
            "objective function cannot be cast to float. Returned value is: "
            "{}".format(trial_number, repr(value))
        )
        study._storage.set_trial_system_attr(trial_id, "fail_reason", message)
        study._tell(trial, TrialState.FAIL, None)
        _logger.warning(message)
        return trial

    if math.isnan(value):
        message = "Trial {} failed, because the objective function returned {}.".format(
            trial_number, value
        )
        study._storage.set_trial_system_attr(trial_id, "fail_reason", message)
        study._tell(trial, TrialState.FAIL, None)
        _logger.warning(message)
        return trial

    study._tell(trial, TrialState.COMPLETE, value)
    study._log_completed_trial(trial, value)
    return trial
