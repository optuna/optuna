from __future__ import annotations

from concurrent.futures import as_completed
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import multiprocessing
from typing import TYPE_CHECKING

import optuna
from optuna.storages import GrpcStorageProxy
from optuna.testing.storages import StorageSupplier


if TYPE_CHECKING:
    from concurrent.futures import Executor
    from typing import Callable
    from typing import Generator


@contextmanager
def grpc_journal_file_context() -> Generator[optuna.Study]:
    # NOTE(nabenabe): Fewer threads in gRPC increases the probability of thread collision on the
    # proxy side. See https://github.com/optuna/optuna/issues/6084
    # However, only one thread guarantees no failure because each get_all_trials call in
    # _pop_waiting_trial_id happens sequentially. This is why, theoretically speaking, the failure
    # is likely to happen with two threads the most.
    with StorageSupplier("grpc_journal_file", thread_pool=ThreadPoolExecutor(2)) as storage:
        study = optuna.create_study(storage=storage)
        for i in range(30):
            study.enqueue_trial({"i": i})
        yield study


def _pop_waiting_trial_id(study: optuna.Study) -> int | None:
    storage = study._storage
    assert isinstance(storage, GrpcStorageProxy)
    port = storage._port
    storage = GrpcStorageProxy(port=port)
    storage.wait_server_ready(timeout=60)
    study_in_spawned_proc = optuna.load_study(storage=storage, study_name=study.study_name)
    return study_in_spawned_proc._pop_waiting_trial_id()


def _verify_racing_condition(
    pool: Executor,
    study: optuna.Study,
    pop_waiting_trial_id_wrapper: Callable[[optuna.Study], int | None],
) -> None:
    n_enqueued = len(study.trials)
    futures = [pool.submit(pop_waiting_trial_id_wrapper, study) for _ in range(n_enqueued)]
    trial_id_set = set()
    for future in as_completed(futures):
        trial_id = future.result()
        if trial_id is not None:
            trial_id_set.add(trial_id)
    assert len(trial_id_set) == n_enqueued


def test_pop_waiting_trial_multiprocess_safe() -> None:
    with grpc_journal_file_context() as study:
        with ProcessPoolExecutor(10, mp_context=multiprocessing.get_context("spawn")) as pool:
            _verify_racing_condition(pool, study, _pop_waiting_trial_id)


def test_pop_waiting_trial_thread_safe() -> None:
    with grpc_journal_file_context() as study:
        with ThreadPoolExecutor(10) as pool:
            _verify_racing_condition(pool, study, lambda study: study._pop_waiting_trial_id())
