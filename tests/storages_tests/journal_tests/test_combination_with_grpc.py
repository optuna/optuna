from __future__ import annotations

from concurrent.futures import as_completed
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

import pytest

import optuna
from optuna.storages import GrpcStorageProxy
from optuna.testing.storages import StorageSupplier


def pop_waiting_trial(port: int, study_name: str) -> int | None:
    storage = GrpcStorageProxy(port=port)
    storage.wait_server_ready(timeout=60)
    study = optuna.load_study(storage=storage, study_name=study_name)
    popped_trial_id = study._pop_waiting_trial_id()
    print(popped_trial_id)
    return popped_trial_id


@pytest.mark.parametrize("n_grpc_threads", [2, None])
def test_pop_waiting_trial_multiprocess_safe(n_grpc_threads: int | None) -> None:
    num_enqueued = 30
    # NOTE(nabenabe): Fewer threads in gRPC increases the probability of thread collision on the
    # proxy side. See https://github.com/optuna/optuna/issues/6084
    # However, only one thread guarantees no failure because each get_all_trials call in
    # _pop_waiting_trial_id happens sequentially. This is why, theoretically speaking, the failure
    # is likely to happen with two threads the most.
    thread_pool = ThreadPoolExecutor(n_grpc_threads) if n_grpc_threads is not None else None
    with StorageSupplier("grpc_journal_file", thread_pool=thread_pool) as proxy:
        assert isinstance(proxy, GrpcStorageProxy)
        port = proxy._port
        study = optuna.create_study(storage=proxy)
        for i in range(num_enqueued):
            study.enqueue_trial({"i": i})

        trial_id_set = set()
        # NOTE(nabe): When we fork our process while there are still active gRPC threads, gRPC will
        # skip its fork handlers (as the log repeatedly warns), which can leave internal locks in
        # an inconsistent state and often ends in a segfault shortly thereafter. In principle,
        # spawning, but not forking, works nicely because spawning does not clone the parent
        # process at all and it starts a brand-new process.
        mp_context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(10, mp_context=mp_context) as pool:
            futures = []
            for i in range(num_enqueued):
                future = pool.submit(pop_waiting_trial, port, study.study_name)
                futures.append(future)

            for future in as_completed(futures):
                trial_id = future.result()
                if trial_id is not None:
                    trial_id_set.add(trial_id)
        assert len(trial_id_set) == num_enqueued
