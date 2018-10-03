import gc
import pytest
from types import TracebackType  # NOQA
from typing import Any  # NOQA
from typing import Dict  # NOQA
from typing import Type  # NOQA

from optuna import create_study
from optuna.integration import minimize_chainermn
from optuna.storages import RDBStorage
from optuna import Study
from optuna.testing.storage import StorageSupplier
from optuna.trial import Trial  # NOQA

try:
    import chainermn
    from chainermn.communicators.communicator_base import CommunicatorBase  # NOQA
    _available = True
except ImportError:
    _available = False


STORAGE_MODES = ['new', 'common']


def setup_module():
    # type: () -> None

    StorageSupplier.setup_common_tempfile()


def teardown_module():
    # type: () -> None

    StorageSupplier.teardown_common_tempfile()


class Func(object):

    def __init__(self):
        # type: () -> None

        self.suggested_values = {}  # type: Dict[int, Dict[str, Any]]

    def __call__(self, trial, comm):
        # type: (Trial, CommunicatorBase) -> float

        x = trial.suggest_uniform('x', -10, 10)
        y = trial.suggest_loguniform('y', 20, 30)
        z = trial.suggest_categorical('z', (-1.0, 1.0))

        self.suggested_values[trial.trial_id] = {}
        self.suggested_values[trial.trial_id]['x'] = x
        self.suggested_values[trial.trial_id]['y'] = y
        self.suggested_values[trial.trial_id]['z'] = z

        return (x - 2) ** 2 + (y - 25) ** 2 + z


class MultiNodeStorageSupplier(StorageSupplier):

    def __init__(self, storage_specifier, comm):
        # type: (str, CommunicatorBase) -> None

        super(MultiNodeStorageSupplier, self).__init__(storage_specifier)
        self.comm = comm

    def __enter__(self):
        # type: () -> RDBStorage

        if self.comm.rank == 0:
            storage = super(MultiNodeStorageSupplier, self).__enter__()
            assert isinstance(storage, RDBStorage)
            url = str(storage.engine.url)
        else:
            url = 'dummy_url'

        url = self.comm.mpi_comm.bcast(url)
        return RDBStorage(url)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # type: (Type[BaseException], BaseException, TracebackType) -> None

        if self.comm.rank == 0:
            super(MultiNodeStorageSupplier, self).__exit__(exc_type, exc_val, exc_tb)


@pytest.mark.parametrize('storage_mode', STORAGE_MODES)
def test_minimize_chainermn(storage_mode):
    # type: (str) -> None

    if not _available:
        pytest.skip('This test requires ChainerMN.')

    comm = chainermn.create_communicator('naive')
    if comm.size < 2:
        pytest.skip("This test is for multi-node only.")

    with MultiNodeStorageSupplier(storage_mode, comm) as storage:
        # Create and broadcast study_name.
        name_local = create_study(storage).study_name if comm.rank == 0 else None
        name_bcast = comm.mpi_comm.bcast(name_local)
        study = Study(name_bcast, storage)

        # Invoke minimize_chainermn.
        n_trials = 20
        func = Func()
        study = minimize_chainermn(func, study, comm, n_trials=n_trials)

        # Assert trial counts.
        assert len(study.trials) == n_trials

        # Assert the same parameters have been suggested among all nodes.
        for trial in study.trials:
            assert trial.params == func.suggested_values[trial.trial_id]

        # Explicitly call storage's __del__ before sqlite tempfile is deleted.
        del storage
        gc.collect()
        comm.mpi_comm.barrier()
