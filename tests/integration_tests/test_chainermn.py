import gc
import pytest
from types import TracebackType  # NOQA
from typing import Any  # NOQA
from typing import Dict  # NOQA
from typing import Optional  # NOQA
from typing import Type  # NOQA

from optuna import create_study
from optuna.integration import ChainerMNStudy
from optuna.storages import BaseStorage  # NOQA
from optuna.storages import InMemoryStorage
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

        return (x - 2)**2 + (y - 25)**2 + z


class MultiNodeStorageSupplier(StorageSupplier):
    def __init__(self, storage_specifier, comm):
        # type: (str, CommunicatorBase) -> None

        super(MultiNodeStorageSupplier, self).__init__(storage_specifier)
        self.comm = comm
        self.storage = None  # type: Optional[RDBStorage]

    def __enter__(self):
        # type: () -> RDBStorage

        if self.comm.rank == 0:
            storage = super(MultiNodeStorageSupplier, self).__enter__()
            assert isinstance(storage, RDBStorage)
            url = str(storage.engine.url)
        else:
            url = 'dummy_url'

        url = self.comm.mpi_comm.bcast(url)
        self.storage = RDBStorage(url)
        return self.storage

    def __exit__(self, exc_type, exc_val, exc_tb):
        # type: (Type[BaseException], BaseException, TracebackType) -> None

        # Explicitly call storage's __del__ before sqlite tempfile is deleted.
        del self.storage
        gc.collect()
        self.comm.mpi_comm.barrier()

        if self.comm.rank == 0:
            super(MultiNodeStorageSupplier, self).__exit__(exc_type, exc_val, exc_tb)


@pytest.fixture
def comm():
    # type: () -> CommunicatorBase

    if not _available:
        pytest.skip('This test requires ChainerMN.')

    return chainermn.create_communicator('naive')


class TestChainerMNStudy(object):
    @staticmethod
    @pytest.mark.parametrize('storage_mode', STORAGE_MODES)
    def test_init(storage_mode, comm):
        # type: (str, CommunicatorBase) -> None

        with MultiNodeStorageSupplier(storage_mode, comm) as storage:
            study = TestChainerMNStudy._create_shared_study(storage, comm)
            mn_study = ChainerMNStudy(study, comm)

            assert mn_study.study_name == study.study_name

    @staticmethod
    @pytest.mark.parametrize('storage_mode', STORAGE_MODES)
    def test_init_with_multiple_study_names(storage_mode, comm):
        # type: (str, CommunicatorBase) -> None

        TestChainerMNStudy._check_multi_node(comm)

        with MultiNodeStorageSupplier(storage_mode, comm) as storage:
            # Create study_name for each rank.
            name = create_study(storage).study_name
            study = Study(name, storage)

            with pytest.raises(ValueError):
                ChainerMNStudy(study, comm)

    @staticmethod
    def test_init_with_incompatible_storage(comm):
        # type: (CommunicatorBase) -> None

        study = TestChainerMNStudy._create_shared_study(InMemoryStorage(), comm)

        with pytest.raises(ValueError):
            ChainerMNStudy(study, comm)

    @staticmethod
    @pytest.mark.parametrize('storage_mode', STORAGE_MODES)
    def test_optimize(storage_mode, comm):
        # type: (str, CommunicatorBase) -> None

        with MultiNodeStorageSupplier(storage_mode, comm) as storage:
            study = TestChainerMNStudy._create_shared_study(storage, comm)
            mn_study = ChainerMNStudy(study, comm)

            # Invoke optimize.
            n_trials = 20
            func = Func()
            mn_study.optimize(func, n_trials=n_trials)

            # Assert trial counts.
            assert len(mn_study.trials) == n_trials

            # Assert the same parameters have been suggested among all nodes.
            for trial in mn_study.trials:
                assert trial.params == func.suggested_values[trial.trial_id]

    @staticmethod
    def _create_shared_study(storage, comm):
        # type: (BaseStorage, CommunicatorBase) -> Study

        name_local = create_study(storage).study_name if comm.rank == 0 else None
        name_bcast = comm.mpi_comm.bcast(name_local)

        return Study(name_bcast, storage)

    @staticmethod
    def _check_multi_node(comm):
        # type: (CommunicatorBase) -> None

        if comm.size < 2:
            pytest.skip('This test is for multi-node only.')
