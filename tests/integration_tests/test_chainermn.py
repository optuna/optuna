import gc

import pytest

from optuna import create_study
from optuna import distributions
from optuna import integration
from optuna.integration import ChainerMNStudy
from optuna import pruners
from optuna.storages import InMemoryStorage
from optuna.storages import RDBStorage
from optuna import Study
from optuna.testing.integration import DeterministicPruner
from optuna.testing.sampler import DeterministicRelativeSampler
from optuna.testing.storage import StorageSupplier
from optuna.trial import Trial
from optuna.trial import TrialState
from optuna import TrialPruned
from optuna import type_checking

if type_checking.TYPE_CHECKING:
    from types import TracebackType  # NOQA
    from typing import Any  # NOQA
    from typing import Callable  # NOQA
    from typing import Dict  # NOQA
    from typing import Optional  # NOQA
    from typing import Type  # NOQA

    from optuna.integration.chainermn import ChainerMNTrial  # NOQA
    from optuna.pruners import BasePruner  # NOQA
    from optuna.samplers import BaseSampler  # NOQA
    from optuna.storages import BaseStorage  # NOQA

try:
    import chainermn
    from chainermn.communicators.communicator_base import CommunicatorBase  # NOQA

    _available = True
except ImportError:
    _available = False

STORAGE_MODES = ["sqlite"]
PRUNER_INIT_FUNCS = [lambda: pruners.MedianPruner(), lambda: pruners.SuccessiveHalvingPruner()]


class Func(object):
    def __init__(self):
        # type: () -> None

        self.suggested_values = {}  # type: Dict[int, Dict[str, Any]]

    def __call__(self, trial, comm):
        # type: (ChainerMNTrial, CommunicatorBase) -> float

        x = trial.suggest_uniform("x", -10, 10)
        y = trial.suggest_loguniform("y", 20, 30)
        z = trial.suggest_categorical("z", (-1.0, 1.0))

        self.suggested_values[trial.number] = {}
        self.suggested_values[trial.number]["x"] = x
        self.suggested_values[trial.number]["y"] = y
        self.suggested_values[trial.number]["z"] = z

        return (x - 2) ** 2 + (y - 25) ** 2 + z


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
            url = "dummy_url"

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
        pytest.skip("This test requires ChainerMN.")

    return chainermn.create_communicator("naive")


class TestChainerMNStudy(object):
    @staticmethod
    @pytest.mark.parametrize("storage_mode", STORAGE_MODES)
    def test_init(storage_mode, comm):
        # type: (str, CommunicatorBase) -> None

        with MultiNodeStorageSupplier(storage_mode, comm) as storage:
            study = TestChainerMNStudy._create_shared_study(storage, comm)
            mn_study = ChainerMNStudy(study, comm)

            assert mn_study.study_name == study.study_name

    @staticmethod
    @pytest.mark.parametrize("storage_mode", STORAGE_MODES)
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

        study = create_study(InMemoryStorage(), study_name="in-memory-study")

        with pytest.raises(ValueError):
            ChainerMNStudy(study, comm)

    @staticmethod
    @pytest.mark.parametrize("storage_mode", STORAGE_MODES)
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
                assert trial.params == func.suggested_values[trial.number]

    @staticmethod
    @pytest.mark.parametrize("storage_mode", STORAGE_MODES)
    @pytest.mark.parametrize("pruner_init_func", PRUNER_INIT_FUNCS)
    def test_pruning(storage_mode, pruner_init_func, comm):
        # type: (str, Callable[[], BasePruner], CommunicatorBase) -> None

        with MultiNodeStorageSupplier(storage_mode, comm) as storage:
            pruner = pruner_init_func()
            study = TestChainerMNStudy._create_shared_study(storage, comm, pruner=pruner)
            mn_study = ChainerMNStudy(study, comm)

            def objective(_trial, _comm):
                # type: (ChainerMNTrial, bool) -> float

                raise TrialPruned  # Always be pruned.

            # Invoke optimize.
            n_trials = 20
            mn_study.optimize(objective, n_trials=n_trials)

            # Assert trial count.
            assert len(mn_study.trials) == n_trials

            # Assert pruned trial count.
            pruned_trials = [t for t in mn_study.trials if t.state == TrialState.PRUNED]
            assert len(pruned_trials) == n_trials

    @staticmethod
    @pytest.mark.parametrize("storage_mode", STORAGE_MODES)
    def test_failure(storage_mode, comm):
        # type: (str, CommunicatorBase) -> None

        with MultiNodeStorageSupplier(storage_mode, comm) as storage:
            study = TestChainerMNStudy._create_shared_study(storage, comm)
            mn_study = ChainerMNStudy(study, comm)

            def objective(_trial, _comm):
                # type: (ChainerMNTrial, bool) -> float

                raise ValueError  # Always fails.

            # Invoke optimize in which `ValueError` is accepted.
            n_trials = 20
            mn_study.optimize(objective, n_trials=n_trials, catch=(ValueError,))

            # Assert trial count.
            assert len(mn_study.trials) == n_trials

            # Assert failed trial count.
            failed_trials = [t for t in mn_study.trials if t.state == TrialState.FAIL]
            assert len(failed_trials) == n_trials

            # Synchronize nodes before executing the next optimization.
            comm.mpi_comm.barrier()

            # Invoke optimize in which no exceptions are accepted.
            with pytest.raises(ValueError):
                mn_study.optimize(objective, n_trials=n_trials, catch=())

            # Assert trial count.
            assert len(mn_study.trials) == n_trials + 1

            # Assert failed trial count.
            failed_trials = [t for t in mn_study.trials if t.state == TrialState.FAIL]
            assert len(failed_trials) == n_trials + 1

    @staticmethod
    @pytest.mark.parametrize("storage_mode", STORAGE_MODES)
    def test_relative_sampling(storage_mode, comm):
        # type: (str, CommunicatorBase) -> None

        relative_search_space = {
            "x": distributions.UniformDistribution(low=-10, high=10),
            "y": distributions.LogUniformDistribution(low=20, high=30),
            "z": distributions.CategoricalDistribution(choices=(-1.0, 1.0)),
        }
        relative_params = {"x": 1.0, "y": 25.0, "z": -1.0}
        sampler = DeterministicRelativeSampler(
            relative_search_space, relative_params  # type: ignore
        )

        with MultiNodeStorageSupplier(storage_mode, comm) as storage:
            study = TestChainerMNStudy._create_shared_study(storage, comm, sampler=sampler)
            mn_study = ChainerMNStudy(study, comm)

            # Invoke optimize.
            n_trials = 20
            func = Func()
            mn_study.optimize(func, n_trials=n_trials)

            # Assert trial counts.
            assert len(mn_study.trials) == n_trials

            # Assert the parameters in `relative_params` have been suggested among all nodes.
            for trial in mn_study.trials:
                assert trial.params == relative_params

    @staticmethod
    def _create_shared_study(storage, comm, pruner=None, sampler=None):
        # type: (BaseStorage, CommunicatorBase, BasePruner, BaseSampler) -> Study

        name_local = create_study(storage).study_name if comm.rank == 0 else None
        name_bcast = comm.mpi_comm.bcast(name_local)

        return Study(name_bcast, storage, pruner=pruner, sampler=sampler)

    @staticmethod
    def _check_multi_node(comm):
        # type: (CommunicatorBase) -> None

        if comm.size < 2:
            pytest.skip("This test is for multi-node only.")


class TestChainerMNTrial(object):
    @staticmethod
    @pytest.mark.parametrize("storage_mode", STORAGE_MODES)
    def test_init(storage_mode, comm):
        # type: (str, CommunicatorBase) -> None

        with MultiNodeStorageSupplier(storage_mode, comm) as storage:
            study = TestChainerMNStudy._create_shared_study(storage, comm)
            mn_trial = _create_new_chainermn_trial(study, comm)
            trial = study.trials[-1]

            assert mn_trial.number == trial.number

    @staticmethod
    @pytest.mark.parametrize("storage_mode", STORAGE_MODES)
    def test_suggest_float(storage_mode, comm):
        # type: (str, CommunicatorBase) -> None

        with MultiNodeStorageSupplier(storage_mode, comm) as storage:
            study = TestChainerMNStudy._create_shared_study(storage, comm)
            low1 = 0.5
            high1 = 1.0
            for _ in range(10):
                mn_trial = _create_new_chainermn_trial(study, comm)

                x1 = mn_trial.suggest_float("x1", low1, high1)
                assert low1 <= x1 <= high1

                x2 = mn_trial.suggest_uniform("x1", low1, high1)

                assert x1 == x2

                with pytest.raises(ValueError):
                    mn_trial.suggest_loguniform("x1", low1, high1)

            low2 = 1e-7
            high2 = 1e-2
            for _ in range(10):
                mn_trial = _create_new_chainermn_trial(study, comm)

                x3 = mn_trial.suggest_float("x2", low2, high2, log=True)
                assert low2 <= x3 <= high2

                x4 = mn_trial.suggest_loguniform("x2", low2, high2)

                assert x3 == x4

                with pytest.raises(ValueError):
                    mn_trial.suggest_uniform("x2", low2, high2)

    @staticmethod
    @pytest.mark.parametrize("storage_mode", STORAGE_MODES)
    def test_suggest_uniform(storage_mode, comm):
        # type: (str, CommunicatorBase) -> None

        with MultiNodeStorageSupplier(storage_mode, comm) as storage:
            study = TestChainerMNStudy._create_shared_study(storage, comm)
            low = 0.5
            high = 1.0
            for _ in range(10):
                mn_trial = _create_new_chainermn_trial(study, comm)

                x1 = mn_trial.suggest_uniform("x", low, high)
                assert low <= x1 <= high

                x2 = mn_trial.suggest_uniform("x", low, high)
                assert x1 == x2

                with pytest.raises(ValueError):
                    mn_trial.suggest_loguniform("x", low, high)

    @staticmethod
    @pytest.mark.parametrize("storage_mode", STORAGE_MODES)
    def test_suggest_loguniform(storage_mode, comm):
        # type: (str, CommunicatorBase) -> None

        with MultiNodeStorageSupplier(storage_mode, comm) as storage:
            study = TestChainerMNStudy._create_shared_study(storage, comm)
            low = 1e-7
            high = 1e-2
            for _ in range(10):
                mn_trial = _create_new_chainermn_trial(study, comm)

                x1 = mn_trial.suggest_loguniform("x", low, high)
                assert low <= x1 <= high

                x2 = mn_trial.suggest_loguniform("x", low, high)
                assert x1 == x2

                with pytest.raises(ValueError):
                    mn_trial.suggest_uniform("x", low, high)

    @staticmethod
    @pytest.mark.parametrize("storage_mode", STORAGE_MODES)
    def test_suggest_discrete_uniform(storage_mode, comm):
        # type: (str, CommunicatorBase) -> None

        with MultiNodeStorageSupplier(storage_mode, comm) as storage:
            study = TestChainerMNStudy._create_shared_study(storage, comm)
            low = 0.0
            high = 10.0
            q = 1.0
            for _ in range(10):
                mn_trial = _create_new_chainermn_trial(study, comm)

                x1 = mn_trial.suggest_discrete_uniform("x", low, high, q)
                assert low <= x1 <= high

                x2 = mn_trial.suggest_discrete_uniform("x", low, high, q)
                assert x1 == x2

                with pytest.raises(ValueError):
                    mn_trial.suggest_uniform("x", low, high)

    @staticmethod
    @pytest.mark.parametrize("storage_mode", STORAGE_MODES)
    @pytest.mark.parametrize("enable_log", [False, True])
    def test_suggest_int(storage_mode: str, comm: CommunicatorBase, enable_log: bool) -> None:

        with MultiNodeStorageSupplier(storage_mode, comm) as storage:
            study = TestChainerMNStudy._create_shared_study(storage, comm)
            low = 1
            high = 10
            step = 1
            for _ in range(10):
                mn_trial = _create_new_chainermn_trial(study, comm)

                x1 = mn_trial.suggest_int("x", low, high, step=step, log=enable_log)
                assert low <= x1 <= high

                x2 = mn_trial.suggest_int("x", low, high, step=step, log=enable_log)
                assert x1 == x2

                with pytest.raises(ValueError):
                    mn_trial.suggest_uniform("x", low, high)

            step = 2
            for _ in range(10):
                mn_trial = _create_new_chainermn_trial(study, comm)

                x1 = mn_trial.suggest_int("x", low, high, step=step, log=enable_log)
                assert low <= x1 <= high

                x2 = mn_trial.suggest_int("x", low, high, step=step, log=enable_log)
                assert x1 == x2

                with pytest.raises(ValueError):
                    mn_trial.suggest_uniform("x", low, high)

    @staticmethod
    @pytest.mark.parametrize("storage_mode", STORAGE_MODES)
    def test_suggest_categorical(storage_mode, comm):
        # type: (str, CommunicatorBase) -> None

        with MultiNodeStorageSupplier(storage_mode, comm) as storage:
            study = TestChainerMNStudy._create_shared_study(storage, comm)
            choices = ("a", "b", "c")
            for _ in range(10):
                mn_trial = _create_new_chainermn_trial(study, comm)

                x1 = mn_trial.suggest_categorical("x", choices)
                assert x1 in choices

                x2 = mn_trial.suggest_categorical("x", choices)
                assert x1 == x2

                with pytest.raises(ValueError):
                    mn_trial.suggest_uniform("x", 0.0, 1.0)

    @staticmethod
    @pytest.mark.parametrize("storage_mode", STORAGE_MODES)
    @pytest.mark.parametrize("is_pruning", [True, False])
    def test_report_and_should_prune(storage_mode, comm, is_pruning):
        # type: (str, CommunicatorBase, bool) -> None

        with MultiNodeStorageSupplier(storage_mode, comm) as storage:
            study = TestChainerMNStudy._create_shared_study(
                storage, comm, DeterministicPruner(is_pruning)
            )
            mn_trial = _create_new_chainermn_trial(study, comm)
            mn_trial.report(1.0, 0)
            assert mn_trial.should_prune() == is_pruning

    @staticmethod
    @pytest.mark.parametrize("storage_mode", STORAGE_MODES)
    def test_params(storage_mode, comm):
        # type: (str, CommunicatorBase) -> None

        with MultiNodeStorageSupplier(storage_mode, comm) as storage:
            study = TestChainerMNStudy._create_shared_study(storage, comm)
            mn_trial = _create_new_chainermn_trial(study, comm)

            x = mn_trial.suggest_categorical("x", [1])
            assert mn_trial.params["x"] == x

    @staticmethod
    @pytest.mark.parametrize("storage_mode", STORAGE_MODES)
    def test_distributions(storage_mode, comm):
        # type: (str, CommunicatorBase) -> None

        with MultiNodeStorageSupplier(storage_mode, comm) as storage:
            study = TestChainerMNStudy._create_shared_study(storage, comm)
            mn_trial = _create_new_chainermn_trial(study, comm)

            mn_trial.suggest_categorical("x", [1])
            assert mn_trial.distributions == {
                "x": distributions.CategoricalDistribution(choices=(1,))
            }

    @staticmethod
    @pytest.mark.parametrize("storage_mode", STORAGE_MODES)
    def test_user_attrs(storage_mode, comm):
        # type: (str, CommunicatorBase) -> None

        with MultiNodeStorageSupplier(storage_mode, comm) as storage:
            study = TestChainerMNStudy._create_shared_study(storage, comm)
            mn_trial = _create_new_chainermn_trial(study, comm)

            mn_trial.set_user_attr("data", "MNIST")
            assert mn_trial.user_attrs["data"] == "MNIST"

    @staticmethod
    @pytest.mark.parametrize("storage_mode", STORAGE_MODES)
    def test_system_attrs(storage_mode, comm):
        # type: (str, CommunicatorBase) -> None

        with MultiNodeStorageSupplier(storage_mode, comm) as storage:
            study = TestChainerMNStudy._create_shared_study(storage, comm)
            mn_trial = _create_new_chainermn_trial(study, comm)

            mn_trial.set_system_attr("system_message", "test")
            assert mn_trial.system_attrs["system_message"] == "test"

    @staticmethod
    @pytest.mark.parametrize("storage_mode", STORAGE_MODES)
    def test_call_with_mpi(storage_mode, comm):
        # type: (str, CommunicatorBase) -> None

        with MultiNodeStorageSupplier(storage_mode, comm) as storage:
            study = TestChainerMNStudy._create_shared_study(storage, comm)
            mn_trial = _create_new_chainermn_trial(study, comm)
            with pytest.raises(RuntimeError):

                def func():
                    # type: () -> None

                    raise RuntimeError

                mn_trial._call_with_mpi(func)

    @staticmethod
    @pytest.mark.parametrize("storage_mode", STORAGE_MODES)
    def test_datetime_start(storage_mode, comm):
        # type: (str, CommunicatorBase) -> None

        with MultiNodeStorageSupplier(storage_mode, comm) as storage:
            study = TestChainerMNStudy._create_shared_study(storage, comm)
            mn_trial = _create_new_chainermn_trial(study, comm)

            assert mn_trial.datetime_start is not None


def _create_new_chainermn_trial(study, comm):
    # type: (Study, CommunicatorBase) -> integration.chainermn.ChainerMNTrial

    if comm.rank == 0:
        trial_id = study._storage.create_new_trial(study._study_id)
        trial = Trial(study, trial_id)
        mn_trial = integration.chainermn.ChainerMNTrial(trial, comm)
    else:
        mn_trial = integration.chainermn.ChainerMNTrial(None, comm)

    comm.mpi_comm.barrier()
    return mn_trial
