import gc
from types import TracebackType
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Type

import pytest

from optuna import create_study
from optuna import distributions
from optuna import integration
from optuna import pruners
from optuna import Study
from optuna import TrialPruned
from optuna.integration.chainermn import ChainerMNStudy
from optuna.integration.chainermn import ChainerMNTrial
from optuna.pruners import BasePruner
from optuna.storages import BaseStorage
from optuna.storages import InMemoryStorage
from optuna.storages import RDBStorage
from optuna.testing.pruners import DeterministicPruner
from optuna.testing.storages import StorageSupplier
from optuna.trial import TrialState


try:
    import chainermn
    from chainermn.communicators.communicator_base import CommunicatorBase  # NOQA

    _available = True
except ImportError:
    _available = False

STORAGE_MODES = ["sqlite"]
PRUNER_INIT_FUNCS = [lambda: pruners.MedianPruner(), lambda: pruners.SuccessiveHalvingPruner()]

pytestmark = pytest.mark.integration


class Func:
    def __init__(self) -> None:
        self.suggested_values: Dict[int, Dict[str, Any]] = {}

    def __call__(self, trial: ChainerMNTrial, comm: "CommunicatorBase") -> float:
        x = trial.suggest_float("x", -10, 10)
        y = trial.suggest_float("y", 20, 30, log=True)
        z = trial.suggest_categorical("z", (-1.0, 1.0))

        self.suggested_values[trial.number] = {}
        self.suggested_values[trial.number]["x"] = x
        self.suggested_values[trial.number]["y"] = y
        self.suggested_values[trial.number]["z"] = z

        return (x - 2) ** 2 + (y - 25) ** 2 + z


class MultiNodeStorageSupplier(StorageSupplier):
    def __init__(self, storage_specifier: str, comm: "CommunicatorBase") -> None:
        super().__init__(storage_specifier)
        self.comm = comm
        self.storage: Optional[RDBStorage] = None

    def __enter__(self) -> RDBStorage:
        if self.comm.rank == 0:
            storage = super(MultiNodeStorageSupplier, self).__enter__()
            assert isinstance(storage, RDBStorage)
            url = str(storage.engine.url)
        else:
            url = "dummy_url"

        url = self.comm.mpi_comm.bcast(url)
        self.storage = RDBStorage(url)
        return self.storage

    def __exit__(
        self, exc_type: Type[BaseException], exc_val: BaseException, exc_tb: TracebackType
    ) -> None:
        # Explicitly call storage's __del__ before sqlite tempfile is deleted.
        del self.storage
        gc.collect()
        self.comm.mpi_comm.barrier()

        if self.comm.rank == 0:
            super(MultiNodeStorageSupplier, self).__exit__(exc_type, exc_val, exc_tb)


@pytest.fixture
def comm() -> "CommunicatorBase":
    if not _available:
        pytest.skip("This test requires ChainerMN.")

    return chainermn.create_communicator("naive")


class TestChainerMNStudy:
    @staticmethod
    @pytest.mark.parametrize("storage_mode", STORAGE_MODES)
    def test_init(storage_mode: str, comm: "CommunicatorBase") -> None:
        with MultiNodeStorageSupplier(storage_mode, comm) as storage:
            study = TestChainerMNStudy._create_shared_study(storage, comm)
            mn_study = ChainerMNStudy(study, comm)

            assert mn_study.study_name == study.study_name

    @staticmethod
    @pytest.mark.parametrize("storage_mode", STORAGE_MODES)
    def test_init_with_multiple_study_names(storage_mode: str, comm: "CommunicatorBase") -> None:
        TestChainerMNStudy._check_multi_node(comm)

        with MultiNodeStorageSupplier(storage_mode, comm) as storage:
            # Create study_name for each rank.
            name = create_study(storage=storage).study_name
            study = Study(name, storage)

            with pytest.raises(ValueError):
                ChainerMNStudy(study, comm)

    @staticmethod
    def test_init_with_incompatible_storage(comm: "CommunicatorBase") -> None:
        study = create_study(storage=InMemoryStorage(), study_name="in-memory-study")

        with pytest.raises(ValueError):
            ChainerMNStudy(study, comm)

    @staticmethod
    @pytest.mark.parametrize("storage_mode", STORAGE_MODES)
    def test_optimize(storage_mode: str, comm: "CommunicatorBase") -> None:
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
    def test_pruning(
        storage_mode: str, pruner_init_func: Callable[[], BasePruner], comm: "CommunicatorBase"
    ) -> None:
        with MultiNodeStorageSupplier(storage_mode, comm) as storage:
            pruner = pruner_init_func()
            study = TestChainerMNStudy._create_shared_study(storage, comm, pruner=pruner)
            mn_study = ChainerMNStudy(study, comm)

            def objective(_trial: ChainerMNTrial, _comm: bool) -> float:
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
    def test_failure(storage_mode: str, comm: "CommunicatorBase") -> None:
        with MultiNodeStorageSupplier(storage_mode, comm) as storage:
            study = TestChainerMNStudy._create_shared_study(storage, comm)
            mn_study = ChainerMNStudy(study, comm)

            def objective(_trial: ChainerMNTrial, _comm: bool) -> float:
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
    def _create_shared_study(
        storage: BaseStorage,
        comm: "CommunicatorBase",
        pruner: Optional[BasePruner] = None,
    ) -> Study:
        name_local = create_study(storage=storage).study_name if comm.rank == 0 else None
        name_bcast = comm.mpi_comm.bcast(name_local)

        return Study(name_bcast, storage, pruner=pruner)

    @staticmethod
    def _check_multi_node(comm: "CommunicatorBase") -> None:
        if comm.size < 2:
            pytest.skip("This test is for multi-node only.")


class TestChainerMNTrial:
    @staticmethod
    @pytest.mark.parametrize("storage_mode", STORAGE_MODES)
    def test_init(storage_mode: str, comm: "CommunicatorBase") -> None:
        with MultiNodeStorageSupplier(storage_mode, comm) as storage:
            study = TestChainerMNStudy._create_shared_study(storage, comm)
            mn_trial = _create_new_chainermn_trial(study, comm)
            trial = study.trials[-1]

            assert mn_trial.number == trial.number

    @staticmethod
    @pytest.mark.parametrize("storage_mode", STORAGE_MODES)
    def test_suggest_float(storage_mode: str, comm: "CommunicatorBase") -> None:
        with MultiNodeStorageSupplier(storage_mode, comm) as storage:
            study = TestChainerMNStudy._create_shared_study(storage, comm)
            low1 = 0.5
            high1 = 1.0
            for _ in range(10):
                mn_trial = _create_new_chainermn_trial(study, comm)

                x1 = mn_trial.suggest_float("x1", low1, high1)
                assert low1 <= x1 <= high1

                x2 = mn_trial.suggest_float("x1", low1, high1)

                assert x1 == x2

                with pytest.raises(ValueError):
                    mn_trial.suggest_float("x1", low1, high1, log=True)

            low2 = 1e-7
            high2 = 1e-2
            for _ in range(10):
                mn_trial = _create_new_chainermn_trial(study, comm)

                x3 = mn_trial.suggest_float("x2", low2, high2, log=True)
                assert low2 <= x3 <= high2

                x4 = mn_trial.suggest_float("x2", low2, high2, log=True)
                assert x3 == x4

                with pytest.raises(ValueError):
                    mn_trial.suggest_float("x2", low2, high2)

    @staticmethod
    @pytest.mark.parametrize("storage_mode", STORAGE_MODES)
    def test_suggest_float_with_step(storage_mode: str, comm: "CommunicatorBase") -> None:
        with MultiNodeStorageSupplier(storage_mode, comm) as storage:
            study = TestChainerMNStudy._create_shared_study(storage, comm)
            low = 0.0
            high = 10.0
            step = 1.0
            for _ in range(10):
                mn_trial = _create_new_chainermn_trial(study, comm)

                x1 = mn_trial.suggest_float("x", low, high, step=step)
                assert low <= x1 <= high

                x2 = mn_trial.suggest_float("x", low, high, step=step)
                assert x1 == x2

                if comm.rank == 0:
                    with pytest.warns(RuntimeWarning):
                        mn_trial.suggest_float("x", low, high)
                else:
                    mn_trial.suggest_float("x", low, high)

    @staticmethod
    @pytest.mark.parametrize("storage_mode", STORAGE_MODES)
    @pytest.mark.parametrize("enable_log", [False, True])
    def test_suggest_int_step1(
        storage_mode: str, comm: "CommunicatorBase", enable_log: bool
    ) -> None:
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
                    mn_trial.suggest_float("x", low, high)

    @staticmethod
    @pytest.mark.parametrize("storage_mode", STORAGE_MODES)
    def test_suggest_int_step2(storage_mode: str, comm: "CommunicatorBase") -> None:
        with MultiNodeStorageSupplier(storage_mode, comm) as storage:
            study = TestChainerMNStudy._create_shared_study(storage, comm)
            low = 1
            high = 9
            step = 2
            for _ in range(10):
                mn_trial = _create_new_chainermn_trial(study, comm)

                x1 = mn_trial.suggest_int("x", low, high, step=step, log=False)
                assert low <= x1 <= high

                x2 = mn_trial.suggest_int("x", low, high, step=step, log=False)
                assert x1 == x2

                with pytest.raises(ValueError):
                    mn_trial.suggest_float("x", low, high)

    @staticmethod
    @pytest.mark.parametrize("storage_mode", STORAGE_MODES)
    def test_suggest_categorical(storage_mode: str, comm: "CommunicatorBase") -> None:
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
                    mn_trial.suggest_float("x", 0.0, 1.0)

    @staticmethod
    @pytest.mark.parametrize("storage_mode", STORAGE_MODES)
    @pytest.mark.parametrize("is_pruning", [True, False])
    def test_report_and_should_prune(
        storage_mode: str, comm: "CommunicatorBase", is_pruning: bool
    ) -> None:
        with MultiNodeStorageSupplier(storage_mode, comm) as storage:
            study = TestChainerMNStudy._create_shared_study(
                storage, comm, DeterministicPruner(is_pruning)
            )
            mn_trial = _create_new_chainermn_trial(study, comm)
            mn_trial.report(1.0, 0)
            assert mn_trial.should_prune() == is_pruning

    @staticmethod
    @pytest.mark.parametrize("storage_mode", STORAGE_MODES)
    def test_params(storage_mode: str, comm: "CommunicatorBase") -> None:
        with MultiNodeStorageSupplier(storage_mode, comm) as storage:
            study = TestChainerMNStudy._create_shared_study(storage, comm)
            mn_trial = _create_new_chainermn_trial(study, comm)

            x = mn_trial.suggest_categorical("x", [1])
            assert mn_trial.params["x"] == x

    @staticmethod
    @pytest.mark.parametrize("storage_mode", STORAGE_MODES)
    def test_distributions(storage_mode: str, comm: "CommunicatorBase") -> None:
        with MultiNodeStorageSupplier(storage_mode, comm) as storage:
            study = TestChainerMNStudy._create_shared_study(storage, comm)
            mn_trial = _create_new_chainermn_trial(study, comm)

            mn_trial.suggest_categorical("x", [1])
            assert mn_trial.distributions == {
                "x": distributions.CategoricalDistribution(choices=(1,))
            }

    @staticmethod
    @pytest.mark.parametrize("storage_mode", STORAGE_MODES)
    def test_user_attrs(storage_mode: str, comm: "CommunicatorBase") -> None:
        with MultiNodeStorageSupplier(storage_mode, comm) as storage:
            study = TestChainerMNStudy._create_shared_study(storage, comm)
            mn_trial = _create_new_chainermn_trial(study, comm)

            mn_trial.set_user_attr("data", "MNIST")
            assert mn_trial.user_attrs["data"] == "MNIST"

    @staticmethod
    @pytest.mark.parametrize("storage_mode", STORAGE_MODES)
    def test_call_with_mpi(storage_mode: str, comm: "CommunicatorBase") -> None:
        with MultiNodeStorageSupplier(storage_mode, comm) as storage:
            study = TestChainerMNStudy._create_shared_study(storage, comm)
            mn_trial = _create_new_chainermn_trial(study, comm)
            with pytest.raises(RuntimeError):

                def func() -> None:
                    raise RuntimeError

                mn_trial._call_with_mpi(func)

    @staticmethod
    @pytest.mark.parametrize("storage_mode", STORAGE_MODES)
    def test_datetime_start(storage_mode: str, comm: "CommunicatorBase") -> None:
        with MultiNodeStorageSupplier(storage_mode, comm) as storage:
            study = TestChainerMNStudy._create_shared_study(storage, comm)
            mn_trial = _create_new_chainermn_trial(study, comm)

            assert mn_trial.datetime_start is not None


def _create_new_chainermn_trial(
    study: Study, comm: "CommunicatorBase"
) -> integration.chainermn.ChainerMNTrial:
    if comm.rank == 0:
        trial = study.ask()
        mn_trial = integration.chainermn.ChainerMNTrial(trial, comm)
    else:
        mn_trial = integration.chainermn.ChainerMNTrial(None, comm)

    comm.mpi_comm.barrier()
    return mn_trial
