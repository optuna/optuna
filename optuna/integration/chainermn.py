import gc
from typing import Optional
import warnings

from optuna.logging import get_logger
from optuna.storages import InMemoryStorage
from optuna.storages import RDBStorage
from optuna.trial import BaseTrial
from optuna import TrialPruned
from optuna import type_checking

if type_checking.TYPE_CHECKING:
    from datetime import datetime  # NOQA
    from typing import Any  # NOQA
    from typing import Callable  # NOQA
    from typing import Dict  # NOQA
    from typing import Sequence  # NOQA
    from typing import Tuple  # NOQA
    from typing import Type  # NOQA
    from typing import Union  # NOQA

    from optuna.distributions import BaseDistribution  # NOQA
    from optuna.distributions import CategoricalChoiceType  # NOQA
    from optuna.study import Study  # NOQA
    from optuna.trial import Trial  # NOQA

try:
    from chainermn.communicators.communicator_base import CommunicatorBase  # NOQA

    _available = True
except ImportError as e:
    _import_error = e
    _available = False


class _ChainerMNObjectiveFunc(object):
    """A wrapper of an objective function to incorporate Optuna with ChainerMN.

    Note that this class is not supposed to be used by library users.

    Args:
        func:
            A callable that implements objective function.
        comm:
            A `ChainerMN communicator <https://docs.chainer.org/en/stable/chainermn/reference/
            index.html#communicators>`_.
    """

    def __init__(self, func, comm):
        # type: (Callable[[ChainerMNTrial, CommunicatorBase], float], CommunicatorBase) -> None

        self.comm = comm
        self.objective = func

    def __call__(self, trial):
        # type: (Trial) -> float

        self.comm.mpi_comm.bcast(True)
        return self.objective(ChainerMNTrial(trial, self.comm), self.comm)


class ChainerMNStudy(object):
    """A wrapper of :class:`~optuna.study.Study` to incorporate Optuna with ChainerMN.

    .. seealso::
        :class:`~optuna.integration.chainermn.ChainerMNStudy` provides the same interface as
        :class:`~optuna.study.Study`. Please refer to :class:`optuna.study.Study` for further
        details.

    See `the example <https://github.com/optuna/optuna/blob/master/
    examples/pruning/chainermn_integration.py>`__
    if you want to optimize an objective function that trains neural network
    written with ChainerMN.

    Args:
        study:
            A :class:`~optuna.study.Study` object.
        comm:
            A `ChainerMN communicator <https://docs.chainer.org/en/stable/chainermn/reference/
            index.html#communicators>`_.
    """

    def __init__(
        self,
        study,  # type: Study
        comm,  # type: CommunicatorBase
    ):
        # type: (...) -> None

        _check_chainermn_availability()

        if isinstance(study._storage, InMemoryStorage):
            raise ValueError("ChainerMN integration is not available with InMemoryStorage.")

        if isinstance(study._storage, RDBStorage):
            if study._storage.engine.dialect.name == "sqlite":
                logger = get_logger(__name__)
                logger.warning(
                    "SQLite may cause synchronization problems when used with "
                    "ChainerMN integration. Please use other DBs like PostgreSQL."
                )

        study_names = comm.mpi_comm.allgather(study.study_name)
        if len(set(study_names)) != 1:
            raise ValueError("Please make sure an identical study name is shared among workers.")

        super(ChainerMNStudy, self).__setattr__("delegate", study)
        super(ChainerMNStudy, self).__setattr__("comm", comm)

    def optimize(
        self,
        func,  # type: Callable[[ChainerMNTrial, CommunicatorBase], float]
        n_trials=None,  # type: Optional[int]
        timeout=None,  # type: Optional[float]
        catch=(),  # type: Union[Tuple[()], Tuple[Type[Exception]]]
    ):
        # type: (...) -> None
        """Optimize an objective function.

        This method provides the same interface as :func:`optuna.study.Study.optimize` except
        the absence of ``n_jobs`` argument.
        """

        if self.comm.rank == 0:
            func_mn = _ChainerMNObjectiveFunc(func, self.comm)
            self.delegate.optimize(func_mn, n_trials=n_trials, timeout=timeout, catch=catch)
            self.comm.mpi_comm.bcast(False)
        else:
            while True:
                has_next_trial = self.comm.mpi_comm.bcast(None)
                if not has_next_trial:
                    break
                try:
                    func(ChainerMNTrial(None, self.comm), self.comm)

                    # We assume that if a node raises an exception,
                    # all other nodes will do the same.
                    #
                    # The responsibility to handle acceptable exceptions (i.e., `TrialPruned` and
                    # `catch`) is in the rank-0 node, so other nodes simply ignore them.
                except TrialPruned:
                    pass
                except catch:
                    pass
                finally:
                    # The following line mitigates memory problems that can be occurred in some
                    # environments (e.g., services that use computing containers such as CircleCI).
                    # Please refer to the following PR for further details:
                    # https://github.com/optuna/optuna/pull/325.
                    gc.collect()

    def __getattr__(self, attr_name):
        # type: (str) -> Any

        return getattr(self.delegate, attr_name)

    def __setattr__(self, attr_name, value):
        # type: (str, Any) -> None

        setattr(self.delegate, attr_name, value)


class ChainerMNTrial(BaseTrial):
    """A wrapper of :class:`~optuna.trial.Trial` to incorporate Optuna with ChainerMN.

    .. seealso::
        :class:`~optuna.integration.chainermn.ChainerMNTrial` provides the same interface as
        :class:`~optuna.trial.Trial`. Please refer to :class:`optuna.trial.Trial` for further
        details.

    Args:
        trial:
            A :class:`~optuna.trial.Trial` object if the caller is rank0 worker,
            :obj:`None` otherwise.
        comm:
            A `ChainerMN communicator <https://docs.chainer.org/en/stable/chainermn/reference/
            index.html#communicators>`_.
    """

    def __init__(self, trial, comm):
        # type: (Optional[Trial], CommunicatorBase) -> None

        self.delegate = trial
        self.comm = comm

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        step: Optional[float] = None,
        log: bool = False
    ) -> float:
        def func() -> float:
            assert self.delegate is not None
            return self.delegate.suggest_float(name, low, high, log=log, step=step)

        return self._call_with_mpi(func)

    def suggest_uniform(self, name, low, high):
        # type: (str, float, float) -> float

        def func():
            # type: () -> float

            assert self.delegate is not None
            return self.delegate.suggest_uniform(name, low, high)

        return self._call_with_mpi(func)

    def suggest_loguniform(self, name, low, high):
        # type: (str, float, float) -> float

        def func():
            # type: () -> float

            assert self.delegate is not None
            return self.delegate.suggest_loguniform(name, low, high)

        return self._call_with_mpi(func)

    def suggest_discrete_uniform(self, name, low, high, q):
        # type: (str, float, float, float) -> float

        def func():
            # type: () -> float

            assert self.delegate is not None
            return self.delegate.suggest_discrete_uniform(name, low, high, q)

        return self._call_with_mpi(func)

    def suggest_int(self, name, low, high, step=1, log=False):
        # type: (str, int, int, int, bool) -> int

        def func():
            # type: () -> int

            assert self.delegate is not None
            return self.delegate.suggest_int(name, low, high, step=step, log=log)

        return self._call_with_mpi(func)

    def suggest_categorical(self, name, choices):
        # type: (str, Sequence[CategoricalChoiceType]) -> Any

        def func():
            # type: () -> CategoricalChoiceType

            assert self.delegate is not None
            return self.delegate.suggest_categorical(name, choices)

        return self._call_with_mpi(func)

    def report(self, value, step):
        # type: (float, int) -> None

        if self.comm.rank == 0:
            assert self.delegate is not None
            self.delegate.report(value, step)
        self.comm.mpi_comm.barrier()

    def should_prune(self, step=None):
        # type: (Optional[int]) -> bool

        def func():
            # type: () -> bool

            assert self.delegate is not None
            return self.delegate.should_prune(step)

        return self._call_with_mpi(func)

    def set_user_attr(self, key, value):
        # type: (str, Any) -> None

        if self.comm.rank == 0:
            assert self.delegate is not None
            self.delegate.set_user_attr(key, value)
        self.comm.mpi_comm.barrier()

    def set_system_attr(self, key, value):
        # type: (str, Any) -> None

        if self.comm.rank == 0:
            assert self.delegate is not None
            self.delegate.set_system_attr(key, value)
        self.comm.mpi_comm.barrier()

    @property
    def number(self):
        # type: () -> int

        def func():
            # type: () -> int

            assert self.delegate is not None
            return self.delegate.number

        return self._call_with_mpi(func)

    @property
    def trial_id(self):
        # type: () -> int

        warnings.warn(
            "The use of `ChainerMNTrial.trial_id` is deprecated. "
            "Please use `ChainerMNTrial.number` instead.",
            DeprecationWarning,
        )
        return self._trial_id

    @property
    def _trial_id(self):
        # type: () -> int

        def func():
            # type: () -> int

            assert self.delegate is not None
            return self.delegate._trial_id

        return self._call_with_mpi(func)

    @property
    def params(self):
        # type: () -> Dict[str, Any]

        def func():
            # type: () -> Dict[str, Any]

            assert self.delegate is not None
            return self.delegate.params

        return self._call_with_mpi(func)

    @property
    def distributions(self):
        # type: () -> Dict[str, BaseDistribution]

        def func():
            # type: () -> Dict[str, BaseDistribution]

            assert self.delegate is not None
            return self.delegate.distributions

        return self._call_with_mpi(func)

    @property
    def user_attrs(self):
        # type: () -> Dict[str, Any]

        def func():
            # type: () -> Dict[str, Any]

            assert self.delegate is not None
            return self.delegate.user_attrs

        return self._call_with_mpi(func)

    @property
    def system_attrs(self):
        # type: () -> Dict[str, Any]

        def func():
            # type: () -> Dict[str, Any]

            assert self.delegate is not None
            return self.delegate.system_attrs

        return self._call_with_mpi(func)

    @property
    def datetime_start(self):
        # type: () -> Optional[datetime]

        def func():
            # type: () -> Optional[datetime]

            assert self.delegate is not None
            return self.delegate.datetime_start

        return self._call_with_mpi(func)

    def _call_with_mpi(self, func):
        # type: (Callable) -> Any

        if self.comm.rank == 0:
            try:
                result = func()
                self.comm.mpi_comm.bcast(result)
                return result
            except Exception as e:
                self.comm.mpi_comm.bcast(e)
                raise
        else:
            result = self.comm.mpi_comm.bcast(None)
            if isinstance(result, Exception):
                raise result
            return result


def _check_chainermn_availability():
    # type: () -> None

    if not _available:
        raise ImportError(
            "ChainerMN is not available. Please install ChainerMN to use this feature. "
            "ChainerMN can be installed by executing `$ pip install chainermn`. "
            "For further information, please refer to the installation guide of ChainerMN. "
            "(The actual import error is as follows: " + str(_import_error) + ")"
        )
