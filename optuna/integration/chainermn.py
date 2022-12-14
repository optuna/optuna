from datetime import datetime
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
import warnings

from optuna import TrialPruned
from optuna._deprecated import deprecated_func
from optuna._imports import try_import
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalChoiceType
from optuna.storages import InMemoryStorage
from optuna.storages import RDBStorage
from optuna.study import Study
from optuna.trial import BaseTrial
from optuna.trial import Trial


with try_import() as _imports:
    from chainermn.communicators.communicator_base import CommunicatorBase  # NOQA


_suggest_deprecated_msg = "Use :func:`~optuna.integration.ChainerMNTrial.suggest_float` instead."


class _ChainerMNObjectiveFunc:
    """A wrapper of an objective function to incorporate Optuna with ChainerMN.

    Note that this class is not supposed to be used by library users.

    Args:
        func:
            A callable that implements objective function.
        comm:
            A `ChainerMN communicator <https://docs.chainer.org/en/stable/chainermn/reference/
            index.html#communicators>`_.
    """

    def __init__(
        self,
        func: Callable[["ChainerMNTrial", "CommunicatorBase"], float],
        comm: "CommunicatorBase",
    ) -> None:

        self.comm = comm
        self.objective = func

    def __call__(self, trial: Trial) -> float:

        self.comm.mpi_comm.bcast(True)
        return self.objective(ChainerMNTrial(trial, self.comm), self.comm)


class ChainerMNStudy:
    """A wrapper of :class:`~optuna.study.Study` to incorporate Optuna with ChainerMN.

    .. seealso::
        :class:`~optuna.integration.chainermn.ChainerMNStudy` provides the same interface as
        :class:`~optuna.study.Study`. Please refer to :class:`optuna.study.Study` for further
        details.

    See `the example <https://github.com/optuna/optuna-examples/blob/main/
    chainer/chainermn_integration.py>`__
    if you want to optimize an objective function that trains neural network
    written with ChainerMN.

    Args:
        study:
            A :class:`~optuna.study.Study` object.
        comm:
            A `ChainerMN communicator <https://docs.chainer.org/en/stable/chainermn/reference/
            index.html#communicators>`_.
    """

    def __init__(self, study: Study, comm: "CommunicatorBase") -> None:

        _imports.check()

        if isinstance(study._storage, InMemoryStorage):
            raise ValueError("ChainerMN integration is not available with InMemoryStorage.")

        if isinstance(study._storage, RDBStorage):
            if study._storage.engine.dialect.name == "sqlite":
                warnings.warn(
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
        func: Callable[["ChainerMNTrial", "CommunicatorBase"], float],
        n_trials: Optional[int] = None,
        timeout: Optional[float] = None,
        catch: Tuple[Type[Exception], ...] = (),
    ) -> None:
        """Optimize an objective function.

        This method provides the same interface as :func:`optuna.study.Study.optimize` except
        the absence of ``n_jobs`` argument.
        """

        if self.comm.rank == 0:
            func_mn = _ChainerMNObjectiveFunc(func, self.comm)
            try:
                self.delegate.optimize(func_mn, n_trials=n_trials, timeout=timeout, catch=catch)
            finally:
                self.comm.mpi_comm.bcast(False)
        else:
            has_next_trial = self.comm.mpi_comm.bcast(None)
            while True:
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
                    has_next_trial = self.comm.mpi_comm.bcast(None)

    def __getattr__(self, attr_name: str) -> Any:

        return getattr(self.delegate, attr_name)

    def __setattr__(self, attr_name: str, value: Any) -> None:

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

    def __init__(self, trial: Optional[Trial], comm: "CommunicatorBase") -> None:

        self.delegate = trial
        self.comm = comm

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        step: Optional[float] = None,
        log: bool = False,
    ) -> float:
        def func() -> float:
            assert self.delegate is not None
            return self.delegate.suggest_float(name, low, high, log=log, step=step)

        return self._call_with_mpi(func)

    @deprecated_func("3.0.0", "6.0.0", text=_suggest_deprecated_msg)
    def suggest_uniform(self, name: str, low: float, high: float) -> float:

        return self.suggest_float(name, low, high)

    @deprecated_func("3.0.0", "6.0.0", text=_suggest_deprecated_msg)
    def suggest_loguniform(self, name: str, low: float, high: float) -> float:

        return self.suggest_float(name, low, high, log=True)

    @deprecated_func("3.0.0", "6.0.0", text=_suggest_deprecated_msg)
    def suggest_discrete_uniform(self, name: str, low: float, high: float, q: float) -> float:

        return self.suggest_float(name, low, high, step=q)

    def suggest_int(self, name: str, low: int, high: int, step: int = 1, log: bool = False) -> int:
        def func() -> int:

            assert self.delegate is not None
            return self.delegate.suggest_int(name, low, high, step=step, log=log)

        return self._call_with_mpi(func)

    def suggest_categorical(self, name: str, choices: Sequence[CategoricalChoiceType]) -> Any:
        def func() -> CategoricalChoiceType:

            assert self.delegate is not None
            return self.delegate.suggest_categorical(name, choices)

        return self._call_with_mpi(func)

    def report(self, value: float, step: int) -> None:

        if self.comm.rank == 0:
            assert self.delegate is not None
            self.delegate.report(value, step)
        self.comm.mpi_comm.barrier()

    def should_prune(self) -> bool:
        def func() -> bool:

            assert self.delegate is not None
            return self.delegate.should_prune()

        return self._call_with_mpi(func)

    def set_user_attr(self, key: str, value: Any) -> None:

        if self.comm.rank == 0:
            assert self.delegate is not None
            self.delegate.set_user_attr(key, value)
        self.comm.mpi_comm.barrier()

    @deprecated_func("3.1.0", "6.0.0")
    def set_system_attr(self, key: str, value: Any) -> None:

        if self.comm.rank == 0:
            assert self.delegate is not None
            self.delegate.storage.set_trial_system_attr(self.delegate._trial_id, key, value)
        self.comm.mpi_comm.barrier()

    @property
    def number(self) -> int:
        def func() -> int:

            assert self.delegate is not None
            return self.delegate.number

        return self._call_with_mpi(func)

    @property
    def _trial_id(self) -> int:
        def func() -> int:

            assert self.delegate is not None
            return self.delegate._trial_id

        return self._call_with_mpi(func)

    @property
    def params(self) -> Dict[str, Any]:
        def func() -> Dict[str, Any]:

            assert self.delegate is not None
            return self.delegate.params

        return self._call_with_mpi(func)

    @property
    def distributions(self) -> Dict[str, BaseDistribution]:
        def func() -> Dict[str, BaseDistribution]:

            assert self.delegate is not None
            return self.delegate.distributions

        return self._call_with_mpi(func)

    @property
    def user_attrs(self) -> Dict[str, Any]:
        def func() -> Dict[str, Any]:

            assert self.delegate is not None
            return self.delegate.user_attrs

        return self._call_with_mpi(func)

    @property
    def system_attrs(self) -> Dict[str, Any]:
        def func() -> Dict[str, Any]:

            assert self.delegate is not None
            return self.delegate.system_attrs

        return self._call_with_mpi(func)

    @property
    def datetime_start(self) -> Optional[datetime]:
        def func() -> Optional[datetime]:

            assert self.delegate is not None
            return self.delegate.datetime_start

        return self._call_with_mpi(func)

    def _call_with_mpi(self, func: Callable) -> Any:

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
