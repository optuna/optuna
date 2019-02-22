from __future__ import absolute_import

from typing import Any  # NOQA
from typing import Callable  # NOQA
from typing import Optional  # NOQA
from typing import Tuple  # NOQA
from typing import Type  # NOQA
from typing import Union  # NOQA

from optuna.logging import get_logger
from optuna.pruners import BasePruner  # NOQA
from optuna.storages import BaseStorage  # NOQA
from optuna.storages import InMemoryStorage
from optuna.storages import RDBStorage
from optuna.structs import TrialPruned
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
        # type: (Callable[[Trial, CommunicatorBase], float], CommunicatorBase) -> None

        self.comm = comm
        self.objective = func

    def __call__(self, trial):
        # type: (Trial) -> float

        self.comm.mpi_comm.bcast((True, trial.trial_id))
        return self.objective(trial, self.comm)


class ChainerMNStudy(object):
    """A wrapper of :class:`~optuna.study.Study` to incorporate Optuna with ChainerMN.

    .. seealso::
        :class:`~optuna.integration.chainermn.ChainerMNStudy` provides the same interface as
        :class:`~optuna.study.Study`. Please refer to :class:`optuna.study.Study` for further
        details.

    Example:

        Optimize an objective function that trains neural network written with ChainerMN.

        .. code::

            comm = chainermn.create_communicator('naive')
            study = optuna.Study(study_name, storage_url)
            chainermn_study = optuna.integration.ChainerMNStudy(study, comm)
            chainermn_study.optimize(objective, n_trials=25)

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

        if isinstance(study.storage, InMemoryStorage):
            raise ValueError('ChainerMN integration is not available with InMemoryStorage.')

        if isinstance(study.storage, RDBStorage):
            if study.storage.engine.dialect.name == 'sqlite':
                logger = get_logger(__name__)
                logger.warning('SQLite may cause synchronization problems when used with '
                               'ChainerMN integration. Please use other DBs like PostgreSQL.')

        study_names = comm.mpi_comm.allgather(study.study_name)
        if len(set(study_names)) != 1:
            raise ValueError('Please make sure an identical study name is shared among workers.')

        study.pruner = _ChainerMNPruner(pruner=study.pruner, comm=comm)
        super(ChainerMNStudy, self).__setattr__('delegate', study)
        super(ChainerMNStudy, self).__setattr__('comm', comm)

    def optimize(
            self,
            func,  # type: Callable[[Trial, CommunicatorBase], float]
            n_trials=None,  # type: Optional[int]
            timeout=None,  # type: Optional[float]
            catch=(Exception, ),  # type: Union[Tuple[()], Tuple[Type[Exception]]]
    ):
        # type: (...) -> None
        """Optimize an objective function.

        This method provides the same interface as :func:`optuna.study.Study.optimize` except
        the absence of ``n_jobs`` argument.
        """

        if self.comm.rank == 0:
            func_mn = _ChainerMNObjectiveFunc(func, self.comm)
            self.delegate.optimize(func_mn, n_trials=n_trials, timeout=timeout, catch=catch)
            self.comm.mpi_comm.bcast((False, None))
        else:
            while True:
                has_next_trial, trial_id = self.comm.mpi_comm.bcast(None)
                if not has_next_trial:
                    break
                trial = Trial(self.delegate, trial_id)
                try:
                    func(trial, self.comm)

                    # We assume that if a node raises an exception,
                    # all other nodes will do the same.
                    #
                    # The responsibility to handle acceptable exceptions (i.e., `TrialPruned` and
                    # `catch`) is in the rank-0 node, so other nodes simply ignore them.
                except TrialPruned:
                    pass
                except catch:
                    pass

    def __getattr__(self, attr_name):
        # type: (str) -> Any

        return getattr(self.delegate, attr_name)

    def __setattr__(self, attr_name, value):
        # type: (str, Any) -> None

        setattr(self.delegate, attr_name, value)


class _ChainerMNPruner(BasePruner):
    def __init__(self, pruner, comm):
        # type: (BasePruner, CommunicatorBase) -> None

        self.delegate = pruner
        self.comm = comm

    def prune(self, storage, study_id, trial_id, step):
        # type: (BaseStorage, int, int, int) -> bool

        if self.comm.rank == 0:
            try:
                result = self.delegate.prune(storage, study_id, trial_id, step)
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
            'ChainerMN is not available. Please install ChainerMN to use this feature. '
            'ChainerMN can be installed by executing `$ pip install chainermn`. '
            'For further information, please refer to the installation guide of ChainerMN. '
            '(The actual import error is as follows: ' + str(_import_error) + ')')
