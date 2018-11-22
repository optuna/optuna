from __future__ import absolute_import

from typing import Callable  # NOQA
from typing import Optional  # NOQA
from typing import Tuple  # NOQA
from typing import Type  # NOQA
from typing import Union  # NOQA

from optuna.logging import get_logger
from optuna.storages import InMemoryStorage
from optuna.storages import RDBStorage
from optuna.study import Study  # NOQA
from optuna.trial import Trial  # NOQA

try:
    from chainermn.communicators.communicator_base import CommunicatorBase  # NOQA
    _available = True
except ImportError as e:
    _import_error = e
    _available = False


class ChainerMNObjectiveFunc(object):

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

        super(ChainerMNStudy, self).__setattr__('delegate', study)
        super(ChainerMNStudy, self).__setattr__('comm', comm)

    def optimize(
        self,
        func,  # type: Callable[[Trial, CommunicatorBase], float]
        n_trials=None,  # type: Optional[int]
        timeout=None,  # type: Optional[float]
        catch=(Exception,),  # type: Union[Tuple[()], Tuple[Type[Exception]]]
    ):
        # type: (...) -> None
        """Optimize an objective function.

        This method provides the same interface as :func:`optuna.study.Study.optimize` except
        the absence of ``n_jobs`` argument.
        """

        if self.comm.rank == 0:
            func_mn = ChainerMNObjectiveFunc(func, self.comm)
            self.delegate.optimize(func_mn, n_trials=n_trials, timeout=timeout, catch=catch)
            self.comm.mpi_comm.bcast((False, None))
        else:
            while True:
                has_next_trial, trial_id = self.comm.mpi_comm.bcast(None)
                if not has_next_trial:
                    break
                trial = Trial(self.delegate, trial_id)
                func(trial, self.comm)

    def __getattr__(self, attr_name):
        return getattr(self.delegate, attr_name)

    def __setattr__(self, attr_name, value):
        setattr(self.delegate, attr_name, value)


def _check_chainermn_availability():
    # type: () -> None

    if not _available:
        raise ImportError(
            'ChainerMN is not available. Please install ChainerMN to use this feature. '
            'ChainerMN can be installed by executing `$ pip install chainermn`. '
            'For further information, please refer to the installation guide of ChainerMN. '
            '(The actual import error is as follows: ' + str(_import_error) + ')')
