import abc
import six

from pfnopt.storage._base import BaseStorage  # NOQA
from pfnopt.distributions import _BaseDistribution  # NOQA


@six.add_metaclass(abc.ABCMeta)
class BaseSampler(object):

    @abc.abstractmethod
    def sample(self, storage, study_id, param_name, param_distribution):
        # type: (BaseStorage, int, str, _BaseDistribution) -> float
        raise NotImplementedError
