import abc
import six


@six.add_metaclass(abc.ABCMeta)
class BaseSampler(object):

    @abc.abstractmethod
    def sample(self, distribution, observation_pairs):
        raise NotImplementedError
