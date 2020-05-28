import abc

from optuna import type_checking

if type_checking.TYPE_CHECKING:
    from typing import Any  # NOQA
    from typing import Dict  # NOQA

    from optuna.distributions import BaseDistribution  # NOQA
    from optuna.study import Study  # NOQA
    from optuna.trial import FrozenTrial  # NOQA


class BaseSampler(object, metaclass=abc.ABCMeta):
    """Base class for samplers.

    Optuna combines two types of sampling strategies, which are called *relative sampling* and
    *independent sampling*.

    *The relative sampling* determines values of multiple parameters simultaneously so that
    sampling algorithms can use relationship between parameters (e.g., correlation).
    Target parameters of the relative sampling are described in a relative search space, which
    is determined by :func:`~optuna.samplers.BaseSampler.infer_relative_search_space`.

    *The independent sampling* determines a value of a single parameter without considering any
    relationship between parameters. Target parameters of the independent sampling are the
    parameters not described in the relative search space.

    More specifically, parameters are sampled by the following procedure.
    At the beginning of a trial, :meth:`~optuna.samplers.BaseSampler.infer_relative_search_space`
    is called to determine the relative search space for the trial. Then,
    :meth:`~optuna.samplers.BaseSampler.sample_relative` is invoked to sample parameters
    from the relative search space. During the execution of the objective function,
    :meth:`~optuna.samplers.BaseSampler.sample_independent` is used to sample
    parameters that don't belong to the relative search space.

    The following figure depicts the lifetime of a trial and how the above three methods are
    called in the trial.

    .. image:: ../../image/sampling-sequence.png

    |

    """

    @abc.abstractmethod
    def infer_relative_search_space(self, study, trial):
        # type: (Study, FrozenTrial) -> Dict[str, BaseDistribution]
        """Infer the search space that will be used by relative sampling in the target trial.

        This method is called right before :func:`~optuna.samplers.BaseSampler.sample_relative`
        method, and the search space returned by this method is pass to it. The parameters not
        contained in the search space will be sampled by using
        :func:`~optuna.samplers.BaseSampler.sample_independent` method.

        Args:
            study:
                Target study object.
            trial:
                Target trial object.
                Take a copy before modifying this object.

        Returns:
            A dictionary containing the parameter names and parameter's distributions.

        .. seealso::
            Please refer to :func:`~optuna.samplers.intersection_search_space` as an
            implementation of :func:`~optuna.samplers.BaseSampler.infer_relative_search_space`.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def sample_relative(self, study, trial, search_space):
        # type: (Study, FrozenTrial, Dict[str, BaseDistribution]) -> Dict[str, Any]
        """Sample parameters in a given search space.

        This method is called once at the beginning of each trial, i.e., right before the
        evaluation of the objective function. This method is suitable for sampling algorithms
        that use relationship between parameters such as Gaussian Process and CMA-ES.

        .. note::
                The failed trials are ignored by any build-in samplers when they sample new
                parameters. Thus, failed trials are regarded as deleted in the samplers'
                perspective.

        Args:
            study:
                Target study object.
            trial:
                Target trial object.
                Take a copy before modifying this object.
            search_space:
                The search space returned by
                :func:`~optuna.samplers.BaseSampler.infer_relative_search_space`.

        Returns:
            A dictionary containing the parameter names and the values.

        """

        raise NotImplementedError

    @abc.abstractmethod
    def sample_independent(self, study, trial, param_name, param_distribution):
        # type: (Study, FrozenTrial, str, BaseDistribution) -> Any
        """Sample a parameter for a given distribution.

        This method is called only for the parameters not contained in the search space returned
        by :func:`~optuna.samplers.BaseSampler.sample_relative` method. This method is suitable
        for sampling algorithms that do not use relationship between parameters such as random
        sampling and TPE.

        .. note::
                The failed trials are ignored by any build-in samplers when they sample new
                parameters. Thus, failed trials are regarded as deleted in the samplers'
                perspective.

        Args:
            study:
                Target study object.
            trial:
                Target trial object.
                Take a copy before modifying this object.
            param_name:
                Name of the sampled parameter.
            param_distribution:
                Distribution object that specifies a prior and/or scale of the sampling algorithm.

        Returns:
            A parameter value.

        """

        raise NotImplementedError

    def reseed_rng(self) -> None:
        """Reseed sampler's random number generator.

        This method is called by the :class:`~optuna.study.Study` instance if trials are executed
        in parallel with the option ``n_jobs>1``. In that case, the sampler instance will be
        replicated including the state of the random number generator, and they may suggest the
        same values. To prevent this issue, this method assigns a different seed to each random
        number generator.
        """

        pass
