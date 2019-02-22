import abc
import six

from optuna import distributions
from optuna import types

if types.TYPE_CHECKING:
    from optuna.study import Study  # NOQA
    from typing import Any  # NOQA
    from typing import Dict  # NOQA
    from typing import Optional  # NOQA
    from typing import Sequence  # NOQA
    from typing import TypeVar  # NOQA

    T = TypeVar('T', float, str)


@six.add_metaclass(abc.ABCMeta)
class BaseTrial(object):
    """Base class for trials.

    Note that this class is not supposed to be directly accessed by library users.
    """

    def suggest_uniform(self, name, low, high):
        # type: (str, float, float) -> float

        raise NotImplementedError

    def suggest_loguniform(self, name, low, high):
        # type: (str, float, float) -> float

        raise NotImplementedError

    def suggest_discrete_uniform(self, name, low, high, q):
        # type: (str, float, float, float) -> float

        raise NotImplementedError

    def suggest_int(self, name, low, high):
        # type: (str, int, int) -> int

        raise NotImplementedError

    def suggest_categorical(self, name, choices):
        # type: (str, Sequence[T]) -> T

        raise NotImplementedError

    def report(self, value, step=None):
        # type: (float, Optional[int]) -> None

        raise NotImplementedError

    def should_prune(self, step):
        # type: (int) -> bool

        raise NotImplementedError

    def set_user_attr(self, key, value):
        # type: (str, Any) -> None

        raise NotImplementedError

    def set_system_attr(self, key, value):
        # type: (str, Any) -> None

        raise NotImplementedError

    @property
    def params(self):
        # type: () -> Dict[str, Any]

        raise NotImplementedError

    @property
    def user_attrs(self):
        # type: () -> Dict[str, Any]

        raise NotImplementedError

    @property
    def system_attrs(self):
        # type: () -> Dict[str, Any]

        raise NotImplementedError


class Trial(BaseTrial):
    """A trial is a process of evaluating an objective function.

    This object is passed to an objective function and provides interfaces to get parameter
    suggestion, manage the trial's state, and set/get user-defined attributes of the trial.

    Note that this object is seamlessly instantiated and passed to the objective function behind
    :func:`optuna.study.Study.optimize()` method (as well as optimize function); hence, in typical
    use cases, library users do not care about instantiation of this object.

    Args:
        study:
            A :class:`~optuna.study.Study` object.
        trial_id:
            A trial ID that is automatically generated.

    """

    def __init__(self, study, trial_id):
        # type: (Study, int) -> None

        self.study = study
        self.trial_id = trial_id

        self.study_id = self.study.study_id
        self.storage = self.study.storage

    def suggest_uniform(self, name, low, high):
        # type: (str, float, float) -> float
        """Suggest a value for the continuous parameter.

        The value is sampled from the range ``[low, high)`` in the linear domain.

        Example:

            Suggest a dropout rate for neural network training.

            .. code::

                >>> def objective(trial):
                >>>     ...
                >>>     dropout_rate = trial.suggest_unifrom('dropout_rate', 0, 1.0)
                >>>     ...

        Args:
            name:
                A parameter name.
            low:
                Lower endpoint of the range of suggested values. ``low`` is included in the range.
            high:
                Upper endpoint of the range of suggested values. ``high`` is excluded from the
                range.

        Returns:
            A suggested float value.
        """

        return self._suggest(name, distributions.UniformDistribution(low=low, high=high))

    def suggest_loguniform(self, name, low, high):
        # type: (str, float, float) -> float
        """Suggest a value for the continuous parameter.

        The value is sampled from the range ``[low, high)`` in the log domain.

        Example:

            Suggest penalty parameter ``C`` of `SVC <https://scikit-learn.org/stable/modules/
            generated/sklearn.svm.SVC.html>`_.

            .. code::

                >>> def objective(trial):
                >>>     ...
                >>>     c = trial.suggest_logunifrom('c', 1e-5, 1e2)
                >>>     clf = sklearn.svm.SVC(C=c)
                >>>     ...

        Args:
            name:
                A parameter name.
            low:
                Lower endpoint of the range of suggested values. ``low`` is included in the range.
            high:
                Upper endpoint of the range of suggested values. ``high`` is excluded from the
                range.

        Returns:
            A suggested float value.
        """

        return self._suggest(name, distributions.LogUniformDistribution(low=low, high=high))

    def suggest_discrete_uniform(self, name, low, high, q):
        # type: (str, float, float, float) -> float
        """Suggest a value for the discrete parameter.

        The value is sampled from the range ``[low, high]``, and the step of discretization is
        ``q``.

        Example:

            Suggest a fraction of samples used for fitting the individual learners of
            `GradientBoostingClassifier <https://scikit-learn.org/stable/modules/generated/
            sklearn.ensemble.GradientBoostingClassifier.html>`_.

            .. code::

                >>> def objective(trial):
                >>>     ...
                >>>     subsample = trial.suggest_discrete_uniform('subsample', 0.1, 1.0, 0.1)
                >>>     clf = sklearn.ensemble.GradientBoostingClassifier(subsample=subsample)
                >>>     ...

        Args:
            name:
                A parameter name.
            low:
                Lower endpoint of the range of suggested values. ``low`` is included in the range.
            high:
                Upper endpoint of the range of suggested values. ``high`` is included in the range.
            q:
                A step of discretization.

        Returns:
            A suggested float value.
        """

        discrete = distributions.DiscreteUniformDistribution(low=low, high=high, q=q)
        return self._suggest(name, discrete)

    def suggest_int(self, name, low, high):
        # type: (str, int, int) -> int
        """Suggest a value for the integer parameter.

        The value is sampled from the integers in ``[low, high]``.

        Example:

            Suggest the number of trees in `RandomForestClassifier <https://scikit-learn.org/
            stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>`_.

            .. code::

                >>> def objective(trial):
                >>>     ...
                >>>     n_estimators = trial.suggest_int('n_estimators', 50, 400)
                >>>     clf = sklearn.ensemble.RandomForestClassifier(n_estimators=n_estimators)
                >>>     ...

        Args:
            name:
                A parameter name.
            low:
                Lower endpoint of the range of suggested values. ``low`` is included in the range.
            high:
                Upper endpoint of the range of suggested values. ``high`` is included in the range.

        Returns:
            A suggested integer value.
        """

        return int(self._suggest(name, distributions.IntUniformDistribution(low=low, high=high)))

    def suggest_categorical(self, name, choices):
        # type: (str, Sequence[T]) -> T
        """Suggest a value for the categorical parameter.

        The value is sampled from ``choices``.

        Example:

            Suggest a kernel function of `SVC <https://scikit-learn.org/stable/modules/generated/
            sklearn.svm.SVC.html>`_.

            .. code::

                >>> def objective(trial):
                >>>     ...
                >>>     kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf'])
                >>>     clf = sklearn.svm.SVC(kernel=kernel)
                >>>     ...

        Args:
            name:
                A parameter name.
            choices:
                Candidates of parameter values.

        Returns:
            A suggested value.
        """

        choices = tuple(choices)
        return self._suggest(name, distributions.CategoricalDistribution(choices=choices))

    def report(self, value, step=None):
        # type: (float, Optional[int]) -> None
        """Report an objective function value.

        If step is set to :obj:`None`, the value is stored as a final value of the trial.
        Otherwise, it is saved as an intermediate value.

        Example:

            Report intermediate scores of `SGDClassifier <https://scikit-learn.org/stable/modules/
            generated/sklearn.linear_model.SGDClassifier.html>`_ training

            .. code::

                >>> def objective(trial):
                >>>     ...
                >>>     clf = sklearn.linear_model.SGDClassifier()
                >>>     for step in range(100):
                >>>         clf.partial_fit(x_train , y_train , classes)
                >>>         intermediate_value = clf.score(x_val , y_val)
                >>>         trial.report(intermediate_value , step=step)
                >>>         if trial.should_prune(step):
                >>>             raise TrialPruned()
                >>>     ...

        Args:
            value:
                A value returned from the objective function.
            step:
                Step of the trial (e.g., Epoch of neural network training).
        """

        self.storage.set_trial_value(self.trial_id, value)
        if step is not None:
            self.storage.set_trial_intermediate_value(self.trial_id, step, value)

    def should_prune(self, step):
        # type: (int) -> bool
        """Judge whether the trial should be pruned.

        This method calls prune method of the pruner, which judges whether the trial should
        be pruned at the given step. Please refer to the example code of
        :func:`optuna.trial.Trial.report`.

        Args:
            step:
                Step of the trial (e.g., epoch of neural network training).

        Returns:
            A boolean value. If :obj:`True`, the trial should be pruned. Otherwise, the trial will
            be continued.
        """

        # TODO(akiba): remove `step` argument

        return self.study.pruner.prune(self.storage, self.study_id, self.trial_id, step)

    def set_user_attr(self, key, value):
        # type: (str, Any) -> None
        """Set user attributes to the trial.

        The user attributes in the trial can be access via :func:`optuna.trial.Trial.user_attrs`.

        Example:

            Save fixed hyperparameters of neural network training:

            .. code::

                >>> def objective(trial):
                >>>     ...
                >>>     trial.set_user_attr('BATCHSIZE', 128)
                >>>
                >>> study.best_trial.user_attrs
                {'BATCHSIZE': 128}


        Args:
            key:
                A key string of the attribute.
            value:
                A value of the attribute. The value should be JSON serializable.
        """

        self.storage.set_trial_user_attr(self.trial_id, key, value)

    def set_system_attr(self, key, value):
        # type: (str, Any) -> None
        """Set system attributes to the trial.

        Note that Optuna internally uses this method to save system messages such as failure
        reason of trials. Please use :func:`~optuna.trial.Trial.set_user_attr` to set users'
        attributes.

        Args:
            key:
                A key string of the attribute.
            value:
                A value of the attribute. The value should be JSON serializable.
        """

        self.storage.set_trial_system_attr(self.trial_id, key, value)

    def _suggest(self, name, distribution):
        # type: (str, distributions.BaseDistribution) -> Any

        param_value_in_internal_repr = self.study.sampler.sample(self.storage, self.study_id, name,
                                                                 distribution)

        set_success = self.storage.set_trial_param(self.trial_id, name,
                                                   param_value_in_internal_repr, distribution)
        if not set_success:
            param_value_in_internal_repr = self.storage.get_trial_param(self.trial_id, name)

        param_value = distribution.to_external_repr(param_value_in_internal_repr)
        return param_value

    @property
    def params(self):
        # type: () -> Dict[str, Any]
        """Return parameters to be optimized.

        Returns:
            A dictionary containing all parameters.
        """

        return self.storage.get_trial_params(self.trial_id)

    @property
    def user_attrs(self):
        # type: () -> Dict[str, Any]
        """Return user attributes.

        Returns:
            A dictionary containing all user attributes.
        """

        return self.storage.get_trial_user_attrs(self.trial_id)

    @property
    def system_attrs(self):
        # type: () -> Dict[str, Any]
        """Return system attributes.

        Returns:
            A dictionary containing all system attributes.
        """

        return self.storage.get_trial_system_attrs(self.trial_id)


class FixedTrial(BaseTrial):
    """A trial class which suggests a fixed value for each parameter.

    This object has the same methods as :class:`~optuna.trial.Trial`, and it suggests pre-defined
    parameter values. The parameter values can be determined at the construction of the
    :class:`~optuna.trial.FixedTrial` object. In contrast to :class:`~optuna.trial.Trial`,
    :class:`~optuna.trial.FixedTrial` does not depend on :class:`~optuna.study.Study`, and it is
    useful for deploying optimization results.

    Example:

        Evaluate an objective function with parameter values given by a user:

        .. code::

            >>> def objective(trial):
            >>>     x = trial.suggest_uniform('x', -100, 100)
            >>>     y = trial.suggest_categorical('y', [-1, 0, 1])
            >>>     return x ** 2 + y
            >>>
            >>> objective(FixedTrial({'x': 1, 'y': 0}))
            1

    .. note::
        Please refer to :class:`~optuna.trial.Trial` for details of methods and properties.

    Args:
        params:
            A dictionary containing all parameters.

    """

    def __init__(self, params):
        # type: (Dict[str, Any]) -> None

        self._params = params
        self._user_attrs = {}  # type: Dict[str, Any]
        self._system_attrs = {}  # type: Dict[str, Any]

    def suggest_uniform(self, name, low, high):
        # type: (str, float, float) -> float

        return self._suggest(name)

    def suggest_loguniform(self, name, low, high):
        # type: (str, float, float) -> float

        return self._suggest(name)

    def suggest_discrete_uniform(self, name, low, high, q):
        # type: (str, float, float, float) -> float

        return self._suggest(name)

    def suggest_int(self, name, low, high):
        # type: (str, int, int) -> int

        return self._suggest(name)

    def suggest_categorical(self, name, choices):
        # type: (str, Sequence[T]) -> T

        return self._suggest(name)

    def _suggest(self, name):
        # type: (str) -> Any

        if name not in self._params:
            raise ValueError('The value of the parameter \'{}\' is not found. Please set it at '
                             'the construction of the FixedTrial object.'.format(name))

        return self._params[name]

    def report(self, value, step=None):
        # type: (float, Optional[int]) -> None

        pass

    def should_prune(self, step):
        # type: (int) -> bool

        return False

    def set_user_attr(self, key, value):
        # type: (str, Any) -> None

        self._user_attrs[key] = value

    def set_system_attr(self, key, value):
        # type: (str, Any) -> None

        self._system_attrs[key] = value

    @property
    def params(self):
        # type: () -> Dict[str, Any]

        return self._params

    @property
    def user_attrs(self):
        # type: () -> Dict[str, Any]

        return self._user_attrs

    @property
    def system_attrs(self):
        # type: () -> Dict[str, Any]

        return self._system_attrs
