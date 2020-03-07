import abc
from datetime import datetime
import decimal
import warnings

from optuna import distributions
from optuna import logging
from optuna import type_checking

if type_checking.TYPE_CHECKING:
    from typing import Any  # NOQA
    from typing import Dict  # NOQA
    from typing import Optional  # NOQA
    from typing import Sequence  # NOQA

    from optuna.distributions import BaseDistribution  # NOQA
    from optuna.distributions import CategoricalChoiceType  # NOQA
    from optuna.study import Study  # NOQA


class BaseTrial(object, metaclass=abc.ABCMeta):
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
        # type: (str, Sequence[CategoricalChoiceType]) -> CategoricalChoiceType

        raise NotImplementedError

    def report(self, value, step):
        # type: (float, int) -> None

        raise NotImplementedError

    def should_prune(self, step=None):
        # type: (Optional[int]) -> bool

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
    def distributions(self):
        # type: () -> Dict[str, BaseDistribution]

        raise NotImplementedError

    @property
    def user_attrs(self):
        # type: () -> Dict[str, Any]

        raise NotImplementedError

    @property
    def system_attrs(self):
        # type: () -> Dict[str, Any]

        raise NotImplementedError

    @property
    def datetime_start(self):
        # type: () -> Optional[datetime]

        raise NotImplementedError


class Trial(BaseTrial):
    """A trial is a process of evaluating an objective function.

    This object is passed to an objective function and provides interfaces to get parameter
    suggestion, manage the trial's state, and set/get user-defined attributes of the trial.

    Note that the direct use of this constructor is not recommended.
    This object is seamlessly instantiated and passed to the objective function behind
    the :func:`optuna.study.Study.optimize()` method; hence library users do not care about
    instantiation of this object.

    Args:
        study:
            A :class:`~optuna.study.Study` object.
        trial_id:
            A trial ID that is automatically generated.

    """

    def __init__(
            self,
            study,  # type: Study
            trial_id,  # type: int
    ):
        # type: (...) -> None

        self.study = study
        self._trial_id = trial_id

        # TODO(Yanase): Remove _study_id attribute, and use study._study_id instead.
        self._study_id = self.study._study_id
        self.storage = self.study._storage
        self.logger = logging.get_logger(__name__)

        self._init_relative_params()

    def _init_relative_params(self):
        # type: () -> None

        trial = self.storage.get_trial(self._trial_id)

        self.relative_search_space = self.study.sampler.infer_relative_search_space(
            self.study, trial)
        self.relative_params = self.study.sampler.sample_relative(self.study, trial,
                                                                  self.relative_search_space)

    def suggest_uniform(self, name, low, high):
        # type: (str, float, float) -> float
        """Suggest a value for the continuous parameter.

        The value is sampled from the range :math:`[\\mathsf{low}, \\mathsf{high})`
        in the linear domain. When :math:`\\mathsf{low} = \\mathsf{high}`, the value of
        :math:`\\mathsf{low}` will be returned.

        Example:

            Suggest a momentum for neural network training.

            .. testsetup::

                import numpy as np
                from sklearn.model_selection import train_test_split

                np.random.seed(seed=0)
                X = np.random.randn(200).reshape(-1, 1)
                y = np.random.randint(0, 2, 200)
                X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

            .. testcode::

                import optuna
                from sklearn.neural_network import MLPClassifier

                def objective(trial):
                    momentum = trial.suggest_uniform('momentum', 0.0, 1.0)
                    clf = MLPClassifier(hidden_layer_sizes=(100, 50), momentum=momentum,
                                        solver='sgd', random_state=0)
                    clf.fit(X_train, y_train)

                    return clf.score(X_test, y_test)

                study = optuna.create_study()
                study.optimize(objective, n_trials=3)

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

        distribution = distributions.UniformDistribution(low=low, high=high)

        self._check_distribution(name, distribution)

        if low == high:
            return self._set_new_param_or_get_existing(name, low, distribution)

        return self._suggest(name, distribution)

    def suggest_loguniform(self, name, low, high):
        # type: (str, float, float) -> float
        """Suggest a value for the continuous parameter.

        The value is sampled from the range :math:`[\\mathsf{low}, \\mathsf{high})`
        in the log domain. When :math:`\\mathsf{low} = \\mathsf{high}`, the value of
        :math:`\\mathsf{low}` will be returned.

        Example:

            Suggest penalty parameter ``C`` of `SVC <https://scikit-learn.org/stable/modules/
            generated/sklearn.svm.SVC.html>`_.

            .. testsetup::

                import numpy as np
                from sklearn.model_selection import train_test_split

                np.random.seed(seed=0)
                X = np.random.randn(50).reshape(-1, 1)
                y = np.random.randint(0, 2, 50)
                X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

            .. testcode::

                import optuna
                from sklearn.svm import SVC

                def objective(trial):
                    c = trial.suggest_loguniform('c', 1e-5, 1e2)
                    clf = SVC(C=c, gamma='scale', random_state=0)
                    clf.fit(X_train, y_train)
                    return clf.score(X_test, y_test)

                study = optuna.create_study()
                study.optimize(objective, n_trials=3)

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

        distribution = distributions.LogUniformDistribution(low=low, high=high)

        self._check_distribution(name, distribution)

        if low == high:
            return self._set_new_param_or_get_existing(name, low, distribution)

        return self._suggest(name, distribution)

    def suggest_discrete_uniform(self, name, low, high, q):
        # type: (str, float, float, float) -> float
        """Suggest a value for the discrete parameter.

        The value is sampled from the range :math:`[\\mathsf{low}, \\mathsf{high}]`,
        and the step of discretization is :math:`q`. More specifically,
        this method returns one of the values in the sequence
        :math:`\\mathsf{low}, \\mathsf{low} + q, \\mathsf{low} + 2 q, \\dots,
        \\mathsf{low} + k q \\le \\mathsf{high}`,
        where :math:`k` denotes an integer. Note that :math:`high` may be changed due to round-off
        errors if :math:`q` is not an integer. Please check warning messages to find the changed
        values.

        Example:

            Suggest a fraction of samples used for fitting the individual learners of
            `GradientBoostingClassifier <https://scikit-learn.org/stable/modules/generated/
            sklearn.ensemble.GradientBoostingClassifier.html>`_.

            .. testsetup::

                import numpy as np
                from sklearn.model_selection import train_test_split

                np.random.seed(seed=0)
                X = np.random.randn(50).reshape(-1, 1)
                y = np.random.randint(0, 2, 50)
                X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

            .. testcode::

                import optuna
                from sklearn.ensemble import GradientBoostingClassifier

                def objective(trial):
                    subsample = trial.suggest_discrete_uniform('subsample', 0.1, 1.0, 0.1)
                    clf = GradientBoostingClassifier(subsample=subsample, random_state=0)
                    clf.fit(X_train, y_train)
                    return clf.score(X_test, y_test)

                study = optuna.create_study()
                study.optimize(objective, n_trials=3)

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

        high = _adjust_discrete_uniform_high(name, low, high, q)
        distribution = distributions.DiscreteUniformDistribution(low=low, high=high, q=q)

        self._check_distribution(name, distribution)

        if low == high:
            return self._set_new_param_or_get_existing(name, low, distribution)

        return self._suggest(name, distribution)

    def suggest_int(self, name, low, high):
        # type: (str, int, int) -> int
        """Suggest a value for the integer parameter.

        The value is sampled from the integers in :math:`[\\mathsf{low}, \\mathsf{high}]`.

        Example:

            Suggest the number of trees in `RandomForestClassifier <https://scikit-learn.org/
            stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>`_.

            .. testsetup::

                import numpy as np
                from sklearn.model_selection import train_test_split

                np.random.seed(seed=0)
                X = np.random.randn(50).reshape(-1, 1)
                y = np.random.randint(0, 2, 50)
                X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

            .. testcode::

                import optuna
                from sklearn.ensemble import RandomForestClassifier

                def objective(trial):
                    n_estimators = trial.suggest_int('n_estimators', 50, 400)
                    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=0)
                    clf.fit(X_train, y_train)
                    return clf.score(X_test, y_test)

                study = optuna.create_study()
                study.optimize(objective, n_trials=3)


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

        distribution = distributions.IntUniformDistribution(low=low, high=high)

        self._check_distribution(name, distribution)

        if low == high:
            return self._set_new_param_or_get_existing(name, low, distribution)

        return int(self._suggest(name, distribution))

    def suggest_categorical(self, name, choices):
        # type: (str, Sequence[CategoricalChoiceType]) -> CategoricalChoiceType
        """Suggest a value for the categorical parameter.

        The value is sampled from ``choices``.

        Example:

            Suggest a kernel function of `SVC <https://scikit-learn.org/stable/modules/generated/
            sklearn.svm.SVC.html>`_.

            .. testsetup::

                import numpy as np
                from sklearn.model_selection import train_test_split

                np.random.seed(seed=0)
                X = np.random.randn(50).reshape(-1, 1)
                y = np.random.randint(0, 2, 50)
                X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

            .. testcode::

                import optuna
                from sklearn.svm import SVC

                def objective(trial):
                    kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf'])
                    clf = SVC(kernel=kernel, gamma='scale', random_state=0)
                    clf.fit(X_train, y_train)
                    return clf.score(X_test, y_test)

                study = optuna.create_study()
                study.optimize(objective, n_trials=3)


        Args:
            name:
                A parameter name.
            choices:
                Parameter value candidates.

        .. seealso::
            :class:`~optuna.distributions.CategoricalDistribution`.

        Returns:
            A suggested value.
        """

        choices = tuple(choices)

        # There is no need to call self._check_distribution because
        # CategoricalDistribution does not support dynamic value space.

        return self._suggest(name, distributions.CategoricalDistribution(choices=choices))

    def report(self, value, step):
        # type: (float, int) -> None
        """Report an objective function value for a given step.

        The reported values are used by the pruners to determine whether this trial should be
        pruned.

        .. seealso::
            Please refer to :class:`~optuna.pruners.BasePruner`.

        .. note::
            The reported value is converted to ``float`` type by applying ``float()``
            function internally. Thus, it accepts all float-like types (e.g., ``numpy.float32``).
            If the conversion fails, a ``TypeError`` is raised.

        Example:

            Report intermediate scores of `SGDClassifier <https://scikit-learn.org/stable/modules/
            generated/sklearn.linear_model.SGDClassifier.html>`_ training.

            .. testsetup::

                import numpy as np
                from sklearn.model_selection import train_test_split

                np.random.seed(seed=0)
                X = np.random.randn(50).reshape(-1, 1)
                y = np.random.randint(0, 2, 50)
                X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

            .. testcode::

                import optuna
                from sklearn.linear_model import SGDClassifier

                def objective(trial):
                    clf = SGDClassifier(random_state=0)
                    for step in range(100):
                        clf.partial_fit(X_train, y_train, np.unique(y))
                        intermediate_value = clf.score(X_test, y_test)
                        trial.report(intermediate_value, step=step)
                        if trial.should_prune():
                            raise TrialPruned()

                    return clf.score(X_test, y_test)

                study = optuna.create_study()
                study.optimize(objective, n_trials=3)


        Args:
            value:
                A value returned from the objective function.
            step:
                Step of the trial (e.g., Epoch of neural network training).
        """

        try:
            # For convenience, we allow users to report a value that can be cast to `float`.
            value = float(value)
        except (TypeError, ValueError):
            message = 'The `value` argument is of type \'{}\' but supposed to be a float.'.format(
                type(value).__name__)
            raise TypeError(message)

        if step < 0:
            raise ValueError('The `step` argument is {} but cannot be negative.'.format(step))

        self.storage.set_trial_intermediate_value(self._trial_id, step, value)

    def should_prune(self, step=None):
        # type: (Optional[int]) -> bool
        """Suggest whether the trial should be pruned or not.

        The suggestion is made by a pruning algorithm associated with the trial and is based on
        previously reported values. The algorithm can be specified when constructing a
        :class:`~optuna.study.Study`.

        .. note::
            If no values have been reported, the algorithm cannot make meaningful suggestions.
            Similarly, if this method is called multiple times with the exact same set of reported
            values, the suggestions will be the same.

        .. seealso::
            Please refer to the example code in :func:`optuna.trial.Trial.report`.

        Args:
            step:
                Deprecated since 0.12.0: Step of the trial (e.g., epoch of neural network
                training). Deprecated in favor of always considering the most recent step.

        Returns:
            A boolean value. If :obj:`True`, the trial should be pruned according to the
            configured pruning algorithm. Otherwise, the trial should continue.
        """
        if step is not None:
            warnings.warn(
                'The use of `step` argument is deprecated. '
                'The last reported step is used instead of '
                'the step given by the argument.', DeprecationWarning)

        trial = self.study._storage.get_trial(self._trial_id)
        return self.study.pruner.prune(self.study, trial)

    def set_user_attr(self, key, value):
        # type: (str, Any) -> None
        """Set user attributes to the trial.

        The user attributes in the trial can be access via :func:`optuna.trial.Trial.user_attrs`.

        Example:

            Save fixed hyperparameters of neural network training.

            .. testsetup::

                import numpy as np
                from sklearn.model_selection import train_test_split

                np.random.seed(seed=0)
                X = np.random.randn(50).reshape(-1, 1)
                y = np.random.randint(0, 2, 50)
                X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

            .. testcode::

                import optuna
                from sklearn.neural_network import MLPClassifier

                def objective(trial):
                    trial.set_user_attr('BATCHSIZE', 128)
                    momentum = trial.suggest_uniform('momentum', 0, 1.0)
                    clf = MLPClassifier(hidden_layer_sizes=(100, 50),
                                        batch_size=trial.user_attrs['BATCHSIZE'],
                                        momentum=momentum, solver='sgd', random_state=0)
                    clf.fit(X_train, y_train)

                    return clf.score(X_test, y_test)

                study = optuna.create_study()
                study.optimize(objective, n_trials=3)
                assert 'BATCHSIZE' in study.best_trial.user_attrs.keys()
                assert study.best_trial.user_attrs['BATCHSIZE'] == 128


        Args:
            key:
                A key string of the attribute.
            value:
                A value of the attribute. The value should be JSON serializable.
        """

        self.storage.set_trial_user_attr(self._trial_id, key, value)

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

        self.storage.set_trial_system_attr(self._trial_id, key, value)

    def _suggest(self, name, distribution):
        # type: (str, BaseDistribution) -> Any

        if self._is_fixed_param(name, distribution):
            param_value = self.system_attrs['fixed_params'][name]
        elif self._is_relative_param(name, distribution):
            param_value = self.relative_params[name]
        else:
            trial = self.storage.get_trial(self._trial_id)
            param_value = self.study.sampler.sample_independent(
                self.study, trial, name, distribution)

        return self._set_new_param_or_get_existing(name, param_value, distribution)

    def _set_new_param_or_get_existing(self, name, param_value, distribution):
        # type: (str, Any, BaseDistribution) -> Any

        param_value_in_internal_repr = distribution.to_internal_repr(param_value)
        set_success = self.storage.set_trial_param(self._trial_id, name,
                                                   param_value_in_internal_repr, distribution)
        if not set_success:
            param_value_in_internal_repr = self.storage.get_trial_param(self._trial_id, name)
            param_value = distribution.to_external_repr(param_value_in_internal_repr)

        return param_value

    def _is_fixed_param(self, name, distribution):
        # type: (str, BaseDistribution) -> bool

        if 'fixed_params' not in self.system_attrs:
            return False

        if name not in self.system_attrs['fixed_params']:
            return False

        param_value = self.system_attrs['fixed_params'][name]
        param_value_in_internal_repr = distribution.to_internal_repr(param_value)

        contained = distribution._contains(param_value_in_internal_repr)
        if not contained:
            warnings.warn("Fixed parameter '{}' with value {} is out of range "
                          "for distribution {}.".format(name, param_value, distribution))
        return contained

    def _is_relative_param(self, name, distribution):
        # type: (str, BaseDistribution) -> bool

        if name not in self.relative_params:
            return False

        if name not in self.relative_search_space:
            raise ValueError("The parameter '{}' was sampled by `sample_relative` method "
                             "but it is not contained in the relative search space.".format(name))

        relative_distribution = self.relative_search_space[name]
        distributions.check_distribution_compatibility(relative_distribution, distribution)

        param_value = self.relative_params[name]
        param_value_in_internal_repr = distribution.to_internal_repr(param_value)
        return distribution._contains(param_value_in_internal_repr)

    def _check_distribution(self, name, distribution):
        # type: (str, BaseDistribution) -> None

        old_distribution = self.distributions.get(name, distribution)
        if old_distribution != distribution:
            warnings.warn('Inconsistent parameter values for distribution with name "{}"! '
                          'This might be a configuration mistake. '
                          'Optuna allows to call the same distribution with the same '
                          'name more then once in a trial. '
                          'When the parameter values are inconsistent optuna only '
                          'uses the values of the first call and ignores all following. '
                          'Using these values: {}'
                          .format(name, old_distribution._asdict()), RuntimeWarning)

    @property
    def number(self):
        # type: () -> int
        """Return trial's number which is consecutive and unique in a study.

        Returns:
            A trial number.
        """

        return self.storage.get_trial_number_from_id(self._trial_id)

    @property
    def trial_id(self):
        # type: () -> int
        """Return trial ID.

        Note that the use of this is deprecated.
        Please use :attr:`~optuna.trial.Trial.number` instead.

        Returns:
            A trial ID.
        """

        warnings.warn(
            'The use of `Trial.trial_id` is deprecated. '
            'Please use `Trial.number` instead.', DeprecationWarning)

        self.logger.warning('The use of `Trial.trial_id` is deprecated. '
                            'Please use `Trial.number` instead.')

        return self._trial_id

    @property
    def params(self):
        # type: () -> Dict[str, Any]
        """Return parameters to be optimized.

        Returns:
            A dictionary containing all parameters.
        """

        return self.storage.get_trial_params(self._trial_id)

    @property
    def distributions(self):
        # type: () -> Dict[str, BaseDistribution]
        """Return distributions of parameters to be optimized.

        Returns:
            A dictionary containing all distributions.
        """

        return self.storage.get_trial(self._trial_id).distributions

    @property
    def user_attrs(self):
        # type: () -> Dict[str, Any]
        """Return user attributes.

        Returns:
            A dictionary containing all user attributes.
        """

        return self.storage.get_trial_user_attrs(self._trial_id)

    @property
    def system_attrs(self):
        # type: () -> Dict[str, Any]
        """Return system attributes.

        Returns:
            A dictionary containing all system attributes.
        """

        return self.storage.get_trial_system_attrs(self._trial_id)

    @property
    def datetime_start(self):
        # type: () -> Optional[datetime]
        """Return start datetime.

        Returns:
            Datetime where the :class:`~optuna.trial.Trial` started.
        """
        return self.storage.get_trial(self._trial_id).datetime_start

    @property
    def study_id(self):
        # type: () -> int
        """Return the study ID.

        .. deprecated:: 0.20.0
            The direct use of this attribute is deprecated and it is recommended that you use
            :attr:`~optuna.trial.Trial.study` instead.

        Returns:
            The study ID.
        """

        message = 'The use of `Trial.study_id` is deprecated. ' \
                  'Please use `Trial.study` instead.'
        warnings.warn(message, DeprecationWarning)
        self.logger.warning(message)

        return self.study._study_id


class FixedTrial(BaseTrial):
    """A trial class which suggests a fixed value for each parameter.

    This object has the same methods as :class:`~optuna.trial.Trial`, and it suggests pre-defined
    parameter values. The parameter values can be determined at the construction of the
    :class:`~optuna.trial.FixedTrial` object. In contrast to :class:`~optuna.trial.Trial`,
    :class:`~optuna.trial.FixedTrial` does not depend on :class:`~optuna.study.Study`, and it is
    useful for deploying optimization results.

    Example:

        Evaluate an objective function with parameter values given by a user.

        .. testcode::

            import optuna

            def objective(trial):
                x = trial.suggest_uniform('x', -100, 100)
                y = trial.suggest_categorical('y', [-1, 0, 1])
                return x ** 2 + y

            assert objective(optuna.trial.FixedTrial({'x': 1, 'y': 0})) == 1


    .. note::
        Please refer to :class:`~optuna.trial.Trial` for details of methods and properties.

    Args:
        params:
            A dictionary containing all parameters.

    """

    def __init__(self, params):
        # type: (Dict[str, Any]) -> None

        self._params = params
        self._suggested_params = {}  # type: Dict[str, Any]
        self._distributions = {}  # type: Dict[str, BaseDistribution]
        self._user_attrs = {}  # type: Dict[str, Any]
        self._system_attrs = {}  # type: Dict[str, Any]
        self._datetime_start = datetime.now()

    def suggest_uniform(self, name, low, high):
        # type: (str, float, float) -> float

        return self._suggest(name, distributions.UniformDistribution(low=low, high=high))

    def suggest_loguniform(self, name, low, high):
        # type: (str, float, float) -> float

        return self._suggest(name, distributions.LogUniformDistribution(low=low, high=high))

    def suggest_discrete_uniform(self, name, low, high, q):
        # type: (str, float, float, float) -> float

        high = _adjust_discrete_uniform_high(name, low, high, q)
        discrete = distributions.DiscreteUniformDistribution(low=low, high=high, q=q)
        return self._suggest(name, discrete)

    def suggest_int(self, name, low, high):
        # type: (str, int, int) -> int

        return int(self._suggest(name, distributions.IntUniformDistribution(low=low, high=high)))

    def suggest_categorical(self, name, choices):
        # type: (str, Sequence[CategoricalChoiceType]) -> CategoricalChoiceType

        choices = tuple(choices)
        return self._suggest(name, distributions.CategoricalDistribution(choices=choices))

    def _suggest(self, name, distribution):
        # type: (str, BaseDistribution) -> Any

        if name not in self._params:
            raise ValueError('The value of the parameter \'{}\' is not found. Please set it at '
                             'the construction of the FixedTrial object.'.format(name))

        value = self._params[name]
        param_value_in_internal_repr = distribution.to_internal_repr(value)
        if not distribution._contains(param_value_in_internal_repr):
            raise ValueError("The value {} of the parameter '{}' is out of "
                             "the range of the distribution {}.".format(value, name, distribution))

        if name in self._distributions:
            distributions.check_distribution_compatibility(self._distributions[name], distribution)

        self._suggested_params[name] = value
        self._distributions[name] = distribution

        return value

    def report(self, value, step):
        # type: (float, int) -> None

        pass

    def should_prune(self, step=None):
        # type: (Optional[int]) -> bool

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

        return self._suggested_params

    @property
    def distributions(self):
        # type: () -> Dict[str, BaseDistribution]

        return self._distributions

    @property
    def user_attrs(self):
        # type: () -> Dict[str, Any]

        return self._user_attrs

    @property
    def system_attrs(self):
        # type: () -> Dict[str, Any]

        return self._system_attrs

    @property
    def datetime_start(self):
        # type: () -> Optional[datetime]

        return self._datetime_start


def _adjust_discrete_uniform_high(name, low, high, q):
    # type: (str, float, float, float) -> float

    d_high = decimal.Decimal(str(high))
    d_low = decimal.Decimal(str(low))
    d_q = decimal.Decimal(str(q))

    d_r = d_high - d_low

    if d_r % d_q != decimal.Decimal('0'):
        high = float((d_r // d_q) * d_q + d_low)
        logger = logging.get_logger(__name__)
        logger.warning('The range of parameter `{}` is not divisible by `q`, and is '
                       'replaced by [{}, {}].'.format(name, low, high))

    return high
