import copy
import datetime
from typing import Optional
import warnings

from optuna import distributions
from optuna.distributions import CategoricalDistribution
from optuna.distributions import DiscreteUniformDistribution
from optuna.distributions import IntLogUniformDistribution
from optuna.distributions import IntUniformDistribution
from optuna.distributions import LogUniformDistribution
from optuna.distributions import UniformDistribution
from optuna import logging
from optuna import pruners
from optuna.trial._base import BaseTrial
from optuna import type_checking

if type_checking.TYPE_CHECKING:
    from typing import Any  # NOQA
    from typing import Dict  # NOQA
    from typing import Sequence  # NOQA
    from typing import Union  # NOQA

    from optuna.distributions import BaseDistribution  # NOQA
    from optuna.distributions import CategoricalChoiceType  # NOQA
    from optuna.study import Study  # NOQA

    FloatingPointDistributionType = Union[UniformDistribution, LogUniformDistribution]


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

        study = pruners._filter_study(self.study, trial)

        self.relative_search_space = self.study.sampler.infer_relative_search_space(study, trial)
        self.relative_params = self.study.sampler.sample_relative(
            study, trial, self.relative_search_space
        )

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        step: Optional[float] = None,
        log: bool = False
    ) -> float:
        """Suggest a value for the floating point parameter.

        Note that this is a wrapper method for :func:`~optuna.trial.Trial.suggest_uniform`,
        :func:`~optuna.trial.Trial.suggest_loguniform` and
        :func:`~optuna.trial.Trial.suggest_discrete_uniform`.

        .. versionadded:: 1.3.0

        .. seealso::
            Please see also :func:`~optuna.trial.Trial.suggest_uniform`,
            :func:`~optuna.trial.Trial.suggest_loguniform` and
            :func:`~optuna.trial.Trial.suggest_discrete_uniform`.

        Example:

            Suggest a momentum, learning rate and scaling factor of learning rate
            for neural network training.

            .. testcode::

                import numpy as np
                from sklearn.datasets import load_iris
                from sklearn.model_selection import train_test_split
                from sklearn.neural_network import MLPClassifier

                import optuna

                X, y = load_iris(return_X_y=True)
                X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=0)

                def objective(trial):
                    momentum = trial.suggest_float('momentum', 0.0, 1.0)
                    learning_rate_init = trial.suggest_float('learning_rate_init',
                                                             1e-5, 1e-3, log=True)
                    power_t = trial.suggest_float('power_t', 0.2, 0.8, step=0.1)
                    clf = MLPClassifier(hidden_layer_sizes=(100, 50), momentum=momentum,
                                        learning_rate_init=learning_rate_init,
                                        solver='sgd', random_state=0, power_t=power_t)
                    clf.fit(X_train, y_train)

                    return clf.score(X_valid, y_valid)

                study = optuna.create_study(direction='maximize')
                study.optimize(objective, n_trials=3)

        Args:
            name:
                A parameter name.
            low:
                Lower endpoint of the range of suggested values. ``low`` is included in the range.
            high:
                Upper endpoint of the range of suggested values. ``high`` is excluded from the
                range.
            step:
                A step of discretization.
            log:
                A flag to sample the value from the log domain or not.
                If ``log`` is true, the value is sampled from the range in the log domain.
                Otherwise, the value is sampled from the range in the linear domain.
                See also :func:`suggest_uniform` and :func:`suggest_loguniform`.

        Returns:
            A suggested float value.
        """

        if step is not None:
            if log:
                raise NotImplementedError(
                    "The parameter `step` is not supported when `log` is True."
                )
            else:
                return self.suggest_discrete_uniform(name, low, high, step)
        else:
            if log:
                return self.suggest_loguniform(name, low, high)
            else:
                return self.suggest_uniform(name, low, high)

    def suggest_uniform(self, name, low, high):
        # type: (str, float, float) -> float
        """Suggest a value for the continuous parameter.

        The value is sampled from the range :math:`[\\mathsf{low}, \\mathsf{high})`
        in the linear domain. When :math:`\\mathsf{low} = \\mathsf{high}`, the value of
        :math:`\\mathsf{low}` will be returned.

        Example:

            Suggest a momentum for neural network training.

            .. testcode::

                import numpy as np
                from sklearn.datasets import load_iris
                from sklearn.model_selection import train_test_split
                from sklearn.neural_network import MLPClassifier

                import optuna

                X, y = load_iris(return_X_y=True)
                X_train, X_valid, y_train, y_valid = train_test_split(X, y)

                def objective(trial):
                    momentum = trial.suggest_uniform('momentum', 0.0, 1.0)
                    clf = MLPClassifier(hidden_layer_sizes=(100, 50), momentum=momentum,
                                        solver='sgd', random_state=0)
                    clf.fit(X_train, y_train)

                    return clf.score(X_valid, y_valid)

                study = optuna.create_study(direction='maximize')
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

        distribution = UniformDistribution(low=low, high=high)

        self._check_distribution(name, distribution)

        if distribution.low == distribution.high:
            return self._set_new_param_or_get_existing(name, distribution.low, distribution)

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

            .. testcode::

                import numpy as np
                from sklearn.datasets import load_iris
                from sklearn.model_selection import train_test_split
                from sklearn.svm import SVC

                import optuna

                X, y = load_iris(return_X_y=True)
                X_train, X_valid, y_train, y_valid = train_test_split(X, y)

                def objective(trial):
                    c = trial.suggest_loguniform('c', 1e-5, 1e2)
                    clf = SVC(C=c, gamma='scale', random_state=0)
                    clf.fit(X_train, y_train)
                    return clf.score(X_valid, y_valid)

                study = optuna.create_study(direction='maximize')
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

        distribution = LogUniformDistribution(low=low, high=high)

        self._check_distribution(name, distribution)

        if distribution.low == distribution.high:
            return self._set_new_param_or_get_existing(name, distribution.low, distribution)

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

            .. testcode::

                import numpy as np
                from sklearn.datasets import load_iris
                from sklearn.ensemble import GradientBoostingClassifier
                from sklearn.model_selection import train_test_split

                import optuna

                X, y = load_iris(return_X_y=True)
                X_train, X_valid, y_train, y_valid = train_test_split(X, y)

                def objective(trial):
                    subsample = trial.suggest_discrete_uniform('subsample', 0.1, 1.0, 0.1)
                    clf = GradientBoostingClassifier(subsample=subsample, random_state=0)
                    clf.fit(X_train, y_train)
                    return clf.score(X_valid, y_valid)

                study = optuna.create_study(direction='maximize')
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

        distribution = DiscreteUniformDistribution(low=low, high=high, q=q)

        self._check_distribution(name, distribution)

        if distribution.low == distribution.high:
            return self._set_new_param_or_get_existing(name, distribution.low, distribution)

        return self._suggest(name, distribution)

    def suggest_int(self, name, low, high, step=1, log=False):
        # type: (str, int, int, int, bool) -> int
        """Suggest a value for the integer parameter.

        The value is sampled from the integers in :math:`[\\mathsf{low}, \\mathsf{high}]`, and the
        step of discretization is :math:`\\mathsf{step}`. More specifically, this method returns
        one of the values in the sequence :math:`\\mathsf{low}, \\mathsf{low} + \\mathsf{step},
        \\mathsf{low} + 2 * \\mathsf{step}, \\dots, \\mathsf{low} + k * \\mathsf{step} \\le
        \\mathsf{high}`, where :math:`k` denotes an integer. Note that :math:`\\mathsf{high}` is
        modified if the range is not divisible by :math:`\\mathsf{step}`. Please check the warning
        messages to find the changed values.

        Example:

            Suggest the number of trees in `RandomForestClassifier <https://scikit-learn.org/
            stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>`_.

            .. testcode::

                import numpy as np
                from sklearn.datasets import load_iris
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.model_selection import train_test_split

                import optuna

                X, y = load_iris(return_X_y=True)
                X_train, X_valid, y_train, y_valid = train_test_split(X, y)

                def objective(trial):
                    n_estimators = trial.suggest_int('n_estimators', 50, 400)
                    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=0)
                    clf.fit(X_train, y_train)
                    return clf.score(X_valid, y_valid)

                study = optuna.create_study(direction='maximize')
                study.optimize(objective, n_trials=3)


        Args:
            name:
                A parameter name.
            low:
                Lower endpoint of the range of suggested values. ``low`` is included in the range.
            high:
                Upper endpoint of the range of suggested values. ``high`` is included in the range.
            step:
                A step of discretization.
            log:
                A flag to sample the value from the log domain or not.
                If ``log`` is true, at first, the range of suggested values is divided into grid
                points of width ``step``. The range of suggested values is then converted to a log
                domain, from which a value is uniformly sampled. The uniformly sampled value is
                re-converted to the original domain and rounded to the nearest grid point that we
                just split, and the suggested value is determined.
                For example,
                if `low = 2`, `high = 8` and `step = 2`,
                then the range of suggested values is divided by ``step`` as `[2, 4, 6, 8]`
                and lower values tend to be more sampled than higher values.
        """

        if log:
            distribution = IntLogUniformDistribution(
                low=low, high=high, step=step
            )  # type: Union[IntUniformDistribution, IntLogUniformDistribution]
        else:
            distribution = IntUniformDistribution(low=low, high=high, step=step)

        self._check_distribution(name, distribution)

        if distribution.low == distribution.high:
            return self._set_new_param_or_get_existing(name, distribution.low, distribution)

        return int(self._suggest(name, distribution))

    def suggest_categorical(self, name, choices):
        # type: (str, Sequence[CategoricalChoiceType]) -> CategoricalChoiceType
        """Suggest a value for the categorical parameter.

        The value is sampled from ``choices``.

        Example:

            Suggest a kernel function of `SVC <https://scikit-learn.org/stable/modules/generated/
            sklearn.svm.SVC.html>`_.

            .. testcode::

                import numpy as np
                from sklearn.datasets import load_iris
                from sklearn.model_selection import train_test_split
                from sklearn.svm import SVC

                import optuna

                X, y = load_iris(return_X_y=True)
                X_train, X_valid, y_train, y_valid = train_test_split(X, y)

                def objective(trial):
                    kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf'])
                    clf = SVC(kernel=kernel, gamma='scale', random_state=0)
                    clf.fit(X_train, y_train)
                    return clf.score(X_valid, y_valid)

                study = optuna.create_study(direction='maximize')
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

        return self._suggest(name, CategoricalDistribution(choices=choices))

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

            .. testcode::

                import numpy as np
                from sklearn.datasets import load_iris
                from sklearn.linear_model import SGDClassifier
                from sklearn.model_selection import train_test_split

                import optuna

                X, y = load_iris(return_X_y=True)
                X_train, X_valid, y_train, y_valid = train_test_split(X, y)

                def objective(trial):
                    clf = SGDClassifier(random_state=0)
                    for step in range(100):
                        clf.partial_fit(X_train, y_train, np.unique(y))
                        intermediate_value = clf.score(X_valid, y_valid)
                        trial.report(intermediate_value, step=step)
                        if trial.should_prune():
                            raise optuna.TrialPruned()

                    return clf.score(X_valid, y_valid)

                study = optuna.create_study(direction='maximize')
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
            message = "The `value` argument is of type '{}' but supposed to be a float.".format(
                type(value).__name__
            )
            raise TypeError(message)

        if step < 0:
            raise ValueError("The `step` argument is {} but cannot be negative.".format(step))

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
                "The use of `step` argument is deprecated. "
                "The last reported step is used instead of "
                "the step given by the argument.",
                DeprecationWarning,
            )

        trial = self.study._storage.get_trial(self._trial_id)
        return self.study.pruner.prune(self.study, trial)

    def set_user_attr(self, key, value):
        # type: (str, Any) -> None
        """Set user attributes to the trial.

        The user attributes in the trial can be access via :func:`optuna.trial.Trial.user_attrs`.

        Example:

            Save fixed hyperparameters of neural network training.

            .. testcode::

                import numpy as np
                from sklearn.datasets import load_iris
                from sklearn.model_selection import train_test_split
                from sklearn.neural_network import MLPClassifier

                import optuna

                X, y = load_iris(return_X_y=True)
                X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=0)

                def objective(trial):
                    trial.set_user_attr('BATCHSIZE', 128)
                    momentum = trial.suggest_uniform('momentum', 0, 1.0)
                    clf = MLPClassifier(hidden_layer_sizes=(100, 50),
                                        batch_size=trial.user_attrs['BATCHSIZE'],
                                        momentum=momentum, solver='sgd', random_state=0)
                    clf.fit(X_train, y_train)

                    return clf.score(X_valid, y_valid)

                study = optuna.create_study(direction='maximize')
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
            param_value = self.storage.get_trial_system_attrs(self._trial_id)["fixed_params"][name]
        elif self._is_relative_param(name, distribution):
            param_value = self.relative_params[name]
        else:
            trial = self.storage.get_trial(self._trial_id)

            study = pruners._filter_study(self.study, trial)

            param_value = self.study.sampler.sample_independent(study, trial, name, distribution)

        return self._set_new_param_or_get_existing(name, param_value, distribution)

    def _set_new_param_or_get_existing(self, name, param_value, distribution):
        # type: (str, Any, BaseDistribution) -> Any

        param_value_in_internal_repr = distribution.to_internal_repr(param_value)
        set_success = self.storage.set_trial_param(
            self._trial_id, name, param_value_in_internal_repr, distribution
        )
        if not set_success:
            param_value_in_internal_repr = self.storage.get_trial_param(self._trial_id, name)
            param_value = distribution.to_external_repr(param_value_in_internal_repr)

        return param_value

    def _is_fixed_param(self, name, distribution):
        # type: (str, BaseDistribution) -> bool

        system_attrs = self.storage.get_trial_system_attrs(self._trial_id)
        if "fixed_params" not in system_attrs:
            return False

        if name not in system_attrs["fixed_params"]:
            return False

        param_value = system_attrs["fixed_params"][name]
        param_value_in_internal_repr = distribution.to_internal_repr(param_value)

        contained = distribution._contains(param_value_in_internal_repr)
        if not contained:
            warnings.warn(
                "Fixed parameter '{}' with value {} is out of range "
                "for distribution {}.".format(name, param_value, distribution)
            )
        return contained

    def _is_relative_param(self, name, distribution):
        # type: (str, BaseDistribution) -> bool

        if name not in self.relative_params:
            return False

        if name not in self.relative_search_space:
            raise ValueError(
                "The parameter '{}' was sampled by `sample_relative` method "
                "but it is not contained in the relative search space.".format(name)
            )

        relative_distribution = self.relative_search_space[name]
        distributions.check_distribution_compatibility(relative_distribution, distribution)

        param_value = self.relative_params[name]
        param_value_in_internal_repr = distribution.to_internal_repr(param_value)
        return distribution._contains(param_value_in_internal_repr)

    def _check_distribution(self, name, distribution):
        # type: (str, BaseDistribution) -> None

        old_distribution = self.storage.get_trial(self._trial_id).distributions.get(
            name, distribution
        )
        if old_distribution != distribution:
            warnings.warn(
                'Inconsistent parameter values for distribution with name "{}"! '
                "This might be a configuration mistake. "
                "Optuna allows to call the same distribution with the same "
                "name more then once in a trial. "
                "When the parameter values are inconsistent optuna only "
                "uses the values of the first call and ignores all following. "
                "Using these values: {}".format(name, old_distribution._asdict()),
                RuntimeWarning,
            )

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
            "The use of `Trial.trial_id` is deprecated. Please use `Trial.number` instead.",
            DeprecationWarning,
        )

        self.logger.warning(
            "The use of `Trial.trial_id` is deprecated. Please use `Trial.number` instead."
        )

        return self._trial_id

    @property
    def params(self):
        # type: () -> Dict[str, Any]
        """Return parameters to be optimized.

        Returns:
            A dictionary containing all parameters.
        """

        return copy.deepcopy(self.storage.get_trial_params(self._trial_id))

    @property
    def distributions(self):
        # type: () -> Dict[str, BaseDistribution]
        """Return distributions of parameters to be optimized.

        Returns:
            A dictionary containing all distributions.
        """

        return copy.deepcopy(self.storage.get_trial(self._trial_id).distributions)

    @property
    def user_attrs(self):
        # type: () -> Dict[str, Any]
        """Return user attributes.

        Returns:
            A dictionary containing all user attributes.
        """

        return copy.deepcopy(self.storage.get_trial_user_attrs(self._trial_id))

    @property
    def system_attrs(self):
        # type: () -> Dict[str, Any]
        """Return system attributes.

        Returns:
            A dictionary containing all system attributes.
        """

        return copy.deepcopy(self.storage.get_trial_system_attrs(self._trial_id))

    @property
    def datetime_start(self):
        # type: () -> Optional[datetime.datetime]
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

        message = "The use of `Trial.study_id` is deprecated. Please use `Trial.study` instead."
        warnings.warn(message, DeprecationWarning)
        self.logger.warning(message)

        return self.study._study_id
