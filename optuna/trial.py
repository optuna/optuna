from typing import TYPE_CHECKING  # NOQA

from optuna import distributions

if TYPE_CHECKING:
    from optuna.study import Study  # NOQA
    from typing import Any  # NOQA
    from typing import Dict  # NOQA
    from typing import Optional  # NOQA
    from typing import Sequence  # NOQA
    from typing import TypeVar  # NOQA

    T = TypeVar('T', float, str)


class Trial(object):

    """A trial is a process of evaluating an objective function.

    This object is passed to an objective function, and provides interfaces to get parameter
    suggestion, manage the trial's state, and set/get user-defined attributes of the trial.

    Note that this object is seamlessly instantiated and passed to the objective function behind
    Study.optimize() method (as well as optimize function); hence, in typical use cases,
    library users do not care about instantiation of this object.

    Args:
        study:
            A study object.
        trial_id:
            A trial ID populated by a storage object.

    """

    def __init__(self, study, trial_id):
        # type: (Study, int) -> None

        self.study = study
        self.trial_id = trial_id

        self.study_id = self.study.study_id
        self.storage = self.study.storage

    def suggest_uniform(self, name, low, high):
        # type: (str, float, float) -> float

        """Suggest a parameter value from uniform distribution.

        The interval of the distribution is ``[low, high)``.
        # TODO(Yanase): Add explanation of distribution.

        Example:

            Suggest a dropout rate for a layer of a neural network.

            .. code::

                >>> def objective(trial):
                >>>     ...
                >>>     dropout_rate = trial.suggest_unifrom('dropout_rate', 0, 1.0)
                >>>     layer = chainer.functions.dropout(layer, ratio=dropout_rate)

        Args:
            name:
                A parameter name.
            low:
                Lower endpoint of the interval. ``low`` can be suggested from the distribution.
            high:
                Upper endpoint of the interval. ``high`` is not suggested from the distribution.

        Returns:
            A float value sampled from uniform distribution.
        """

        return self._suggest(name, distributions.UniformDistribution(low=low, high=high))

    def suggest_loguniform(self, name, low, high):
        # type: (str, float, float) -> float

        """Suggest a parameter value from log-scaled uniform distribution.

        The interval of the distribution is ``[low, high)``.
        # TODO(Yanase): Add explanation of distribution.

        Example:

            Suggest learning rate of neural network training.

            .. code::

                >>> def objective(trial):
                >>>     ...
                >>>     lr = trial.suggest_logunifrom('lr', 1e-5, 1e-1)
                >>>     optimizer = chainer.optimizers.MomentumSGD(lr=lr)

        Args:
            name:
                A parameter name.
            low:
                Lower endpoint of the interval. ``low`` can be suggested from the distribution.
            high:
                Upper endpoint of the interval. ``high`` is not suggested from the distribution.

        Returns:
            A float value sampled from log-scaled uniform distribution.
        """

        return self._suggest(name, distributions.LogUniformDistribution(low=low, high=high))

    def suggest_discrete_uniform(self, name, low, high, q):
        # type: (str, float, float, float) -> float

        """Suggest a parameter value from discretized Uniform distribution.

        The interval of the distribution is ``[low, high]``.
        # TODO(Yanase): Add explanation of distribution.

        Example:

            Suggest a gradient-clipping threshold of neural-network training.

            .. code::

                >>> def objective(trial):
                >>>     ...
                >>>     clip = trial.suggest_discrete_uniform('gradient_clipping', 0.5, 5.0, 0.5)
                >>>     optimizer = chainer.optimizers.GradientClipping(threshold=clip)

        Args:
            name:
                A parameter name.
            low:
                Lower endpoint of the interval. ``low`` can be suggested from the distribution.
            high:
                Upper endpoint of the interval. ``high`` can be suggested from the distribution.
            q:
                A quantization step of the distribution.

        Returns:
            A float value sampled from discretized uniform distribution.
        """

        discrete = distributions.DiscreteUniformDistribution(low=low, high=high, q=q)
        return self._suggest(name, discrete)

    def suggest_int(self, name, low, high):
        # type: (str, int, int) -> int

        """Suggest a parameter value from uniform distribution of integer.

        The range of this distribution is ``[low, high]``.
        # TODO(Yanase): Add explanation of distribution.

        Example:

            Suggest the number of layers of multilayer perceptrons.

            .. code::

                >>> def objective(trial):
                >>>     ...
                >>>     n_layers = trial.suggest_int('n_layers', 1, 3)
                >>>     layers = [chainer.links.Linear(None, 128) for _ in range(n_layers)]
                >>>     layers.append(chainer.links.Linear(None, 10))
                >>>     chainer.Sequential(*layers)

        Args:
            name:
                A parameter name.
            low:
                Lower endpoint of the interval. ``low`` can be suggested from the distribution.
            high:
                Upper endpoint of the interval. ``high`` can be suggested from the distribution.

        Returns:
            A integer value sampled from uniform distribution.
        """

        return int(self._suggest(name, distributions.IntUniformDistribution(low=low, high=high)))

    def suggest_categorical(self, name, choices):
        # type: (str, Sequence[T]) -> T

        """Suggest a parameter value from categorical distribution.

        The parameter value is chosen from ``choices``.
        # TODO(Yanase): Add explanation of distribution.

        Example:

            Suggest optimizers of neural network training.

            .. code::

                >>> def objective(trial):
                >>>     ...
                >>>     name = trial.suggest_categorical('optimizer', ['Adam', 'MomentumSGD'])
                >>>     if name == 'Adam':
                >>>         optimizer = chainer.optimizers.Adam()
                >>>     else:
                >>>         optimizer = chainer.optimizers.MomentumSGD()

        Args:
            name:
                A parameter name.
            choices:
                Candidates of parameter values.

        Returns:
            A value sampled from categorical distribution.
        """

        choices = tuple(choices)
        return self._suggest(name, distributions.CategoricalDistribution(choices=choices))

    def report(self, value, step=None):
        # type: (float, Optional[int]) -> None

        """Report objective function value to storage.

        If step is set to None, the value is stored to storage as a final value of the trial.
        Otherwise, it is saved as an intermediate value.

        Example:

            Report intermediate scores of SGDClassifier training

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
                Step of the trial (e.g., Epoch of neural-network training).
        """

        self.storage.set_trial_value(self.trial_id, value)
        if step is not None:
            self.storage.set_trial_intermediate_value(self.trial_id, step, value)

    def should_prune(self, step):
        # type: (int) -> bool

        """Judge whether the trial should be pruned.

        This method calls prune method of the pruner, which judges whether the trial should
        be pruned at the given step. Please refer to the example code of :method:`Trial.report`.

        Args:
            step:
                Step of the trial (e.g., epoch of neural-network training).

        Returns:
            A boolean value. If True, the trial should be pruned. Otherwise, the trial will be
            continued.
        """

        # TODO(akiba): remove `step` argument

        return self.study.pruner.prune(
            self.storage, self.study_id, self.trial_id, step)

    def set_user_attr(self, key, value):
        # type: (str, Any) -> None

        """Set user attributes to the trial.

        The user attributes in the trial can be access via :method:`Trial.user_attrs`.

        Example:

            Save fixed hyperparameters of neural-network training:

            .. code::

                >>> BATCHSIZE = 128
                >>>
                >>> def objective(trial):
                >>>     ...
                >>>     trial.set_user_attr('BATCHSIZE', BATCHSIZE)
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
        reason of trials. Please use :method:`Trial.set_user_attr` to set users'
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

        param_value_in_internal_repr = self.study.sampler.sample(
            self.storage, self.study_id, name, distribution)

        set_success = self.storage.set_trial_param(
            self.trial_id, name, param_value_in_internal_repr, distribution)
        if not set_success:
            param_value_in_internal_repr = self.storage.get_trial_param(self.trial_id, name)

        param_value = distribution.to_external_repr(param_value_in_internal_repr)
        return param_value

    @property
    def params(self):
        # type: () -> Dict[str, Any]

        """

        A dictionary of parameters to be optimized.

        """

        return self.storage.get_trial_params(self.trial_id)

    @property
    def user_attrs(self):
        # type: () -> Dict[str, Any]

        """

        A dictionary of user attributes.

        """
        return self.storage.get_trial_user_attrs(self.trial_id)

    @property
    def system_attrs(self):
        # type: () -> Dict[str, Any]

        """

        A dictionary of system attributes.

        """

        return self.storage.get_trial_system_attrs(self.trial_id)
