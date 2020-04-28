import optuna

try:
    import xgboost as xgb  # NOQA

    _available = True
except ImportError as e:
    _import_error = e
    # XGBoostPruningCallback is disabled because XGBoost is not available.
    _available = False


def _get_callback_context(env):
    # type: (xgb.core.CallbackEnv) -> str
    """Return whether the current callback context is cv or train.

    .. note::
        `Reference
        <https://github.com/dmlc/xgboost/blob/master/python-package/xgboost/callback.py>`_.
    """

    if env.model is None and env.cvfolds is not None:
        context = "cv"
    else:
        context = "train"
    return context


class XGBoostPruningCallback(object):
    """Callback for XGBoost to prune unpromising trials.

    See `the example <https://github.com/optuna/optuna/blob/master/
    examples/pruning/xgboost_integration.py>`__
    if you want to add a pruning callback which observes validation AUC of
    a XGBoost model.

    Args:
        trial:
            A :class:`~optuna.trial.Trial` corresponding to the current evaluation of the
            objective function.
        observation_key:
            An evaluation metric for pruning, e.g., ``validation-error`` and
            ``validation-merror``. Please refer to ``eval_metric`` in
            `XGBoost reference <https://xgboost.readthedocs.io/en/latest/parameter.html>`_
            for further details.
        interval:
            Check if trial should be pruned every n-th iteration. By default `interval=1` and
            pruning is performed after every iteration. Increase `interval` to run several
            iterations faster before applying pruning.
    """

    def __init__(self, trial, observation_key, interval=1):
        # type: (optuna.trial.Trial, str, int) -> None

        _check_xgboost_availability()

        self._trial = trial
        self._observation_key = observation_key
        self._interval = interval

    def __call__(self, env):
        # type: (xgb.core.CallbackEnv) -> None

        context = _get_callback_context(env)

        if (env.iteration + 1) % self._interval != 0:
            return

        evaluation_result_list = env.evaluation_result_list
        if context == "cv":
            # Remove a third element: the stddev of the metric across the cross-valdation folds.
            evaluation_result_list = [(key, metric) for key, metric, _ in evaluation_result_list]
        current_score = dict(evaluation_result_list)[self._observation_key]
        self._trial.report(current_score, step=env.iteration)
        if self._trial.should_prune():
            message = "Trial was pruned at iteration {}.".format(env.iteration)
            raise optuna.exceptions.TrialPruned(message)


def _check_xgboost_availability():
    # type: () -> None

    if not _available:
        raise ImportError(
            "XGBoost is not available. Please install XGBoost to use this feature. "
            "XGBoost can be installed by executing `$ pip install xgboost`. "
            "For further information, please refer to the installation guide of XGBoost. "
            "(The actual import error is as follows: " + str(_import_error) + ")"
        )
