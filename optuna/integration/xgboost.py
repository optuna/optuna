from __future__ import absolute_import

import optuna

try:
    import xgboost as xgb  # NOQA
    _available = True
except ImportError as e:
    _import_error = e
    # XGBoostPruningCallback is disabled because XGBoost is not available.
    _available = False


class XGBoostPruningCallback(object):
    """Callback for XGBoost to prune unpromising trials.

    Example:

        Add a pruning callback which observes validation errors to training of an XGBoost model.

        .. code::

                pruning_callback = XGBoostPruningCallback(trial, 'validation-error')
                bst = xgb.train(param, dtrain, evals=[(dtest, 'validation')],
                                callbacks=[pruning_callback])


    Args:
        trial:
            A :class:`~optuna.trial.Trial` corresponding to the current evaluation of the
            objective function.
        observation_key:
            An evaluation metric for pruning, e.g., ``validation-error`` and
            ``validation-merror``. Please refer to ``eval_metric`` in
            `XGBoost reference <https://xgboost.readthedocs.io/en/latest/parameter.html>`_
            for further details.
    """

    def __init__(self, trial, observation_key):
        # type: (optuna.trial.Trial, str) -> None

        _check_xgboost_availability()

        self.trial = trial
        self.observation_key = observation_key

    def __call__(self, env):
        # type: (xgb.core.CallbackEnv) -> None

        current_score = dict(env.evaluation_result_list)[self.observation_key]
        self.trial.report(current_score, step=env.iteration)
        if self.trial.should_prune():
            message = "Trial was pruned at iteration {}.".format(env.iteration)
            raise optuna.structs.TrialPruned(message)


def _check_xgboost_availability():
    # type: () -> None

    if not _available:
        raise ImportError(
            'XGBoost is not available. Please install XGBoost to use this feature. '
            'XGBoost can be installed by executing `$ pip install xgboost`. '
            'For further information, please refer to the installation guide of XGBoost. '
            '(The actual import error is as follows: ' + str(_import_error) + ')')
