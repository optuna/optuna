from __future__ import absolute_import

import pfnopt

try:
    import xgboost as xgb  # NOQA
    _available = True
except ImportError as e:
    _import_error = e
    # XGBoostPruningCallback is disabled because XGBoost is not available.
    _available = False


class XGBoostPruningCallback(object):

    def __init__(self, trial, observation_key):
        # type: (pfnopt.trial.Trial, str) -> None

        _check_xgboost_availability()

        self.trial = trial
        self.observation_key = observation_key

    def __call__(self, env):
        # type: (xgb.core.CallbackEnv) -> None

        current_score = dict(env.evaluation_result_list)[self.observation_key]
        self.trial.report(current_score, step=env.iteration)
        if self.trial.should_prune(env.iteration):
            message = "Trial was pruned at iteration {}.".format(env.iteration)
            raise pfnopt.structs.TrialPruned(message)


def _check_xgboost_availability():
    # type: () -> None

    if not _available:
        raise ImportError(
            'XGBoost is not available. Please install XGBoost to use this feature. '
            'XGBoost can be installed by executing `$ pip install xgboost`. '
            'For further information, please refer to the installation guide of XGBoost. '
            '(The actual import error is as follows: ' + str(_import_error) + ')')
