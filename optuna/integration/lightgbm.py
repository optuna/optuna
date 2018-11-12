from __future__ import absolute_import

import optuna

try:
    import lightgbm as lgb  # NOQA
    _available = True
except ImportError as e:
    _import_error = e
    # LightGBMPruningCallback is disabled because LightGBM is not available.
    _available = False


class LightGBMPruningCallback(object):

    """Callback for LightGBM to prune unpromising trials.

    Example:

        Add a pruning callback which observes validation scores to training of a LightGBM model.

        .. code::

                param = {'objective': 'binary', 'metric': 'binary_logloss'}
                pruning_callback = LightGBMPruningCallback(trial, 'validation-binary_logloss')
                gbm = lgb.train(param, dtrain, num_round,
                                valid_sets=[dtest], valid_names=['validation'],
                                callbacks=[pruning_callback])

    Args:
        trial:
            A trial corresponding to the current evaluation of the objective function.
        observation_key:
            A key used for identifying evaluation metric for pruning.
            This key consists of ``"{VALIDATION_NAME}-{METRIC_NAME}"`` where
            ``{VALIDATION_NAME}`` is the name specified by ``valid_names`` option of
            `train method
            <https://lightgbm.readthedocs.io/en/latest/Python-API.html#lightgbm.train>`_
            and ``{METRIC_NAME}`` is the name of the
            `metric <https://lightgbm.readthedocs.io/en/latest/Parameters.html#metric>`_
            used for pruning.
    """

    def __init__(self, trial, observation_key):
        # type: (optuna.trial.Trial, str) -> None

        _check_lightgbm_availability()

        self.trial = trial
        self.observation_key = observation_key

    def __call__(self, env):
        # type: (lgb.callback.CallbackEnv) -> None

        for valid_name, metric_name, current_score, is_higher_better in env.evaluation_result_list:
            key = '{}-{}'.format(valid_name, metric_name)
            if key == self.observation_key:
                if is_higher_better:
                    current_score = 1 - current_score

                self.trial.report(current_score, step=env.iteration)
                if self.trial.should_prune(env.iteration):
                    message = "Trial was pruned at iteration {}.".format(env.iteration)
                    raise optuna.structs.TrialPruned(message)
                return None

        raise ValueError(
            'The entry associated to the observation key "' + self.observation_key + '" '
            'is not found in the evaluation result list ' + str(env.evaluation_result_list))


def _check_lightgbm_availability():
    # type: () -> None

    if not _available:
        raise ImportError(
            'LightGBM is not available. Please install LightGBM to use this feature. '
            'LightGBM can be installed by executing `$ pip install lightgbm`. '
            'For further information, please refer to the installation guide of LightGBM. '
            '(The actual import error is as follows: ' + str(_import_error) + ')')
