from __future__ import absolute_import
import sys

import optuna


try:
    import lightgbm as lgb
    _available = True
except ImportError as e:
    _import_error = e
    # LightGBMPruningCallback is disabled because LightGBM is not available.
    _available = False


# Attach lightgbm API
if _available:
    # API from lightgbm
    for api_name in lgb.__dict__['__all__']:
        if api_name in ['train', 'LGBMModel', 'LGBMClassifier', 'LGBMRegressor']:
            continue
        setattr(sys.modules[__name__], api_name, lgb.__dict__[api_name])

    # API from optuna integration
    import optuna.integration.lightgbm_autotune as autotune
    for api_name in ['train', 'LGBMModel', 'LGBMClassifier', 'LGBMRegressor', 'LGBMAutoTune']:
        setattr(sys.modules[__name__], api_name, autotune.__dict__[api_name])


class LightGBMPruningCallback(object):
    """Callback for LightGBM to prune unpromising trials.

    Example:

        Add a pruning callback which observes validation scores to training of a LightGBM model.

        .. code::

                param = {'objective': 'binary', 'metric': 'binary_error'}
                pruning_callback = LightGBMPruningCallback(trial, 'binary_error')
                gbm = lgb.train(param, dtrain, valid_sets=[dtest], callbacks=[pruning_callback])

    Args:
        trial:
            A :class:`~optuna.trial.Trial` corresponding to the current evaluation of
            the objective function.
        metric:
            An evaluation metric for pruning, e.g., ``binary_error`` and ``multi_error``.
            Please refer to
            `LightGBM reference
            <https://lightgbm.readthedocs.io/en/latest/Parameters.html#metric>`_
            for further details.
        valid_name:
            The name of the target validation.
            Validation names are specified by ``valid_names`` option of
            `train method
            <https://lightgbm.readthedocs.io/en/latest/Python-API.html#lightgbm.train>`_.
            If omitted, ``valid_0`` is used which is the default name of the first validation.
            Note that this argument will be ignored if you are calling
            `cv method <https://lightgbm.readthedocs.io/en/latest/Python-API.html#lightgbm.cv>`_
            instead of train method.
    """

    def __init__(self, trial, metric, valid_name='valid_0'):
        # type: (optuna.trial.Trial, str, str) -> None

        _check_lightgbm_availability()

        self.trial = trial
        self.valid_name = valid_name
        self.metric = metric

    def __call__(self, env):
        # type: (lgb.callback.CallbackEnv) -> None

        # If this callback has been passed to `lightgbm.cv` function,
        # the value of `is_cv` becomes `True`. See also:
        # https://github.com/Microsoft/LightGBM/blob/v2.2.2/python-package/lightgbm/engine.py#L329
        # Note that `5` is not the number of folds but the length of sequence.
        is_cv = len(env.evaluation_result_list) > 0 and len(env.evaluation_result_list[0]) == 5
        if is_cv:
            target_valid_name = 'cv_agg'
        else:
            target_valid_name = self.valid_name

        for evaluation_result in env.evaluation_result_list:
            valid_name, metric, current_score, is_higher_better = evaluation_result[:4]
            if valid_name != target_valid_name or metric != self.metric:
                continue

            if is_higher_better:
                if self.trial.storage.get_study_direction(self.trial.study_id) != \
                        optuna.structs.StudyDirection.MAXIMIZE:
                    raise ValueError(
                        "The intermediate values are inconsistent with the objective values in "
                        "terms of study directions. Please specify a metric to be minimized for "
                        "LightGBMPruningCallback.")
            else:
                if self.trial.storage.get_study_direction(self.trial.study_id) != \
                        optuna.structs.StudyDirection.MINIMIZE:
                    raise ValueError(
                        "The intermediate values are inconsistent with the objective values in "
                        "terms of study directions. Please specify a metric to be maximized for "
                        "LightGBMPruningCallback.")

            self.trial.report(current_score, step=env.iteration)
            if self.trial.should_prune():
                message = "Trial was pruned at iteration {}.".format(env.iteration)
                raise optuna.structs.TrialPruned(message)

            return None

        raise ValueError(
            'The entry associated with the validation name "{}" and the metric name "{}" '
            'is not found in the evaluation result list {}.'.format(
                target_valid_name, self.metric, str(env.evaluation_result_list)))


def _check_lightgbm_availability():
    # type: () -> None

    if not _available:
        raise ImportError(
            'LightGBM is not available. Please install LightGBM to use this feature. '
            'LightGBM can be installed by executing `$ pip install lightgbm`. '
            'For further information, please refer to the installation guide of LightGBM. '
            '(The actual import error is as follows: ' + str(_import_error) + ')')
