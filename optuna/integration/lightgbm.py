import sys

import optuna
from optuna._imports import try_import
from optuna.integration import _lightgbm_tuner as tuner


with try_import() as _imports:
    import lightgbm as lgb  # NOQA
    from lightgbm.callback import CallbackEnv  # NOQA

# Attach lightgbm API.
if _imports.is_successful():
    # To pass tests/integration_tests/lightgbm_tuner_tests/test_optimize.py.
    from lightgbm import Dataset  # NOQA

    from optuna.integration._lightgbm_tuner import LightGBMTuner  # NOQA
    from optuna.integration._lightgbm_tuner import LightGBMTunerCV  # NOQA

    _names_from_tuners = ["train", "LGBMModel", "LGBMClassifier", "LGBMRegressor"]

    # API from lightgbm.
    for api_name in lgb.__dict__["__all__"]:
        if api_name in _names_from_tuners:
            continue
        setattr(sys.modules[__name__], api_name, lgb.__dict__[api_name])

    # API from lightgbm_tuner.
    for api_name in _names_from_tuners:
        setattr(sys.modules[__name__], api_name, tuner.__dict__[api_name])
else:
    # To create docstring of train.
    setattr(sys.modules[__name__], "train", tuner.__dict__["train"])
    setattr(sys.modules[__name__], "LightGBMTuner", tuner.__dict__["LightGBMTuner"])
    setattr(sys.modules[__name__], "LightGBMTunerCV", tuner.__dict__["LightGBMTunerCV"])


class LightGBMPruningCallback(object):
    """Callback for LightGBM to prune unpromising trials.

    See `the example <https://github.com/optuna/optuna/blob/master/
    examples/pruning/lightgbm_integration.py>`__
    if you want to add a pruning callback which observes AUC
    of a LightGBM model.

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

    def __init__(
        self, trial: optuna.trial.Trial, metric: str, valid_name: str = "valid_0"
    ) -> None:

        _imports.check()

        self._trial = trial
        self._valid_name = valid_name
        self._metric = metric

    def __call__(self, env: "CallbackEnv") -> None:

        # If this callback has been passed to `lightgbm.cv` function,
        # the value of `is_cv` becomes `True`. See also:
        # https://github.com/Microsoft/LightGBM/blob/v2.2.2/python-package/lightgbm/engine.py#L329
        # Note that `5` is not the number of folds but the length of sequence.
        is_cv = len(env.evaluation_result_list) > 0 and len(env.evaluation_result_list[0]) == 5
        if is_cv:
            target_valid_name = "cv_agg"
        else:
            target_valid_name = self._valid_name

        for evaluation_result in env.evaluation_result_list:
            valid_name, metric, current_score, is_higher_better = evaluation_result[:4]
            if valid_name != target_valid_name or metric != self._metric:
                continue

            if is_higher_better:
                if self._trial.study.direction != optuna.study.StudyDirection.MAXIMIZE:
                    raise ValueError(
                        "The intermediate values are inconsistent with the objective values in "
                        "terms of study directions. Please specify a metric to be minimized for "
                        "LightGBMPruningCallback."
                    )
            else:
                if self._trial.study.direction != optuna.study.StudyDirection.MINIMIZE:
                    raise ValueError(
                        "The intermediate values are inconsistent with the objective values in "
                        "terms of study directions. Please specify a metric to be maximized for "
                        "LightGBMPruningCallback."
                    )

            self._trial.report(current_score, step=env.iteration)
            if self._trial.should_prune():
                message = "Trial was pruned at iteration {}.".format(env.iteration)
                raise optuna.TrialPruned(message)

            return None

        raise ValueError(
            'The entry associated with the validation name "{}" and the metric name "{}" '
            "is not found in the evaluation result list {}.".format(
                target_valid_name, self._metric, str(env.evaluation_result_list)
            )
        )
