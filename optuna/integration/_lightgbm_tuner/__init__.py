from typing import Any

from optuna.integration._lightgbm_tuner.optimize import _imports
from optuna.integration._lightgbm_tuner.optimize import LightGBMTuner
from optuna.integration._lightgbm_tuner.optimize import LightGBMTunerCV


if _imports.is_successful():
    from optuna.integration._lightgbm_tuner.sklearn import LGBMClassifier
    from optuna.integration._lightgbm_tuner.sklearn import LGBMModel
    from optuna.integration._lightgbm_tuner.sklearn import LGBMRegressor

__all__ = ["LightGBMTuner", "LightGBMTunerCV", "LGBMClassifier", "LGBMModel", "LGBMRegressor"]


def train(*args: Any, **kwargs: Any) -> Any:
    """Wrapper of LightGBM Training API to tune hyperparameters.

    It tunes important hyperparameters (e.g., ``min_child_samples`` and ``feature_fraction``) in a
    stepwise manner. It is a drop-in replacement for `lightgbm.train()`_. See
    `a simple example of LightGBM Tuner <https://github.com/optuna/optuna-examples/tree/main/
    lightgbm/lightgbm_tuner_simple.py>`_ which optimizes the validation log loss of cancer
    detection.

    :func:`~optuna.integration.lightgbm.train` is a wrapper function of
    :class:`~optuna.integration.lightgbm.LightGBMTuner`. To use feature in Optuna such as
    suspended/resumed optimization and/or parallelization, refer to
    :class:`~optuna.integration.lightgbm.LightGBMTuner` instead of this function.

    Arguments and keyword arguments for `lightgbm.train()`_ can be passed.

    .. _lightgbm.train(): https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.train.html
    """
    _imports.check()

    auto_booster = LightGBMTuner(*args, **kwargs)
    auto_booster.run()
    return auto_booster.get_best_booster()
