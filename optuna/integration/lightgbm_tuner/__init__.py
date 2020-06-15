from typing import Any

from optuna._experimental import experimental
from optuna.integration.lightgbm_tuner._optimize import _imports as _imports
from optuna.integration.lightgbm_tuner._optimize import LightGBMTuner
from optuna.integration.lightgbm_tuner._optimize import LightGBMTunerCV  # NOQA
from optuna import type_checking

if _imports.is_successful():
    from optuna.integration.lightgbm_tuner._sklearn import LGBMClassifier  # NOQA
    from optuna.integration.lightgbm_tuner._sklearn import LGBMModel  # NOQA
    from optuna.integration.lightgbm_tuner._sklearn import LGBMRegressor  # NOQA

if type_checking.TYPE_CHECKING:
    from typing import Any  # NOQA


@experimental("0.18.0")
def train(*args: Any, **kwargs: Any) -> Any:
    """Wrapper of LightGBM Training API to tune hyperparameters.

    It tunes important hyperparameters (e.g., ``min_child_samples`` and ``feature_fraction``) in a
    stepwise manner. It is a drop-in replacement for `lightgbm.train()`_. See
    `a simple example of LightGBM Tuner <https://github.com/optuna/optuna/blob/master/examples/lig
    htgbm_tuner_simple.py>`_ which optimizes the validation log loss of cancer detection.

    :func:`~optuna.integration.lightgbm.train` is a wrapper function of
    :class:`~optuna.integration.lightgbm_tuner.LightGBMTuner`. To use feature in Optuna such as
    suspended/resumed optimization and/or parallelization, refer to
    :class:`~optuna.integration.lightgbm_tuner.LightGBMTuner` instead of this function.

    Arguments and keyword arguments for `lightgbm.train()`_ can be passed.

    .. _lightgbm.train(): https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.train.html
    """
    _imports.check()

    auto_booster = LightGBMTuner(*args, **kwargs)
    auto_booster.run()
    return auto_booster.get_best_booster()
