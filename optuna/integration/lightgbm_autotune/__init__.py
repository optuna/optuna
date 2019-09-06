from optuna import type_checking
from optuna.integration.lightgbm_autotune.sklearn import LGBMClassifier, LGBMModel, LGBMRegressor  # NOQA
from optuna.integration.lightgbm_autotune.optimize import LGBMAutoTune


if type_checking.TYPE_CHECKING:
    from type_checking import Any  # NOQA
    from type_checking import Optional  # NOQA


def train(*args, **kwargs):
    auto_booster = LGBMAutoTune(*args, **kwargs)
    booster = auto_booster.run()
    return booster
