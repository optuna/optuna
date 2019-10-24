from optuna.integration.lightgbm_tuner.sklearn import LGBMClassifier, LGBMModel, LGBMRegressor  # NOQA
from optuna.integration.lightgbm_tuner.optimize import LightGBMTuner
from optuna import type_checking

if type_checking.TYPE_CHECKING:
    from type_checking import Any  # NOQA
    from type_checking import Dict  # NOQA
    from type_checking import List  # NOQA
    from type_checking import Optional  # NOQA


def train(*args, **kwargs):
    # type: (List[Any], Optional[Dict[Any, Any]]) -> Any
    """Wrapper function of LightGBM API: train()

    Arguments and keyword arguments for `lightgbm.train()` can be passed.
    """

    auto_booster = LightGBMTuner(*args, **kwargs)
    booster = auto_booster.run()
    return booster
