import warnings

import lightgbm as lgb

from optuna import type_checking


if type_checking.TYPE_CHECKING:
    from type_checking import Any  # NOQA
    from type_checking import Callable  # NOQA
    from type_checking import Dict  # NOQA
    from type_checking import List  # NOQA
    from type_checking import Optional  # NOQA
    from type_checking import Tuple  # NOQA
    from type_checking import Union  # NOQA

    import numpy as np  # NOQA
    from scipy.sparse.compressed import _cs_matrix  # NOQA


class LGBMModel(lgb.LGBMModel):
    """Proxy of lightgbm.LGBMModel.

    See: `pydoc lightgbm.LGBMModel`
    """

    def __init__(self, *args, **kwargs):
        # type: (List[Any], Dict[str, Any]) -> None

        warnings.warn("LightGBMTuner doesn't support sklearn API. "
                      "Use `train()` or `LightGBMTuner` for hyperparameter tuning.")
        super(LGBMModel, self).__init__(*args, **kwargs)


class LGBMClassifier(lgb.LGBMClassifier):
    """Proxy of lightgbm.LGBMClassifier.

    See: `pydoc lightgbm.LGBMClassifier`
    """

    def __init__(self, *args, **kwargs):
        # type: (List[Any], Dict[str, Any]) -> None

        warnings.warn("LightGBMTuner doesn't support sklearn API. "
                      "Use `train()` or `LightGBMTuner` for hyperparameter tuning.")
        super(LGBMClassifier, self).__init__(*args, **kwargs)


class LGBMRegressor(lgb.LGBMRegressor):
    """Proxy of LGBMRegressor.

    See: `pydoc lightgbm.LGBMRegressor`
    """

    def __init__(self, *args, **kwargs):
        # type: (List[Any], Dict[str, Any]) -> None

        warnings.warn("LightGBMTuner doesn't support sklearn API. "
                      "Use `train()` or `LightGBMTuner` for hyperparameter tuning.")
        super(LGBMRegressor, self).__init__(*args, **kwargs)
