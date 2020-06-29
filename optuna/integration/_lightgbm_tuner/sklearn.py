import warnings

import lightgbm as lgb

from optuna import type_checking

from typing import Any
from typing import Dict
from typing import List


class LGBMModel(lgb.LGBMModel):
    """Proxy of lightgbm.LGBMModel.

    See: `pydoc lightgbm.LGBMModel`
    """

    def __init__(self, *args: List[Any], **kwargs: Dict[str, Any]) -> None:

        warnings.warn(
            "LightGBMTuner doesn't support sklearn API. "
            "Use `train()` or `LightGBMTuner` for hyperparameter tuning."
        )
        super(LGBMModel, self).__init__(*args, **kwargs)


class LGBMClassifier(lgb.LGBMClassifier):
    """Proxy of lightgbm.LGBMClassifier.

    See: `pydoc lightgbm.LGBMClassifier`
    """

    def __init__(self, *args: List[Any], **kwargs: Dict[str, Any]) -> None:

        warnings.warn(
            "LightGBMTuner doesn't support sklearn API. "
            "Use `train()` or `LightGBMTuner` for hyperparameter tuning."
        )
        super(LGBMClassifier, self).__init__(*args, **kwargs)


class LGBMRegressor(lgb.LGBMRegressor):
    """Proxy of LGBMRegressor.

    See: `pydoc lightgbm.LGBMRegressor`
    """

    def __init__(self, *args: List[Any], **kwargs: Dict[str, Any]) -> None:

        warnings.warn(
            "LightGBMTuner doesn't support sklearn API. "
            "Use `train()` or `LightGBMTuner` for hyperparameter tuning."
        )
        super(LGBMRegressor, self).__init__(*args, **kwargs)
