from typing import Any
import warnings

import lightgbm as lgb


class LGBMModel(lgb.LGBMModel):
    """Proxy of lightgbm.LGBMModel.

    See: `pydoc lightgbm.LGBMModel`
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            "LightGBMTuner doesn't support sklearn API. "
            "Use `train()` or `LightGBMTuner` for hyperparameter tuning."
        )
        super().__init__(*args, **kwargs)


class LGBMClassifier(lgb.LGBMClassifier):
    """Proxy of lightgbm.LGBMClassifier.

    See: `pydoc lightgbm.LGBMClassifier`
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            "LightGBMTuner doesn't support sklearn API. "
            "Use `train()` or `LightGBMTuner` for hyperparameter tuning."
        )
        super().__init__(*args, **kwargs)


class LGBMRegressor(lgb.LGBMRegressor):
    """Proxy of LGBMRegressor.

    See: `pydoc lightgbm.LGBMRegressor`
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            "LightGBMTuner doesn't support sklearn API. "
            "Use `train()` or `LightGBMTuner` for hyperparameter tuning."
        )
        super().__init__(*args, **kwargs)
