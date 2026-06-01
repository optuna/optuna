from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from optuna.visualization._rank import _convert_color_idxs_to_scaled_rgb_colors


def test_convert_color_idxs_raises_when_no_backend_available() -> None:
    with patch("optuna.visualization._rank.plotly_is_available", False), patch(
        "optuna.visualization._rank.matplotlib_imports.is_successful", return_value=False
    ):
        with pytest.raises(ImportError, match="Neither plotly nor matplotlib"):
            _convert_color_idxs_to_scaled_rgb_colors(np.array([0.1, 0.5]))
