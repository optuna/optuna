from unittest.mock import Mock
from unittest.mock import patch

import numpy as np
import pytest

import optuna
import optuna.multi_objective
from optuna.multi_objective.samplers._carbs import CARBSSampler

