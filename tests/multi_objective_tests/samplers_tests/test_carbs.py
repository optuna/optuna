from typing import Dict
from typing import Tuple
from unittest.mock import Mock
from unittest.mock import patch

import pytest

import optuna
from optuna import multi_objective
from optuna.multi_objective.samplers import CARBSSampler