import abc
import six

import optuna
from optuna.distributions import BaseDistribution  # NOQA
from optuna.structs import FrozenTrial  # NOQA
from optuna import types

if types.TYPE_CHECKING:
    from optuna.study import RunningStudy  # NOQA
    from typing import Dict  # NOQA


@six.add_metaclass(abc.ABCMeta)
class BaseSampler(object):
    """Base class for samplers."""

    def define_relative_search_space(self, trial):
        # type: (FrozenTrial) -> Dict[str, BaseDistribution]
        """TODO: Add doc"""

        return self.study.full_search_space

    def sample_relative(self, trial, search_space):
        # type: (FrozenTrial, Dict[str, BaseDistribution]) -> Dict[str, float]
        """TODO: Add doc"""

        return {}

    def sample_independent(self, trial, param_name, param_distribution):
        # type: (FrozenTrial, str, BaseDistribution) -> float
        """TODO: Add doc"""

        tpe = optuna.samplers.TPESampler()
        tpe._set_study(self.study)
        return tpe.sample_independent(trial, param_name, param_distribution)

    @property
    def study(self):
        # type: () -> RunningStudy
        """Return the target study."""

        if not hasattr(self, '_study'):
            raise RuntimeError('`_study` field has not yet been set.')

        return self._study

    def _set_study(self, study):
        # type: (RunningStudy) -> None

        self._study = study
