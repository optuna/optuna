from collections import OrderedDict
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

import numpy

from optuna._imports import try_import
from optuna._transform import _SearchSpaceTransform
from optuna.importance._base import _get_distributions
from optuna.importance._base import BaseImportanceEvaluator
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
from optuna.visualization._utils import _filter_nonfinite


with try_import() as _imports:
    from sklearn.ensemble import RandomForestRegressor


class MeanDecreaseImpurityImportanceEvaluator(BaseImportanceEvaluator):
    """Mean Decrease Impurity (MDI) parameter importance evaluator.

    This evaluator fits a random forest that predicts objective values given hyperparameter
    configurations. Feature importances are then computed using MDI.

    .. note::

        This evaluator requires the `sklean <https://scikit-learn.org/stable/>`_ Python package and
        is based on `sklearn.ensemble.RandomForestClassifier.feature_importances_
        <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier.feature_importances_>`_.

    Args:
        n_trees:
            Number of trees in the random forest.
        max_depth:
            The maximum depth of each tree in the random forest.
        seed:
            Seed for the random forest.
    """

    def __init__(
        self, *, n_trees: int = 64, max_depth: int = 64, seed: Optional[int] = None
    ) -> None:
        _imports.check()

        self._forest = RandomForestRegressor(
            n_estimators=n_trees,
            max_depth=max_depth,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=seed,
        )
        self._trans_params = numpy.empty(0)
        self._trans_values = numpy.empty(0)
        self._param_names: List[str] = list()

    def evaluate(
        self,
        study: Study,
        params: Optional[List[str]] = None,
        *,
        target: Optional[Callable[[FrozenTrial], float]] = None,
    ) -> Dict[str, float]:
        if target is None and study._is_multi_objective():
            raise ValueError(
                "If the `study` is being used for multi-objective optimization, "
                "please specify the `target`. For example, use "
                "`target=lambda t: t.values[0]` for the first objective value."
            )

        distributions = _get_distributions(study, params)
        if len(distributions) == 0:
            return OrderedDict()

        trials = []
        for trial in _filter_nonfinite(
            study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)), target=target
        ):
            if any(name not in trial.params for name in distributions.keys()):
                continue
            trials.append(trial)

        trans = _SearchSpaceTransform(distributions, transform_log=False, transform_step=False)

        n_trials = len(trials)
        self._trans_params = numpy.empty((n_trials, trans.bounds.shape[0]), dtype=numpy.float64)
        self._trans_values = numpy.empty(n_trials, dtype=numpy.float64)

        for trial_idx, trial in enumerate(trials):
            self._trans_params[trial_idx] = trans.transform(trial.params)
            self._trans_values[trial_idx] = trial.value if target is None else target(trial)

        encoded_column_to_column = trans.encoded_column_to_column

        if self._trans_params.size == 0:  # `params` were given but as an empty list.
            return OrderedDict()

        forest = self._forest
        forest.fit(self._trans_params, self._trans_values)
        feature_importances = forest.feature_importances_
        feature_importances_reduced = numpy.zeros(len(distributions))
        numpy.add.at(feature_importances_reduced, encoded_column_to_column, feature_importances)

        param_importances = OrderedDict()
        self._param_names = list(distributions.keys())
        for i in feature_importances_reduced.argsort()[::-1]:
            param_importances[self._param_names[i]] = feature_importances_reduced[i].item()

        return param_importances
