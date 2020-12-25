from collections import OrderedDict
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

import numpy

from optuna._transform import _SearchSpaceTransform
from optuna.importance._base import _get_distributions
from optuna.importance._base import BaseImportanceEvaluator
from optuna.importance._fanova._fanova import _Fanova
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


class FanovaImportanceEvaluator(BaseImportanceEvaluator):
    """fANOVA importance evaluator.

    Implements the fANOVA hyperparameter importance evaluation algorithm in
    `An Efficient Approach for Assessing Hyperparameter Importance
    <http://proceedings.mlr.press/v32/hutter14.html>`_.

    Given a study, fANOVA fits a random forest regression model that predicts the objective value
    given a parameter configuration. The more accurate this model is, the more reliable the
    importances assessed by this class are.

    .. note::

        Requires the `sklearn <https://github.com/scikit-learn/scikit-learn>`_ Python package.

    .. note::

        Pairwise and higher order importances are not supported through this class. They can be
        computed using :class:`~optuna.importance._fanova._fanova._Fanova` directly but is not
        recommended as interfaces may change without prior notice.

    .. note::

        The performance of fANOVA depends on the prediction performance of the underlying
        random forest model. In order to obtain high prediction performance, it is necessary to
        cover a wide range of the hyperparameter search space. It is recommended to use an
        exploration-oriented sampler such as :class:`~optuna.samplers.RandomSampler`.

    .. note::

        For how to cite the original work, please refer to
        https://automl.github.io/fanova/cite.html.

    Args:
        n_trees:
            The number of trees in the forest.
        max_depth:
            The maximum depth of the trees in the forest.
        seed:
            Controls the randomness of the forest. For deterministic behavior, specify a value
            other than :obj:`None`.

    """

    def __init__(
        self, *, n_trees: int = 64, max_depth: int = 64, seed: Optional[int] = None
    ) -> None:
        self._evaluator = _Fanova(
            n_trees=n_trees,
            max_depth=max_depth,
            min_samples_split=2,
            min_samples_leaf=1,
            seed=seed,
        )

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
                "please specify the `target`."
            )

        distributions = _get_distributions(study, params)
        if len(distributions) == 0:
            return OrderedDict()

        trials = []
        for trial in study.trials:
            if trial.state != TrialState.COMPLETE:
                continue
            if any(name not in trial.params for name in distributions.keys()):
                continue
            trials.append(trial)

        trans = _SearchSpaceTransform(distributions, transform_log=False, transform_step=False)

        n_trials = len(trials)
        trans_params = numpy.empty((n_trials, trans.bounds.shape[0]), dtype=numpy.float64)
        trans_values = numpy.empty(n_trials, dtype=numpy.float64)

        for trial_idx, trial in enumerate(trials):
            trans_params[trial_idx] = trans.transform(trial.params)
            trans_values[trial_idx] = trial.value if target is None else target(trial)

        trans_bounds = trans.bounds
        column_to_encoded_columns = trans.column_to_encoded_columns

        if trans_params.size == 0:  # `params` were given but as an empty list.
            return OrderedDict()

        # Many (deep) copies of the search spaces are required during the tree traversal and using
        # Optuna distributions will create a bottleneck.
        # Therefore, search spaces (parameter distributions) are represented by a single
        # `numpy.ndarray`, coupled with a list of flags that indicate whether they are categorical
        # or not.

        evaluator = self._evaluator
        evaluator.fit(
            X=trans_params,
            y=trans_values,
            search_spaces=trans_bounds,
            column_to_encoded_columns=column_to_encoded_columns,
        )

        importances = {}
        for i, name in enumerate(distributions.keys()):
            importance, _ = evaluator.get_importance((i,))
            importances[name] = importance

        total_importance = sum(importances.values())
        for name in importances.keys():
            importances[name] /= total_importance

        sorted_importances = OrderedDict(
            reversed(
                sorted(importances.items(), key=lambda name_and_importance: name_and_importance[1])
            )
        )
        return sorted_importances
