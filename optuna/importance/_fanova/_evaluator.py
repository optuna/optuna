from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

import numpy

from optuna._transform import _SearchSpaceTransform
from optuna.importance._base import _get_distributions
from optuna.importance._base import _get_filtered_trials
from optuna.importance._base import _get_target_values
from optuna.importance._base import _get_trans_params
from optuna.importance._base import _param_importances_to_dict
from optuna.importance._base import _sort_dict_by_importance
from optuna.importance._base import _split_nonsingle_and_single_distributions
from optuna.importance._base import BaseImportanceEvaluator
from optuna.importance._fanova._fanova import _Fanova
from optuna.study import Study
from optuna.trial import FrozenTrial


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

        distributions = _get_distributions(study, params=params)

        if params is None:
            params = list(distributions.keys())
        assert params is not None
        if target is None:

            def default_target(trial: FrozenTrial) -> float:
                assert trial.value is not None
                return trial.value

            target = default_target
        assert target is not None

        non_single_distributions, single_distributions = _split_nonsingle_and_single_distributions(
            distributions
        )

        if len(non_single_distributions) == 0:
            return {}

        trials: List[FrozenTrial] = _get_filtered_trials(study, params=params, target=target)

        trans = _SearchSpaceTransform(
            non_single_distributions, transform_log=False, transform_step=False
        )

        trans_params: numpy.ndarray = _get_trans_params(trials, trans)
        values: numpy.ndarray = _get_target_values(trials, target)

        evaluator = self._evaluator
        evaluator.fit(
            X=trans_params,
            y=values,
            search_spaces=trans.bounds,
            column_to_encoded_columns=trans.column_to_encoded_columns,
        )
        param_importances = numpy.array(
            [evaluator.get_importance((i,))[0] for i in range(len(non_single_distributions))]
        )
        param_importances /= numpy.sum(param_importances)

        return _sort_dict_by_importance(
            {
                **_param_importances_to_dict(non_single_distributions.keys(), param_importances),
                **_param_importances_to_dict(single_distributions.keys(), 0.0),
            }
        )
