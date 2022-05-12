from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

from optuna.importance._base import BaseImportanceEvaluator
from optuna.importance._fanova._fanova import _Fanova
from optuna.importance._utils import _gather_importance_values
from optuna.importance._utils import _gather_study_info
from optuna.importance._utils import _StudyInfo
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

        info: _StudyInfo
        info = _gather_study_info(study, params=params, target=target)

        importances = {}
        if len(info.non_single_distributions) > 0:
            trans = info.trans
            assert trans is not None

            trans_bounds = trans.bounds
            column_to_encoded_columns = trans.column_to_encoded_columns

            # Many (deep) copies of the search spaces are required during the tree traversal and
            # using Optuna distributions will create a bottleneck.
            # Therefore, search spaces (parameter distributions) are represented by a single
            # `numpy.ndarray`, coupled with a list of flags that indicate whether they are
            # categorical or not.

            evaluator = self._evaluator
            evaluator.fit(
                X=info.trans_params,
                y=info.trans_values,
                search_spaces=trans_bounds,
                column_to_encoded_columns=column_to_encoded_columns,
            )

            for i, name in enumerate(info.non_single_distributions.keys()):
                importance, _ = evaluator.get_importance((i,))
                importances[name] = importance

            total_importance = sum(importances.values())
            for name in importances:
                importances[name] /= total_importance

        return _gather_importance_values(importances, info.single_distributions)
