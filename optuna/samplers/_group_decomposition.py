from typing import Any
from typing import Dict
from typing import Optional

from optuna._experimental import experimental
from optuna.distributions import BaseDistribution
from optuna.samplers._base import BaseSampler
from optuna.samplers._search_space import _GroupDecomposedSearchSpace
from optuna.samplers._search_space import _SearchSpaceGroup
from optuna.samplers._tpe.sampler import TPESampler
from optuna.study import BaseStudy
from optuna.study import Study
from optuna.trial import FrozenTrial


@experimental("2.8.0")
class GroupDecompositionSampler(BaseSampler):
    """Sampler with a group decomposed search space.

    This sampler decomposes the search space based on past trials and samples from the joint
    distribution in each decomposed subspace.

    The search space is decomposed based on the following recursive rules.
    - Initialize the group of the search space with the empty set. The elements of the group are
    the subset of the search space, and the type is the dictionary of
    :class:`~optuna.distributions.BaseDistribution`.
    - Update the group with the following process by looking at past trials in order.
        - Let ``T = trial.distributions`.
        - If the intersection of any element of the group and ``T`` is empty, add ``T`` to the
        group.
        - If an element ``S`` of the group is contained in ``T``, then add ``T-S`` to the group.
        We recursively add ``T-S`` to the group because the intersection of ``T-S`` and
        some other elements of the group may not be empty.
        - If an element ``S`` of a group contains ``T``, remove ``S`` from the group and add ``T``
        and ``S-T`` to the group.
        - If the intersection of an element ``S`` of the group and ``T`` is not empty, remove ``S``
        from the group and add ``S∩T``, ``S-T``, and ``T-S`` to the group.
        We recursively add ``T-S`` to the group because the intersection of ``T-S`` and
        some other elements of the group may not be empty.

    The group of the search space recursively constructed based on the above rules are disjoint and
    the union is the entire search space. We perform sampling from the joint distribution for each
    element of this decomposed group of the search space.

    Sampling from the joint distribution on the subspace is realized by
    :meth:`~optuna.samplers.BaseSampler.sample_relative` of ``base_sampler`` specified by the user.


    Example:

       .. testcode::

           import optuna


           def objective(trial):
               x = trial.suggest_categorial("x", ["A", "B"])
               if x == "A":
                   return trial.suggest_float("y", -10, 10)
               else:
                   return trial.suggest_int("z", -10, 10)


           base_sampler = optuna.samplers.TPESampler(multivariate=True)
           sampler = optuna.samplers.GroupDecompositionSampler(base_sampler=base_sampler)
           study = optuna.create_study(sampler=sampler)
           study.optimize(objective, n_trials=10)

    Args:
        base_sampler:
            A sampler to sample from the joint distribution for the each decomposed search space.

    """

    def __init__(self, base_sampler: BaseSampler):
        self._base_sampler = base_sampler
        self._include_pruned = True if isinstance(base_sampler, TPESampler) else False
        self._search_space_group: Optional[_SearchSpaceGroup] = None

    def infer_relative_search_space(
        self, study: BaseStudy, trial: FrozenTrial
    ) -> Dict[str, BaseDistribution]:
        self._search_space_group = _GroupDecomposedSearchSpace(self._include_pruned).calculate(
            study
        )
        search_space = {}
        for sub_space in self._search_space_group.group:
            search_space.update(sub_space)
        return search_space

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: Dict[str, BaseDistribution]
    ) -> Dict[str, Any]:
        assert self._search_space_group is not None
        params = {}
        for sub_space in self._search_space_group.group:
            params.update(self._base_sampler.sample_relative(study, trial, sub_space))
        return params

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        return self._base_sampler.sample_independent(study, trial, param_name, param_distribution)
