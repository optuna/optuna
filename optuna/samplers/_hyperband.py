from __future__ import annotations

from threading import local
from threading import Lock
from typing import Any
from typing import Protocol
from typing import TYPE_CHECKING

import numpy as np

import optuna
from optuna.samplers._base import BaseSampler
from optuna.samplers._lazy_random_state import LazyRandomState
from optuna.samplers._tpe.sampler import TPESampler


if TYPE_CHECKING:
    from collections.abc import Sequence

    from optuna.distributions import BaseDistribution
    from optuna.pruners import HyperbandPruner
    from optuna.storages import BaseStorage
    from optuna.study import Study
    from optuna.trial import FrozenTrial
    from optuna.trial import TrialState


class _SamplerFactory(Protocol):
    def __call__(self, *, seed: int) -> BaseSampler: ...


class _ThreadLocalBracketStudy(local):
    cached: tuple[Study, int, Study] | None = None


class HyperbandSampler(BaseSampler):
    """Fit a separate sampler to each bracket of a Hyperband study.

    By default, each bracket uses its own :class:`~optuna.samplers.TPESampler`.
    Set :class:`~optuna.pruners.HyperbandPruner` as the study's pruner and give it an integer
    ``max_resource`` so the bracket is known before sampling. Starting a trial raises
    :exc:`ValueError` if the study uses another pruner or ``max_resource`` is still ``"auto"``.

    .. note::
        TPE starts up independently in each bracket. See :class:`~optuna.pruners.HyperbandPruner`
        for guidance on setting ``n_trials`` or ``timeout``.
        Pass :class:`~optuna.samplers.RandomSampler` as ``sampler_factory`` to use random search
        as in the original Hyperband paper.

    Example:

        .. code-block:: python

            import optuna

            study = optuna.create_study(
                sampler=optuna.samplers.HyperbandSampler(seed=0),
                pruner=optuna.pruners.HyperbandPruner(min_resource=1, max_resource=81),
            )

    Args:
        sampler_factory:
            A callable that accepts a keyword argument ``seed`` and returns a new sampler.
            Defaults to :class:`~optuna.samplers.TPESampler`. For each child sampler,
            :meth:`~optuna.study.Study.get_trials` returns the history of its bracket, with
            the original trial numbers.
        seed:
            Seed for generating the seeds passed to the sampler factory.
    """

    def __init__(
        self, sampler_factory: _SamplerFactory = TPESampler, *, seed: int | None = None
    ) -> None:
        self._sampler_factory = sampler_factory
        self._rng = LazyRandomState(seed)
        self._bracket_samplers: dict[
            tuple[BaseStorage, int, HyperbandPruner, int], BaseSampler
        ] = {}
        self._lock = Lock()
        self._thread_local = _ThreadLocalBracketStudy()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        del state["_lock"]
        del state["_thread_local"]
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        vars(self).update(state)
        self._lock = Lock()
        self._thread_local = _ThreadLocalBracketStudy()

    def _bracket_study(self, study: Study, trial: FrozenTrial) -> Study:
        pruner = study.pruner
        cached = self._thread_local.cached
        if cached is not None:
            cached_study, trial_id, bracket_study = cached
            if (
                cached_study is study
                and trial_id == trial._trial_id
                and bracket_study.pruner is pruner
            ):
                return bracket_study

        if not isinstance(pruner, optuna.pruners.HyperbandPruner):
            raise ValueError("HyperbandSampler requires HyperbandPruner.")
        if pruner._max_resource == "auto":
            raise ValueError("HyperbandSampler requires an explicit max_resource.")

        bracket_id = pruner._get_bracket_id(study, trial)
        key = (study._storage, study._study_id, pruner, bracket_id)
        with self._lock:
            sampler = self._bracket_samplers.get(key)
            if sampler is None:
                seed = int(self._rng.rng.randint(np.iinfo(np.int32).max))
                sampler = self._sampler_factory(seed=seed)
                self._bracket_samplers[key] = sampler

        bracket_study = pruner._create_bracket_study(study, bracket_id)
        bracket_study.sampler = sampler
        self._thread_local.cached = (study, trial._trial_id, bracket_study)
        return bracket_study

    def before_trial(self, study: Study, trial: FrozenTrial) -> None:
        self._thread_local.cached = None
        bracket_study = self._bracket_study(study, trial)
        bracket_study.sampler.before_trial(bracket_study, trial)

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> dict[str, BaseDistribution]:
        bracket_study = self._bracket_study(study, trial)
        return bracket_study.sampler.infer_relative_search_space(bracket_study, trial)

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        bracket_study = self._bracket_study(study, trial)
        return bracket_study.sampler.sample_relative(bracket_study, trial, search_space)

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        bracket_study = self._bracket_study(study, trial)
        return bracket_study.sampler.sample_independent(
            bracket_study, trial, param_name, param_distribution
        )

    def after_trial(
        self, study: Study, trial: FrozenTrial, state: TrialState, values: Sequence[float] | None
    ) -> None:
        bracket_study = self._bracket_study(study, trial)
        self._thread_local.cached = None
        # Match Study.tell(), which clears the history cache before post-processing.
        bracket_study._thread_local.cached_all_trials = None
        bracket_study.sampler.after_trial(bracket_study, trial, state, values)
