from typing import Optional
from typing import Sequence
from typing import Dict
from typing import Any
from collections import namedtuple

import numpy as np

import optuna
from optuna.multi_objective.samplers._base import BaseMultiObjectiveSampler
from optuna.multi_objective.study import Study, StudyDirection
from optuna.trial import FrozenTrial, TrialState
from optuna.distributions import BaseDistribution
from optuna.search_space import IntersectionSearchSpace
from optuna.samplers._random import RandomSampler

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, Matern, RBF

class CARBSSampler(BaseMultiObjectiveSampler):
    def __init__(
            self, 
            n_candidates: int = 3,
            _n_startup_trials: int = 10,
            seed: Optional[int] = None,
        ) -> None:
        self.n_candidates = n_candidates
        self._n_startup_trials = _n_startup_trials

        self._rng = np.random.RandomState(seed=seed)
        self._search_space = IntersectionSearchSpace(include_pruned=True)
        self._random_sampler = RandomSampler(seed=seed)
        self._current_trial = None  # Current state. 

    def reseed_rng(self) -> None:
        self._rng.seed()
        self._random_sampler.reseed_rng()

    def _calculate_pareto_front(self, trials, direction):

        # Sort trials by cost, assuming "cost" is the first objective
        sorted_trials = sorted(trials, key=lambda x: x.values[0])

        # Initialize best_performance, assuming "performance" is the second objective
        pareto_front = [sorted_trials[0]]
        best_performance = sorted_trials[0].values[1]

        # Check the direction of "performance" optimization
        if direction == StudyDirection.MAXIMIZE:

            for t in sorted_trials:
                if t.values[1] > best_performance:
                    pareto_front.append(t)
                    best_performance = t.values[1]
        else:   # If we're minimizing performance
            for t in sorted_trials:
                if t.values[1] < best_performance:
                    pareto_front.append(t)
                    best_performance = t.values[1]

        return pareto_front
 
        
    #TODO: not sure i'm using sigma correctly
    def _sample_candidates(self, trial, n_candidates, sigma):
        
        xi = np.array(list(trial.params.values()))
        covariate_matrix = np.eye(len(xi)) * sigma**2 
        params = self._rng.multivariate_normal(xi, covariate_matrix, n_candidates) 

        # calculate p_search (CARBS eq. 2)
        p_search = np.exp(- np.linalg.norm(xi - params, axis=1)**2 / (2 * sigma**2))

        candidate = namedtuple('candidate', ['params', 'p_search'])
        return [candidate(params[i], p_search[i]) for i in range(n_candidates)]
    
    def sample_relative(
        self,
        study,
        trial,
        search_space
    ) -> Dict[str, Any]:
        if search_space == {}:
            return {}

        states = (TrialState.COMPLETE, TrialState.PRUNED)
        trials = study._get_trials(deepcopy=False, states=states, use_cache=True)
        # If the number of samples is insufficient, we run random trial.
        if len(trials) < self._n_startup_trials:
            return {}

        # CARBS algorithm.
        # 1. Calculate pareto front, passing direction for performance
        pareto_front = self._calculate_pareto_front(trials, study.directions[1])

        # 2. Generate candidates
        candidates = [self._sample_candidates(point, self.n_candidates, sigma=.5)
                        for point in pareto_front]
        candidates = [c for cands in candidates for c in cands]
        
        # 3. fit GP models to the candidates
        X = [list(t.params.values()) for t in trials]
        costs = [t.values[0]for t in trials]
        performances = [t.values[1] for t in trials]

        pf_costs = np.array([t.values[0]for t in pareto_front]).reshape(-1, 1)
        pf_performances = [t.values[1] for t in pareto_front]

        # fit performance model
        performance_kernel = DotProduct() + Matern()
        GPy = GaussianProcessRegressor(kernel=performance_kernel,
                random_state=self._rng).fit(X, performances)
        
        # fit cost model
        cost_kernel = DotProduct() + Matern()
        GPc = GaussianProcessRegressor(kernel=cost_kernel,
                random_state=self._rng).fit(X, costs)
        
        # fit pareto front model
        pareto_kernel = RBF()
        GPpf = GaussianProcessRegressor(kernel=pareto_kernel,
                random_state=self._rng).fit(pf_costs, pf_performances)

        best_acquisition = 0
        best_cand = candidates[0]
        for cand in candidates:
            params = cand.params.reshape(1, -1)
            predicted_performance = GPy.predict(params)
            predicted_cost = GPc.predict(params)
            pf_performance = GPpf.predict(predicted_cost.reshape(1, -1))

            # calculate expected improvement using relu
            expected_improvement = max(0, predicted_performance - pf_performance)

            acq = cand.p_search * expected_improvement

            #TODO model odds of failure

            if acq > best_acquisition:
                best_acquisition = acq
                best_cand = cand

        # obviously a hack
        ret = dict(zip(trials[0].params.keys(), best_cand.params))
        return ret
 
    def infer_relative_search_space(
        self, 
        study: Study, 
        trial: FrozenTrial
    ) -> Dict[str, BaseDistribution]:
        return self._search_space.calculate(study)
    
    # maybe we randomly sample the search center ?
    def sample_independent(
        self,
        study,
        trial,
        param_name,
        param_distribution
    ):
        states = (TrialState.COMPLETE, TrialState.PRUNED)
        trials = study._get_trials(deepcopy=False, states=states, use_cache=True)

        # If the number of samples is insufficient, we run random trial.
        if len(trials) < self._n_startup_trials:
            return self._random_sampler.sample_independent(
                study, trial, param_name, param_distribution
            )
    
    def before_trial(
        self,
        study: Study,
        trial: FrozenTrial
    ) -> None:
        pass

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Optional[Sequence[float]],
    ) -> None:
        pass