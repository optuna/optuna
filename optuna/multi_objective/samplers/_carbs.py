from typing import Dict
from typing import Optional
from typing import Sequence
from collections import namedtuple

import numpy as np

import optuna
from optuna.multi_objective.samplers import BaseMultiObjectiveSampler
from optuna.multi_objective.study import Study, StudyDirection
from optuna.trial import FrozenTrial, TrialState


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, Matern, RBF

class CARBSSampler(BaseMultiObjectiveSampler):
    def __init__(self, n_candidates=3):
        self._rng = np.random.RandomState()
        self.n_candidates = n_candidates
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
        else:  # If performance is to be minimized
            for t in sorted_trials:
                if t.values[1] < best_performance:
                    pareto_front.append(t)
                    best_performance = t.values[1]

        return pareto_front
 
        
    #TODO: I don't think i'm using sigma correctly
    def _sample_candidates(self, trial, n_candidates, sigma):
        
        xi = np.array(list(trial.params.values()))
        covariate_matrix = np.eye(len(xi)) * sigma**2 
        params = self._rng.multivariate_normal(xi, covariate_matrix, n_candidates) 

        # calculate p_search (CARBS eq. 2)
        p_search = np.exp(- np.linalg.norm(xi - params, axis=1)**2 / (2 * sigma**2))

        candidate = namedtuple('candidate', ['params', 'p_search'])
        return [candidate(params[i], p_search[i]) for i in range(n_candidates)]
    
    def sample_relative(self, study, trial, search_space):
        if search_space == {}:
            return {}
        
        complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

        # CARBS algorithm.
        # 1. Calculate pareto front, passing direction for performance
        pareto_front = self._calculate_pareto_front(complete_trials, study.directions[1])

        # 2. Generate candidates
        candidates = [self._sample_candidates(point, self.n_candidates, sigma=.5) for point in pareto_front]
        candidates = [item for row in candidates for item in row]
        
        # 3. fit GP models to the candidates
        X = [list(t.params.values()) for t in complete_trials]
        costs = [t.values[0]for t in complete_trials]
        performances = [t.values[1] for t in complete_trials]

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

        return best_cand.params

    # boilerplate
    def infer_relative_search_space(self, study, trial):
        return optuna.search_space.intersection_search_space(study.get_trials(deepcopy=False))
    
    # maybe we randomly sample the search center ?
    def sample_independent(self, study, trial, param_name, param_distribution):
        independent_sampler = optuna.samplers.RandomSampler()
        return independent_sampler.sample_independent(study, trial, param_name, param_distribution)
    
    def before_trial(self, study: Study, trial: FrozenTrial) -> None:
        pass

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Optional[Sequence[float]],
    ) -> None:
        pass


def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    y = trial.suggest_float("y", -5, 5)

    cost = abs(x) / 2
    performance = x**2 + y

    return cost, performance


sampler = CARBSSampler()
study = optuna.create_study(directions=['minimize', 'maximize'], sampler=sampler)
study.optimize(objective, n_trials=20)

optuna.visualization.plot_pareto_front(study, target_names=["cost", "performance"])

print(f"Number of trials on the Pareto front: {len(study.best_trials)}")

trial_with_highest_performance = max(study.best_trials, key=lambda t: t.values[1])
print(f"Trial with highest performance: ")
print(f"\tnumber: {trial_with_highest_performance.number}")
print(f"\tparams: {trial_with_highest_performance.params}")
print(f"\tvalues: {trial_with_highest_performance.values}")