from typing import Dict

import numpy as np

import optuna
from optuna import multi_objective
from optuna.distributions import BaseDistribution
from optuna.multi_objective.samplers import BaseMultiObjectiveSampler
from optuna.study import Study
from optuna.trial import FrozenTrial

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, Matern, RBF

class CARBSSampler(optuna.samplers.BaseMultiObjectiveSampler):
    def __init__(self, temperature=100):
        self._rng = np.random.default_rng()
        self._current_trial = None  # Current state. 

    def reseed_rng(self) -> None:
        self._rng.seed()
        self._random_sampler.reseed_rng()

    def _calc_pareto_front(self, study):
        complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

        # Sort trials by cost, assuming "cost" is the first objective
        sorted_trials = sorted(complete_trials, key=lambda x: x.values[0])

        # Initialize best_performance, assuming "performance" is the second objective
        pareto_front = []
        best_performance = sorted_trials[0].values[1]

        # Check the direction of "performance" optimization
        if study.directions[1] == optuna.study.StudyDirection.MAXIMIZE:

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
 
        
    #TODO: make this work right
    def _sample_candidates(self, trial, n_candidates, search_space):
        
        xi = np.array(list(trial.hparams.values()))
        samples = self._rng.multivariate_normal(xi, covariate_matrix, n_candidates)
        
        # convert covar matrix to sigma ?
        sigma = np.magnitude(covariate_matrix)
    
        # calculate p_search (CARBS eq. 2)
        p_search = np.exp(- abs(xi - samples)**2 / (2 * sigma**2))

        return [(samples[i], p_search[i]) for i in range(n_candidates)]
    
    def sample_relative(self, study, trial, search_space):
        if search_space == {}:
            return {}
        
        # CARBS algorithm.
        # 1. Calculate pareto front
        pareto_front = _calculate_pareto_front(study)

        # 2. Generate candidates
        candidates = [_generate_candidates(point, n_candidates, search_space) for point in pareto_front]
        
        # 3. fit GP models to the candidates
        X = [ob.hparams for ob in obs]
        performances = [ob.performance for ob in obs]
        costs = [ob.cost for ob in obs]

        # fit performance model
        performance_kernel = DotProduct + Matern()
        GPy = GaussianProcessRegressor(kernel=performance_kernel,
                random_state=self._rng).fit(X, performances)
        
        # fit cost model
        cost_kernel = DotProduct + Matern()
        GPc = GaussianProcessRegressor(kernel=cost_kernel,
                random_state=self._rng).fit(X, costs)
        
        # fit pareto front model
        pareto_kernel = RBF()
        GPpf = GaussianProcessRegressor(kernel=pareto_kernel,
                random_state=self._rng).fit(costs[pareto_front], performances[pareto_front])

        acquisitions = []
        for cand in x_candidates:
            predicted_performance = GPy.predict(cand)
            predicted_cost = GPc.predict(cand)
            pf_performance = GPpf.predict(predicted_cost)

            # calculate expected improvement
            expected_improvement = relu(predicted_performance - pf_performance)

            # sample
            cand.acq = cand.p_search * expected_improvement
            acquisitions.append(cand)

            #TODO model odds of failure

        return max(acquisitions, key=lambda x: x.acq)

    # boilerplate
    def infer_relative_search_space(self, study, trial):
        return optuna.search_space.intersection_search_space(study.get_trials(deepcopy=False))
    
    # maybe we randomly sample the search center ?
    def sample_independent(self, study, trial, param_name, param_distribution):
        independent_sampler = optuna.samplers.RandomSampler()
        return independent_sampler.sample_independent(study, trial, param_name, param_distribution)


if __name__ == "__main__":
    def objective(trial):
        x = trial.suggest_float("x", -10, 10)
        y = trial.suggest_float("y", -5, 5)

        cost = x
        performance = x**2 + y

        return cost, performance


    #sampler = CARBSSampler()
    study = optuna.create_study(['minimize', 'maximize'])
    study.optimize(objective, n_trials=100)

    best_trial = study.best_trial
    print("Best value: ", best_trial.value)
    print("Parameters that achieve the best value: ", best_trial.params)