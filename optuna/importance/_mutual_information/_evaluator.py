from copy import deepcopy
import math
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np

from optuna.distributions import BaseDistribution
from optuna.distributions import UniformDistribution
from optuna.importance._base import BaseImportanceEvaluator
from optuna.importance._mutual_information._parzen_estimator import _MultivariateParzenEstimator
from optuna.study import Study
from optuna.trial import TrialState


class MutualInformationImportanceEvaluator(BaseImportanceEvaluator):
    def __init__(self, n_MC: Optional[int] = None) -> None:
        self._n_MC = n_MC

    def evaluate(
        self, study: Study, params: Optional[List[str]] = None, n_MC: Optional[int] = None
    ) -> Dict[str, float]:

        search_space = study.best_trial.distributions
        if params is None:
            params = list(search_space.keys())

        values, scores = _get_multivariate_observation_pairs(study, params)
        parameters = {k: np.array(v) for k, v in values.items()}
        scores = np.array([score for step, score in scores])

        score_range = np.max(scores) - np.min(scores)
        score_distribution = UniformDistribution(
            np.min(scores) - 0.3 * score_range, np.max(scores) + 0.3 * score_range
        )  # type: BaseDistribution
        score_space = {"score": score_distribution}

        if self._n_MC is None:
            mpe_score = _MultivariateParzenEstimator({"score": scores}, score_space)
            score_log_prob = mpe_score.log_pdf({"score": scores})

            mi = {}
            for param_name, param_values in parameters.items():
                # We construct parameter space.
                param_space = {param_name: search_space[param_name]}
                joint_space = deepcopy(param_space)
                joint_space.update(score_space)
                # We construct Parzen estimators.
                mpe_param = _MultivariateParzenEstimator({param_name: param_values}, param_space)
                mpe_joint = _MultivariateParzenEstimator(
                    {param_name: param_values, "score": scores}, joint_space
                )
                # We calculate log_probs.
                param_log_prob = mpe_param.log_pdf({param_name: param_values})
                joint_log_prob = mpe_joint.log_pdf({param_name: param_values, "score": scores})

                mi[param_name] = np.mean(joint_log_prob - score_log_prob - param_log_prob)
        else:
            mi = {}
            for param_name, param_values in parameters.items():
                # We construct parameter space.
                param_space = {param_name: search_space[param_name]}
                joint_space = deepcopy(param_space)
                joint_space.update(score_space)
                # We construct Parzen estimators.
                mpe_score = _MultivariateParzenEstimator({"score": scores}, score_space)
                mpe_param = _MultivariateParzenEstimator({param_name: param_values}, param_space)
                mpe_joint = _MultivariateParzenEstimator(
                    {param_name: param_values, "score": scores}, joint_space
                )
                # We get Monte Carlo samples.
                rng = np.random.RandomState()
                mc_samples = mpe_joint.sample(rng, self._n_MC)
                # We calculate log_probs.
                score_log_prob = mpe_score.log_pdf({"score": mc_samples["score"]})
                param_log_prob = mpe_param.log_pdf({param_name: mc_samples[param_name]})
                joint_log_prob = mpe_joint.log_pdf(mc_samples)

                mi[param_name] = np.mean(joint_log_prob - score_log_prob - param_log_prob)

        mi = {k: v for k, v in sorted(mi.items(), key=lambda item: item[1], reverse=True)}
        return mi


def _get_multivariate_observation_pairs(
    study: Study, param_names: List[str]
) -> Tuple[Dict[str, List[Optional[float]]], List[Tuple[float, float]]]:

    scores = []
    values = {
        param_name: [] for param_name in param_names
    }  # type: Dict[str, List[Optional[float]]]
    for trial in study._storage.get_all_trials(study._study_id, deepcopy=False):

        # We extract score from the trial.
        if trial.state is TrialState.COMPLETE and trial.value is not None:
            score = (-float("inf"), trial.value)
        elif trial.state is TrialState.PRUNED:
            if len(trial.intermediate_values) > 0:
                step, intermediate_value = max(trial.intermediate_values.items())
                if math.isnan(intermediate_value):
                    score = (-step, float("inf"))
                else:
                    score = (-step, intermediate_value)
            else:
                score = (float("inf"), 0.0)
        else:
            continue
        scores.append(score)

        # We extract param_value from the trial.
        for param_name in param_names:
            assert param_name in trial.params
            distribution = trial.distributions[param_name]
            param_value = distribution.to_internal_repr(trial.params[param_name])
            values[param_name].append(param_value)

    return values, scores
