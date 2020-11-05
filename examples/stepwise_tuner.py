import copy
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import lightgbm as lgb

import optuna
from optuna.distributions import UniformDistribution
from optuna.integration._lightgbm_tuner.optimize import _DEFAULT_LIGHTGBM_PARAMETERS
from optuna.integration._lightgbm_tuner.optimize import _DEFAULT_TUNER_TREE_DEPTH
from optuna.integration._lightgbm_tuner.optimize import _EPS
from optuna.integration._lightgbm_tuner.optimize import VALID_SET_TYPE
from optuna.samplers._stepwise import Step
from optuna.samplers._stepwise import StepwiseSampler


class Tuner:
    def __init__(
        self,
        params: Dict[str, Any],
        train_set: "lgb.Dataset",
        num_boost_round: int = 1000,
        valid_sets: Optional["VALID_SET_TYPE"] = None,
        valid_names: Optional[Any] = None,
        fobj: Optional[Callable[..., Any]] = None,
        feval: Optional[Callable[..., Any]] = None,
        feature_name: str = "auto",
        categorical_feature: str = "auto",
        early_stopping_rounds: Optional[int] = None,
        verbose_eval: Optional[Union[bool, int]] = True,
        callbacks: Optional[List[Callable[..., Any]]] = None,
        time_budget: Optional[int] = None,
        sample_size: Optional[int] = None,
        study: Optional[optuna.study.Study] = None,
    ):
        self._lgbm_params = params
        self.train_set = train_set
        self.lgbm_kwargs: Dict[str, Any] = dict(
            num_boost_round=num_boost_round,
            fobj=fobj,
            feval=feval,
            feature_name=feature_name,
            categorical_feature=categorical_feature,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval,
            callbacks=callbacks,
            valid_sets=valid_sets,
            valid_names=valid_names,
        )

    def suggest_lgbm_params(self, trial):
        lgbm_params = self._lgbm_params
        lgbm_params["lambda_l1"] = trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True)
        lgbm_params["lambda_l2"] = trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True)

        tree_depth = lgbm_params.get("max_depth", _DEFAULT_TUNER_TREE_DEPTH)
        max_num_leaves = 2 ** tree_depth if tree_depth > 0 else 2 ** _DEFAULT_TUNER_TREE_DEPTH
        lgbm_params["num_leaves"] = trial.suggest_int("num_leaves", 2, max_num_leaves)

        lgbm_params["feature_fraction"] = min(
            trial.suggest_float("feature_fraction", 0.4, 1.0 + _EPS), 1.0
        )
        lgbm_params["bagging_fraction"] = min(
            trial.suggest_float("bagging_fraction", 0.4, 1.0 + _EPS), 1.0
        )
        lgbm_params["bagging_freq"] = trial.suggest_int("bagging_freq", 1, 7)

        lgbm_params["min_child_samples"] = trial.suggest_int("min_child_samples", 5, 100)
        return

    def _copy_valid_sets(self, valid_sets: "VALID_SET_TYPE") -> "VALID_SET_TYPE":
        if isinstance(valid_sets, list):
            return [copy.copy(d) for d in valid_sets]
        if isinstance(valid_sets, tuple):
            return tuple([copy.copy(d) for d in valid_sets])
        return copy.copy(valid_sets)

    def _metric_with_eval_at(self, metric: str) -> str:

        lgbm_params = self._lgbm_params
        if metric != "ndcg" and metric != "map":
            return metric

        eval_at = lgbm_params.get("eval_at")
        if eval_at is None:
            eval_at = lgbm_params.get("{}_at".format(metric))
        if eval_at is None:
            eval_at = lgbm_params.get("{}_eval_at".format(metric))
        if eval_at is None:
            # Set default value of LightGBM.
            # See https://lightgbm.readthedocs.io/en/latest/Parameters.html#eval_at.
            eval_at = [1, 2, 3, 4, 5]

        # Optuna can handle only a single metric. Choose first one.
        if type(eval_at) in [list, tuple]:
            return "{}@{}".format(metric, eval_at[0])
        if type(eval_at) is int:
            return "{}@{}".format(metric, eval_at)
        raise ValueError(
            "The value of eval_at is expected to be int or a list/tuple of int."
            "'{}' is specified.".format(eval_at)
        )

    def _get_metric_for_objective(self) -> str:
        metric = self._lgbm_params.get("metric", "binary_logloss")

        # todo (smly): This implementation is different logic from the LightGBM's python bindings.
        if type(metric) is str:
            pass
        elif type(metric) is list:
            metric = metric[-1]
        elif type(metric) is set:
            metric = list(metric)[-1]
        else:
            raise NotImplementedError
        metric = self._metric_with_eval_at(metric)

        return metric

    def _get_booster_best_score(self, booster: "lgb.Booster") -> float:

        metric = self._get_metric_for_objective()
        valid_sets: Optional[VALID_SET_TYPE] = self.lgbm_kwargs.get("valid_sets")

        if self.lgbm_kwargs.get("valid_names") is not None:
            if type(self.lgbm_kwargs["valid_names"]) is str:
                valid_name = self.lgbm_kwargs["valid_names"]
            elif type(self.lgbm_kwargs["valid_names"]) in [list, tuple]:
                valid_name = self.lgbm_kwargs["valid_names"][-1]
            else:
                raise NotImplementedError

        elif type(valid_sets) is lgb.Dataset:
            valid_name = "valid_0"

        elif isinstance(valid_sets, (list, tuple)) and len(valid_sets) > 0:
            valid_set_idx = len(valid_sets) - 1
            valid_name = "valid_{}".format(valid_set_idx)

        else:
            raise NotImplementedError

        val_score = booster.best_score[valid_name][metric]
        return val_score

    def objective(self, trial):
        self.suggest_lgbm_params(trial)

        train_set = copy.copy(self.train_set)
        kwargs = copy.copy(self.lgbm_kwargs)
        kwargs["valid_sets"] = self._copy_valid_sets(kwargs["valid_sets"])
        booster = lgb.train(self._lgbm_params, train_set, **kwargs)

        val_score = self._get_booster_best_score(booster)
        return val_score

    def tune_feature_fraction(self, n_trials=7):
        param_name = "feature_fraction"
        param_values = np.linspace(0.4, 1.0, n_trials).tolist()
        search_space = {param_name: UniformDistribution(0.4, 1.0 + _EPS)}
        sampler = optuna.samplers.GridSampler({param_name: param_values})
        return lambda _: search_space, lambda _: sampler, n_trials

    def tune_num_leaves(self, n_trials: int = 20):
        tree_depth = self._lgbm_params.get("max_depth", _DEFAULT_TUNER_TREE_DEPTH)
        max_num_leaves = 2 ** tree_depth if tree_depth > 0 else 2 ** _DEFAULT_TUNER_TREE_DEPTH
        search_space = {
            "num_leaves": optuna.distributions.IntUniformDistribution(2, max_num_leaves)
        }
        return lambda _: search_space, lambda _: optuna.samplers.TPESampler(), n_trials

    def tune_bagging(self, n_trials: int = 10):
        search_space = {
            "bagging_fraction": optuna.distributions.UniformDistribution(0.4, 1.0 + _EPS),
            "bagging_freq": optuna.distributions.IntUniformDistribution(1, 7),
        }
        return lambda _: search_space, lambda _: optuna.samplers.TPESampler(), n_trials

    def tune_feature_fraction_stage2(self, n_trials: int = 6):
        param_name = "feature_fraction"

        def search_space_fn(params):
            search_space = {param_name: UniformDistribution(0.4, 1.0 + _EPS)}
            return search_space

        def sampler_fn(params):
            best_feature_fraction = params[param_name]
            param_values = np.linspace(
                best_feature_fraction - 0.08, best_feature_fraction + 0.08, n_trials
            ).tolist()
            param_values = [val for val in param_values if val >= 0.4 and val <= 1.0]
            return optuna.samplers.GridSampler({"feature_fraction": param_values})

        return search_space_fn, sampler_fn, n_trials

    def tune_regularization_factors(self, n_trials: int = 20):
        search_space = {
            "lambda_l1": optuna.distributions.LogUniformDistribution(1e-8, 10.0),
            "lambda_l2": optuna.distributions.LogUniformDistribution(1e-8, 10.0),
        }
        return lambda _: search_space, lambda _: optuna.samplers.TPESampler(), n_trials

    def tune_min_data_in_leaf(self):
        param_name = "min_child_samples"
        param_values = [5, 10, 25, 50, 100]
        search_space = {param_name: optuna.distributions.IntUniformDistribution(5, 100)}

        sampler = optuna.samplers.GridSampler({param_name: param_values})
        return lambda _: search_space, lambda _: sampler, len(param_values)

    def run(self):
        steps = [
            Step(*self.tune_feature_fraction()),
            Step(*self.tune_num_leaves()),
            Step(*self.tune_bagging()),
            Step(*self.tune_feature_fraction_stage2()),
            Step(*self.tune_regularization_factors()),
            Step(*self.tune_min_data_in_leaf()),
        ]
        sampler = StepwiseSampler(steps, default_params=_DEFAULT_LIGHTGBM_PARAMETERS)
        study = optuna.create_study(sampler=sampler)
        study.optimize(self.objective)


if __name__ == "__main__":
    import numpy as np
    import sklearn.datasets
    from sklearn.model_selection import train_test_split

    data, target = sklearn.datasets.load_breast_cancer(return_X_y=True)
    train_x, val_x, train_y, val_y = train_test_split(data, target, test_size=0.25)
    dtrain = lgb.Dataset(train_x, label=train_y)
    dval = lgb.Dataset(val_x, label=val_y)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
    }

    tuner = Tuner(
        params, dtrain, valid_sets=[dtrain, dval], verbose_eval=100, early_stopping_rounds=100
    )
    tuner.run()
