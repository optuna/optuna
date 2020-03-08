import contextlib
import copy
import time

import lightgbm as lgb
import numpy as np
import tqdm

import optuna
from optuna.integration.lightgbm_tuner.alias import _handling_alias_parameters
from optuna import type_checking

if type_checking.TYPE_CHECKING:
    from typing import Any  # NOQA
    from typing import Callable  # NOQA
    from typing import Dict  # NOQA
    from typing import Generator  # NOQA
    from typing import List  # NOQA
    from typing import Optional  # NOQA
    from typing import Tuple  # NOQA
    from typing import Union  # NOQA

    from optuna.distributions import BaseDistribution  # NOQA
    from optuna.structs import FrozenTrial  # NOQA
    from optuna.study import Study  # NOQA
    from optuna.trial import Trial  # NOQA

    VALID_SET_TYPE = Union[List[lgb.Dataset], Tuple[lgb.Dataset, ...], lgb.Dataset]


# EPS is used to ensure that a sampled parameter value is in pre-defined value range.
EPS = 1e-12

# Default value of tree_depth, used for upper bound of num_leaves
DEFAULT_TUNER_TREE_DEPTH = 8

# Default parameter values described in the official webpage.
DEFAULT_LIGHTGBM_PARAMETERS = {
    'lambda_l1': 0.0,
    'lambda_l2': 0.0,
    'num_leaves': 31,
    'feature_fraction': 1.0,
    'bagging_fraction': 1.0,
    'bagging_freq': 0,
    'min_child_samples': 20,
}


class _GridSamplerUniform1D(optuna.samplers.BaseSampler):

    def __init__(self, param_name, param_values):
        # type: (str, Any) -> None

        self.param_name = param_name
        self.param_values = tuple(param_values)
        self.value_idx = 0

    def sample_relative(self, study, trial, search_space):
        # type: (Study, FrozenTrial, Dict[str, BaseDistribution]) -> Dict[str, float]

        # todo (g-votte): Take care of distributed optimization.
        assert self.value_idx < len(self.param_values)
        param_value = self.param_values[self.value_idx]
        self.value_idx += 1
        return {self.param_name: param_value}

    def sample_independent(self, study, trial, param_name, param_distribution):
        # type: (Study, FrozenTrial, str, BaseDistribution) -> None

        raise ValueError(
            'Suggest method is called for an invalid parameter: {}.'.format(param_name))

    def infer_relative_search_space(self, study, trial):
        # type: (Study, FrozenTrial) -> Dict[str, BaseDistribution]

        distribution = optuna.distributions.UniformDistribution(-float('inf'), float('inf'))
        return {self.param_name: distribution}


class _TimeKeeper(object):
    def __init__(self):
        # type: () -> None

        self.time = time.time()

    def elapsed_secs(self):
        # type: () -> float

        return time.time() - self.time


@contextlib.contextmanager
def _timer():
    # type: () -> Generator[_TimeKeeper, None, None]

    timekeeper = _TimeKeeper()
    yield timekeeper


class BaseTuner(object):
    def __init__(
            self,
            lgbm_params=None,
            lgbm_kwargs=None
    ):
        # type: (Dict[str, Any], Dict[str,Any]) -> None

        self.lgbm_params = lgbm_params or {}
        self.lgbm_kwargs = lgbm_kwargs or {}

    def _get_booster_best_score(self, booster):
        # type: (lgb.Booster) -> float

        metric = self.lgbm_params.get('metric', 'binary_logloss')

        # todo (smly): This implementation is different logic from the LightGBM's python bindings.
        if type(metric) is str:
            pass
        elif type(metric) is list:
            metric = metric[-1]
        elif type(metric) is set:
            metric = list(metric)[-1]
        else:
            raise NotImplementedError
        valid_sets = self.lgbm_kwargs.get('valid_sets')  # type: Optional[VALID_SET_TYPE]

        if self.lgbm_kwargs.get('valid_names') is not None:
            if type(self.lgbm_kwargs['valid_names']) is str:
                valid_name = self.lgbm_kwargs['valid_names']
            elif type(self.lgbm_kwargs['valid_names']) in [list, tuple]:
                valid_name = self.lgbm_kwargs['valid_names'][-1]
            else:
                raise NotImplementedError

        elif type(valid_sets) is lgb.Dataset:
            valid_name = 'valid_0'

        elif isinstance(valid_sets, (list, tuple)) and len(valid_sets) > 0:
            valid_set_idx = len(valid_sets) - 1
            valid_name = 'valid_{}'.format(valid_set_idx)

        else:
            raise NotImplementedError

        metric = self._metric_with_eval_at(metric)
        val_score = booster.best_score[valid_name][metric]
        return val_score

    def _metric_with_eval_at(self, metric):
        # type: (str) -> str

        if metric != 'ndcg' and metric != 'map':
            return metric

        eval_at = self.lgbm_params.get('eval_at')
        if eval_at is None:
            eval_at = self.lgbm_params.get('{}_at'.format(metric))
        if eval_at is None:
            eval_at = self.lgbm_params.get('{}_eval_at'.format(metric))
        if eval_at is None:
            # Set default value of LightGBM.
            # See https://lightgbm.readthedocs.io/en/latest/Parameters.html#eval_at.
            eval_at = [1, 2, 3, 4, 5]

        # Optuna can handle only a single metric. Choose first one.
        if type(eval_at) in [list, tuple]:
            return '{}@{}'.format(metric, eval_at[0])
        if type(eval_at) is int:
            return '{}@{}'.format(metric, eval_at)
        raise ValueError('The value of eval_at is expected to be int or a list/tuple of int.'
                         '\'{}\' is specified.'.format(eval_at))

    def higher_is_better(self):
        # type: () -> bool

        metric_name = self.lgbm_params.get('metric', 'binary_logloss')
        return metric_name.startswith(('auc', 'ndcg', 'map', 'accuracy'))

    def compare_validation_metrics(self, val_score, best_score):
        # type: (float, float) -> bool

        if self.higher_is_better():
            return val_score > best_score
        else:
            return val_score < best_score


class OptunaObjective(BaseTuner):
    """Objective for hyperparameter-tuning with Optuna."""

    def __init__(
            self,
            target_param_names,  # type: List[str]
            lgbm_params,  # type: Dict[str, Any]
            train_set,  # type: lgb.Dataset
            lgbm_kwargs,  # type: Dict[str, Any]
            best_score,  # type: float
            pbar=None,  # type: Optional[tqdm.tqdm]
    ):

        self.target_param_names = target_param_names
        self.pbar = pbar
        self.lgbm_params = lgbm_params
        self.lgbm_kwargs = lgbm_kwargs
        self.train_set = train_set

        self.report = []  # type: List[Dict[str, Any]]
        self.trial_count = 0
        self.best_score = best_score
        self.best_booster = None
        self.action = 'tune_' + '_and_'.join(self.target_param_names)

        self._check_target_names_supported()

    def _check_target_names_supported(self):
        # type: () -> None

        supported_param_names = [
            'lambda_l1',
            'lambda_l2',
            'num_leaves',
            'feature_fraction',
            'bagging_fraction',
            'bagging_freq',
            'min_child_samples',
        ]
        for target_param_name in self.target_param_names:
            if target_param_name not in supported_param_names:
                raise NotImplementedError("Parameter `{}` is not supported for tunning.")

    def __call__(self, trial):
        # type: (Trial) -> float

        pbar_fmt = "{}, val_score: {:.6f}"

        if self.pbar is not None:
            self.pbar.set_description(pbar_fmt.format(self.action, self.best_score))

        if 'lambda_l1' in self.target_param_names:
            self.lgbm_params['lambda_l1'] = trial.suggest_loguniform('lambda_l1', 1e-8, 10.0)
        if 'lambda_l2' in self.target_param_names:
            self.lgbm_params['lambda_l2'] = trial.suggest_loguniform('lambda_l2', 1e-8, 10.0)
        if 'num_leaves' in self.target_param_names:
            tree_depth = self.lgbm_params.get('max_depth', DEFAULT_TUNER_TREE_DEPTH)
            max_num_leaves = 2**tree_depth if tree_depth > 0 else 2**DEFAULT_TUNER_TREE_DEPTH
            self.lgbm_params['num_leaves'] = trial.suggest_int(
                'num_leaves', 2, max_num_leaves)
        if 'feature_fraction' in self.target_param_names:
            # `_GridSamplerUniform1D` is used for sampling feature_fraction value.
            # The value 1.0 for the hyperparameter is always sampled.
            param_value = min(trial.suggest_uniform('feature_fraction', 0.4, 1.0 + EPS), 1.0)
            self.lgbm_params['feature_fraction'] = param_value
        if 'bagging_fraction' in self.target_param_names:
            # `TPESampler` is used for sampling bagging_fraction value.
            # The value 1.0 for the hyperparameter might by sampled.
            param_value = min(trial.suggest_uniform('bagging_fraction', 0.4, 1.0 + EPS), 1.0)
            self.lgbm_params['bagging_fraction'] = param_value
        if 'bagging_freq' in self.target_param_names:
            self.lgbm_params['bagging_freq'] = trial.suggest_int('bagging_freq', 1, 7)
        if 'min_child_samples' in self.target_param_names:
            # `_GridSamplerUniform1D` is used for sampling min_child_samples value.
            # The value 1.0 for the hyperparameter is always sampled.
            param_value = int(trial.suggest_uniform('min_child_samples', 5, 100 + EPS))
            self.lgbm_params['min_child_samples'] = param_value

        with _timer() as t:
            booster = lgb.train(self.lgbm_params, self.train_set, **self.lgbm_kwargs)

        val_score = self._get_booster_best_score(booster)
        elapsed_secs = t.elapsed_secs()
        average_iteration_time = elapsed_secs / booster.current_iteration()
        if self.compare_validation_metrics(val_score, self.best_score):
            self.best_score = val_score
            self.best_booster = booster

        if self.pbar is not None:
            self.pbar.set_description(pbar_fmt.format(self.action, self.best_score))
            self.pbar.update(1)

        self.report.append(dict(
            action=self.action,
            trial=self.trial_count,
            value=str(trial.params),
            val_score=val_score,
            elapsed_secs=elapsed_secs,
            average_iteration_time=average_iteration_time))

        self.trial_count += 1

        return val_score


class LightGBMTuner(BaseTuner):
    """Hyperparameter-tuning with Optuna for LightGBM."""

    def __init__(
            self,
            params,  # type: Dict[str, Any]
            train_set,  # type: lgb.Dataset
            num_boost_round=1000,  # type: int
            valid_sets=None,  # type: Optional[VALID_SET_TYPE]
            valid_names=None,  # type: Optional[Any]
            fobj=None,  # type: Optional[Callable[..., Any]]
            feval=None,  # type: Optional[Callable[..., Any]]
            feature_name='auto',  # type: str
            categorical_feature='auto',  # type: str
            early_stopping_rounds=None,  # type: Optional[int]
            evals_result=None,  # type: Optional[Dict[Any, Any]]
            verbose_eval=True,  # type: Optional[bool]
            learning_rates=None,  # type: Optional[List[float]]
            keep_training_booster=False,  # type: Optional[bool]
            callbacks=None,  # type: Optional[List[Callable[..., Any]]]
            time_budget=None,  # type: Optional[int]
            sample_size=None,  # type: Optional[int]
            best_params=None,  # type: Optional[Dict[str, Any]]
            tuning_history=None,  # type: Optional[List[Dict[str, Any]]]
            verbosity=1,  # type: Optional[int]
    ):
        params = copy.deepcopy(params)
        args = [params, train_set]
        kwargs = dict(num_boost_round=num_boost_round,
                      valid_sets=valid_sets,
                      valid_names=valid_names,
                      fobj=fobj,
                      feval=feval,
                      feature_name=feature_name,
                      categorical_feature=categorical_feature,
                      early_stopping_rounds=early_stopping_rounds,
                      evals_result=evals_result,
                      verbose_eval=verbose_eval,
                      learning_rates=learning_rates,
                      keep_training_booster=keep_training_booster,
                      callbacks=callbacks,
                      time_budget=time_budget,
                      verbosity=verbosity,
                      sample_size=sample_size)  # type: Dict[str, Any]
        self._parse_args(*args, **kwargs)
        self.best_booster = None

        self.best_score = -np.inf if self.higher_is_better() else np.inf
        self.best_params = {} if best_params is None else best_params
        self.tuning_history = [] if tuning_history is None else tuning_history

        # Set default parameters as best.
        self.best_params.update(DEFAULT_LIGHTGBM_PARAMETERS)

        if valid_sets is None:
            raise ValueError("`valid_sets` is required.")

    def _get_params(self):
        # type: () -> Dict[str, Any]

        params = copy.deepcopy(self.lgbm_params)
        params.update(self.best_params)
        return params

    def _parse_args(self, *args, **kwargs):
        # type: (Any, Any) -> None

        self.auto_options = {
            option_name: kwargs.get(option_name)
            for option_name in [
                'time_budget',
                'sample_size',
                'best_params',
                'tuning_history',
                'verbosity',
            ]
        }

        # Split options.
        for option_name in self.auto_options.keys():
            if option_name in kwargs:
                del kwargs[option_name]

        self.lgbm_params = args[0]
        self.train_set = args[1]
        self.train_subset = None  # Use for sampling.
        self.lgbm_kwargs = kwargs

    def run(self):
        # type: () -> lgb.Booster
        """Perform the hyperparameter-tuning with given parameters.

        Returns:

            booster : Booster
                The trained Booster model.
        """
        # Surpress log messages.
        if self.auto_options['verbosity'] == 0:
            optuna.logging.disable_default_handler()
            self.lgbm_params['verbose'] = -1
            self.lgbm_params['seed'] = 111
            self.lgbm_kwargs['verbose_eval'] = False

        # Handling aliases.
        _handling_alias_parameters(self.lgbm_params)

        # Sampling.
        self.sample_train_set()

        # Tuning.
        time_budget = self.auto_options['time_budget']

        self.start_time = time.time()
        with _timer() as t:
            self.tune_feature_fraction()
            if time_budget is not None and time_budget < t.elapsed_secs():
                return self.best_booster

            self.tune_num_leaves()
            if time_budget is not None and time_budget < t.elapsed_secs():
                return self.best_booster

            self.tune_bagging()
            if time_budget is not None and time_budget < t.elapsed_secs():
                return self.best_booster

            self.tune_feature_fraction_stage2()
            if time_budget is not None and time_budget < t.elapsed_secs():
                return self.best_booster

            self.tune_regularization_factors()
            if time_budget is not None and time_budget < t.elapsed_secs():
                return self.best_booster

            self.tune_min_data_in_leaf()
            if time_budget is not None and time_budget < t.elapsed_secs():
                return self.best_booster

        return self.best_booster

    def sample_train_set(self):
        # type: () -> None
        """Make subset of `self.train_set` Dataset object."""

        if self.auto_options['sample_size'] is None:
            return

        self.train_set.construct()
        n_train_instance = self.train_set.get_label().shape[0]
        if n_train_instance > self.auto_options['sample_size']:
            offset = n_train_instance - self.auto_options['sample_size']
            idx_list = offset + np.arange(self.auto_options['sample_size'])
            self.train_subset = self.train_set.subset(idx_list)

    def tune_feature_fraction(self, n_trials=7):
        # type: (int) -> None

        param_name = 'feature_fraction'
        param_values = list(np.linspace(0.4, 1.0, n_trials))
        sampler = _GridSamplerUniform1D(param_name, param_values)
        self.tune_params([param_name], len(param_values), sampler)

    def tune_num_leaves(self, n_trials=20):
        # type: (int) -> None

        self.tune_params(['num_leaves'], n_trials, optuna.samplers.TPESampler())

    def tune_bagging(self, n_trials=10):
        # type: (int) -> None

        self.tune_params(['bagging_fraction', 'bagging_freq'],
                         n_trials,
                         optuna.samplers.TPESampler())

    def tune_feature_fraction_stage2(self, n_trials=6):
        # type: (int) -> None

        param_name = 'feature_fraction'
        param_values = list(np.linspace(
            self.lgbm_params[param_name] - 0.08,
            self.lgbm_params[param_name] + 0.08,
            n_trials))
        param_values = [val for val in param_values if val >= 0.4 and val <= 1.0]
        sampler = _GridSamplerUniform1D(param_name, param_values)
        self.tune_params([param_name], len(param_values), sampler)

    def tune_regularization_factors(self, n_trials=20):
        # type: (int) -> None

        self.tune_params(['lambda_l1', 'lambda_l2'], n_trials, optuna.samplers.TPESampler())

    def tune_min_data_in_leaf(self):
        # type: () -> None

        param_name = 'min_child_samples'
        param_values = [5, 10, 25, 50, 100]
        sampler = _GridSamplerUniform1D(param_name, param_values)
        self.tune_params([param_name], len(param_values), sampler)

    def tune_params(self, target_param_names, n_trials, sampler):
        # type: (List[str], int, optuna.samplers.BaseSampler) -> None

        pbar = tqdm.tqdm(total=n_trials, ascii=True)

        # Set current best parameters.
        self.lgbm_params.update(self.best_params)

        train_set = self.train_set
        if self.train_subset is not None:
            train_set = self.train_subset

        objective = OptunaObjective(
            target_param_names,
            self.lgbm_params,
            train_set,
            self.lgbm_kwargs,
            self.best_score,
            pbar=pbar,
        )
        study = optuna.create_study(
            direction='maximize' if self.higher_is_better() else 'minimize',
            sampler=sampler)
        study.optimize(objective, n_trials=n_trials, catch=())

        pbar.close()
        del pbar

        # Add tuning history.
        self.tuning_history += objective.report

        if self.compare_validation_metrics(study.best_value, self.best_score):
            self.best_score = study.best_value
            self.best_booster = objective.best_booster

            updated_params = {p: study.best_trial.params[p] for p in target_param_names}
            self.lgbm_params.update(updated_params)
            self.best_params.update(updated_params)
