import contextlib
import time
import math
import copy

import lightgbm as lgb
import numpy as np
import tqdm

import optuna
from optuna import type_checking
from optuna.integration.lightgbm_autotune.alias import handling_alias_parameters


if type_checking.TYPE_CHECKING:
    from type_checking import Any  # NOQA
    from type_checking import Dict  # NOQA
    from type_checking import List  # NOQA
    from type_checking import Optional  # NOQA


# Default time budget for tuning `learning_rate`
DEFAULT_TIME_BUDGET_FOR_TUNING_LR = 4 * 60 * 60

EPS = 1e-12


class _GridSamplerUniform1D(optuna.samplers.BaseSampler):

    def __init__(self, param_name, param_values):
        self.param_name = param_name
        self.param_values = tuple(param_values)
        self.value_idx = 0

    def sample_relative(self, study, trial, search_space):
        # todo (g-votte): Take care of distributed optimization.
        assert self.value_idx < len(self.param_values)
        param_value = self.param_values[self.value_idx]
        self.value_idx += 1
        return {self.param_name: param_value}

    def sample_independent(self, study, trial, param_name, param_distribution):
        raise ValueError(
            'Suggest method is called for an invalid parameter: {}.'.format(param_name))

    def infer_relative_search_space(self, study, trial):
        distribution = optuna.distributions.UniformDistribution(-float('inf'), float('inf'))
        return {self.param_name: distribution}


class _TimeKeeper:
    def __init__(self):
        self.time = time.time()

    def elapsed_secs(self):
        return time.time() - self.time


@contextlib.contextmanager
def _timer():
    timekeeper = _TimeKeeper()
    yield timekeeper


class BaseTuner:
    def __init__(
            self,
            lgbm_params=None,
            lgbm_kwargs=None
    ):
        # type: (Dict[str, Any], Dict[str,Any]) -> None
        self.lgbm_params = lgbm_params if lgbm_params is not None else {}
        self.lgbm_kwargs = lgbm_kwargs if lgbm_kwargs is not None else {}

    def get_booster_best_score(self, booster):
        # type: () -> lgb.Booster
        metric = self.lgbm_params.get('metric', 'binary_logloss')

        valid_sets = self.lgbm_kwargs.get('valid_sets')
        if self.lgbm_kwargs.get('valid_names', None) is not None:
            if type(self.lgbm_kwargs['valid_names']) is str:
                valid_name = self.lgbm_kwargs['valid_names']
            elif type(self.lgbm_kwargs['valid_names']) in [list, tuple]:
                valid_name = self.lgbm_kwargs['valid_names'][-1]
            else:
                raise NotImplementedError

        elif type(valid_sets) is lgb.Dataset:
            valid_name = 'valid_0'

        elif type(valid_sets) in [list, tuple] and len(valid_sets) > 0:
            valid_set_idx = len(valid_sets) - 1
            valid_name = 'valid_{}'.format(valid_set_idx)

        else:
            raise NotImplementedError

        val_score = booster.best_score[valid_name][metric]
        return val_score

    def higher_is_better(self):
        metric_name = self.lgbm_params.get('metric', 'binary_logloss')
        return metric_name.startswith(('auc', 'ndcg@', 'map@', 'accuracy'))

    def compare_validation_metrics(self, val_score, best_score):
        if self.higher_is_better():
            return val_score > best_score
        else:
            return val_score < best_score


class OptunaObjective(BaseTuner):
    """Objective for hyperparameter-tuning with Optuna.
    """

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
        supported_param_names = [
            'lambda_l1',
            'lambda_l2',
            'num_leaves',
            'feature_fraction',
            'bagging_fraction',
            'min_child_samples',
        ]
        for target_param_name in self.target_param_names:
            if target_param_name not in supported_param_names:
                raise NotImplementedError("Parameter `{}` is not supported for tunning")

    def __call__(self, trial):
        pbar_fmt = "{}, val_score: {:.6f}"

        if self.pbar is not None:
            self.pbar.set_description(pbar_fmt.format(self.action, self.best_score))

        if 'lambda_l1' in self.target_param_names:
            self.lgbm_params['lambda_l1'] = trial.suggest_loguniform('lambda_l1', 1e-8, 10.0)
        if 'lambda_l2' in self.target_param_names:
            self.lgbm_params['lambda_l2'] = trial.suggest_loguniform('lambda_l2', 1e-8, 10.0)
        if 'num_leaves' in self.target_param_names:
            max_depth = self.lgbm_params.get('max_depth', 8)
            self.lgbm_params['num_leaves'] = trial.suggest_int(
                'num_leaves', 2, 2 ** max_depth)
        if 'feature_fraction' in self.target_param_names:
            param_value = min(trial.suggest_uniform('feature_fraction', 0.4, 1.0 + EPS), 1.0)
            self.lgbm_params['feature_fraction'] = param_value
        if 'bagging_fraction' in self.target_param_names:
            param_value = min(trial.suggest_uniform('bagging_fraction', 0.4, 1.0 + EPS), 1.0)
            self.lgbm_params['bagging_fraction'] = param_value
        if 'min_child_samples' in self.target_param_names:
            param_value = int(trial.suggest_uniform('min_child_samples', 5, 100 + EPS))
            self.lgbm_params['min_child_samples'] = param_value

        with _timer() as t:
            booster = lgb.train(self.lgbm_params, self.train_set, **self.lgbm_kwargs)

        val_score = self.get_booster_best_score(booster)
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


class LGBMAutoTune(BaseTuner):
    """Hyperparameter-tuning with Optuna for LightGBM.
    """

    def __init__(
            self,
            params,  # type: Dict[str, Any]
            train_set,  # type: lgb.Dataset
            num_boost_round=1000,  # type: int
            valid_sets=None,  # type: Optional[Any]
            valid_names=None,  # type: Optional[Any]
            fobj=None,
            feval=None,
            feature_name='auto',  # type: str
            categorical_feature='auto',  # type: str
            early_stopping_rounds=None,  # type: Optional[int]
            evals_result=None,
            verbose_eval=True,
            learning_rates=None,
            keep_training_booster=False,
            callbacks=None,
            time_budget=None,  # type: Optional[int]
            sample_size=None,  # type: Optional[int]
            best_params=None,  # type: Optional[Dict[str, Any]]
            tuning_history=None,  # type: Optional[List[Dict[str, Any]]]
            enable_adjusting_lr=False,  # type: bool
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
                      sample_size=sample_size)  # type: Dict[str, Any]
        self._parse_args(*args, **kwargs)
        self.best_booster = None

        self.best_score = -np.inf if self.higher_is_better() else np.inf
        self.best_params = best_params if best_params is not None else {}
        self.tuning_history = tuning_history if tuning_history is not None else []
        self.enable_adjusting_lr = enable_adjusting_lr

        # Check optuna logging
        self.is_optuna_logging_enabled = (optuna.logging._default_handler is None)

        # Check args
        if early_stopping_rounds is None:
            self._suggest_early_stopping_rounds()
        if valid_sets is None:
            raise ValueError("`valid_sets` is required.")

    def get_params(self):
        params = copy.deepcopy(self.lgbm_params)
        params.update(self.best_params)
        return params

    def _parse_args(self, *args, **kwargs):
        self.auto_options = {
            option_name: kwargs.get(option_name)
            for option_name in [
                'time_budget',
                'sample_size',
                'best_params',
                'tuning_history'
                'enable_adjusting_lr'
            ]
        }

        # Split options
        for option_name in self.auto_options.keys():
            if option_name in kwargs:
                del kwargs[option_name]

        self.lgbm_params = args[0]
        self.train_set = args[1]
        self.train_subset = None  # Use for sampling
        self.lgbm_kwargs = kwargs

        # Keep original kwargs
        self.original_lgbm_kwargs = kwargs.copy()
        self.original_lgbm_params = self.lgbm_params.copy()

    def _suggest_early_stopping_rounds(self):
        num_boost_round = self.lgbm_kwargs.get('num_boost_round', 1000)
        early_stopping_rounds = min(int(num_boost_round * 0.05), 50)
        return early_stopping_rounds

    def run(self):
        """Perform the hyperparameter-tuning with given parameters.

        Returns:

            booster : Booster
                The trained Booster model.
        """
        # Surpress log messages
        optuna.logging.disable_default_handler()
        self.lgbm_params['verbose'] = -1
        self.lgbm_params['seed'] = 111
        self.lgbm_kwargs['verbose_eval'] = False

        # Handling aliases
        handling_alias_parameters(self.lgbm_params)

        # Sampling
        self.sampling_train_set()

        # Tuning
        time_budget = self.auto_options['time_budget']

        self.start_time = time.time()
        with _timer() as t:
            self.tune_feature_fraction()
            if time_budget is not None and time_budget > t.elapsed_secs():
                self.best_params.update(self.get_params())
                return self.best_booster

            self.tune_num_leaves()
            if time_budget is not None and time_budget > t.elapsed_secs():
                self.best_params.update(self.get_params())
                return self.best_booster

            self.tune_bagging_fraction()
            if time_budget is not None and time_budget > t.elapsed_secs():
                self.best_params.update(self.get_params())
                return self.best_booster

            self.tune_feature_fraction_stage2()
            if time_budget is not None and time_budget > t.elapsed_secs():
                self.best_params.update(self.get_params())
                return self.best_booster

            self.tune_regularization_factors()
            if time_budget is not None and time_budget > t.elapsed_secs():
                self.best_params.update(self.get_params())
                return self.best_booster

            self.tune_min_data_in_leaf()
            if time_budget is not None and time_budget > t.elapsed_secs():
                self.best_params.update(self.get_params())
                return self.best_booster

            if self.enable_adjusting_lr:
                self.tune_learning_rate()

        self.best_params.update(self.get_params())
        return self.best_booster

    def sampling_train_set(self):
        # type: () -> None
        """ Make subset of `self.train_set` Dataset object
        """
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

    def tune_bagging_fraction(self, n_trials=7):
        # type: (int) -> None
        param_name = 'bagging_fraction'
        param_values = np.linspace(0.4, 1.0, n_trials)
        sampler = _GridSamplerUniform1D(param_name, param_values)
        self.tune_params([param_name], len(param_values), sampler)

    def tune_feature_fraction_stage2(self, n_trials=6):
        # type: (int) -> None
        param_name = 'feature_fraction'
        param_values = list(np.linspace(
            self.lgbm_params[param_name] - 0.08,
            self.lgbm_params[param_name] + 0.08,
            n_trials))
        param_values = [val for val in param_values if val >= 0.0 and val <= 1.0]
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

        # Set current best parameters
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
        study.optimize(objective, n_trials=n_trials)

        pbar.close()
        del pbar

        # Add tuning history
        self.tuning_history += objective.report

        updated_params = {p: study.best_trial.params[p] for p in target_param_names}
        self.lgbm_params.update(updated_params)
        self.best_params.update(updated_params)

        if self.compare_validation_metrics(study.best_value, self.best_score):
            self.best_score = study.best_value
            self.best_booster = objective.best_booster

    def tune_learning_rate(self):
        # type: () -> None

        # Update parameter
        self.lgbm_params.update(self.best_params)

        if self.higher_is_better():
            best_model_running_time = list(sorted(
                self.tuning_history, key=lambda x: x['val_score']))[-1]['elapsed_secs']
        else:
            best_model_running_time = list(sorted(
                self.tuning_history, key=lambda x: x['val_score']))[0]['elapsed_secs']

        sec_per_round = best_model_running_time / self.lgbm_kwargs['num_boost_round']
        time_budget = self.auto_options['time_budget']
        if time_budget is None:
            time_budget = DEFAULT_TIME_BUDGET_FOR_TUNING_LR

        max_feasible_rounds = int(time_budget / sec_per_round)
        if max_feasible_rounds > 10000:
            num_boost_round = 10000
            n_trials = math.floor(max_feasible_rounds / num_boost_round)

            predefined_params = {
                'lgbm_kwargs': {
                    'num_boost_round': num_boost_round,
                    'early_stopping_rounds': int(num_boost_round / 20),
                    'verbose_eval': int(num_boost_round / 10),
                },
                'lgbm_params': {},
            }  # type: Dict[str, Dict[str, Any]]

        elif max_feasible_rounds > 1000:
            num_boost_round = 1000
            n_trials = math.floor(max_feasible_rounds / num_boost_round)

            predefined_params = {
                'lgbm_kwargs': {
                    'num_boost_round': num_boost_round,
                    'early_stopping_rounds': int(num_boost_round / 20),
                    'verbose_eval': int(num_boost_round / 10),
                },
                'lgbm_params': {}
            }
        else:
            n_trials = math.floor(max_feasible_rounds / self.lgbm_kwargs['num_boost_round'])
            predefined_params = {
                'lgbm_kwargs': {},
                'lgbm_params': {
                    'learning_rate': {},
                },
            }

        # Adjusting learning rate
        if not self.enable_adjusting_lr:
            return

        if n_trials <= 1:
            predefined_params['lgbm_params']['learning_rate'] = [0.01]
        elif n_trials == 2:
            predefined_params['lgbm_params']['learning_rate'] = [0.01, 0.001]
        else:
            predefined_params['lgbm_params']['learning_rate'] = [0.01, 0.003, 0.001]

        # Fix num_boost_round and early_stopping_rounds
        for kwargs_name in predefined_params['lgbm_kwargs'].keys():
            self.lgbm_kwargs[kwargs_name] = predefined_params['lgbm_kwargs'][kwargs_name]

        for i_trial, lr in enumerate(predefined_params['lgbm_params']['learning_rate']):
            self.lgbm_params['learning_rate'] = lr

            with _timer() as t:
                train_set = self.train_set
                if self.train_subset is not None:
                    train_set = self.train_subset

                booster = lgb.train(self.lgbm_params, train_set, **self.lgbm_kwargs)

            val_score = self.get_booster_best_score(booster)
            elapsed_secs = t.elapsed_secs()
            average_iteration_time = elapsed_secs / booster.current_iteration()
            if self.compare_validation_metrics(val_score, self.best_score):
                self.best_score = val_score
                self.best_booster = booster

                self.tuning_history.append(dict(
                    action='adjust_learning_rate',
                    trial=i_trial,
                    value=lr,
                    val_score=val_score,
                    elapsed_secs=elapsed_secs,
                    average_iteration_time=average_iteration_time))
            else:
                # End if lower lr got worse result.
                break

            # Break the time limitation
            if time.time() > self.start_time + time_budget:
                break
