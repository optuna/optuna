import copy
import json
import os
import pickle
import time
from typing import Any
from typing import Dict
import warnings

import lightgbm as lgb
import numpy as np
import tqdm

import optuna
from optuna.integration.lightgbm_tuner.alias import _handling_alias_metrics
from optuna.integration.lightgbm_tuner.alias import _handling_alias_parameters
from optuna import type_checking

if type_checking.TYPE_CHECKING:
    from typing import Callable  # NOQA
    from typing import Generator  # NOQA
    from typing import List  # NOQA
    from typing import Optional  # NOQA
    from typing import Tuple  # NOQA
    from typing import Union  # NOQA

    from optuna.trial import FrozenTrial  # NOQA
    from optuna.study import Study  # NOQA
    from optuna.trial import Trial  # NOQA

    VALID_SET_TYPE = Union[List[lgb.Dataset], Tuple[lgb.Dataset, ...], lgb.Dataset]


# Define key names of `Trial.system_attrs`.
_ELAPSED_SECS_KEY = "lightgbm_tuner:elapsed_secs"
_AVERAGE_ITERATION_TIME_KEY = "lightgbm_tuner:average_iteration_time"
_STEP_NAME_KEY = "lightgbm_tuner:step_name"
_LGBM_PARAMS_KEY = "lightgbm_tuner:lgbm_params"

# EPS is used to ensure that a sampled parameter value is in pre-defined value range.
EPS = 1e-12

# Default value of tree_depth, used for upper bound of num_leaves.
DEFAULT_TUNER_TREE_DEPTH = 8

# Default parameter values described in the official webpage.
DEFAULT_LIGHTGBM_PARAMETERS = {
    "lambda_l1": 0.0,
    "lambda_l2": 0.0,
    "num_leaves": 31,
    "feature_fraction": 1.0,
    "bagging_fraction": 1.0,
    "bagging_freq": 0,
    "min_child_samples": 20,
}

_logger = optuna.logging.get_logger(__name__)


class BaseTuner(object):
    def __init__(self, lgbm_params=None, lgbm_kwargs=None):
        # type: (Dict[str, Any], Dict[str,Any]) -> None

        # Handling alias metrics.
        if lgbm_params is not None:
            _handling_alias_metrics(lgbm_params)

        self.lgbm_params = lgbm_params or {}
        self.lgbm_kwargs = lgbm_kwargs or {}

    def _get_booster_best_score(self, booster):
        # type: (lgb.Booster) -> float

        metric = self.lgbm_params.get("metric", "binary_logloss")

        # todo (smly): This implementation is different logic from the LightGBM's python bindings.
        if type(metric) is str:
            pass
        elif type(metric) is list:
            metric = metric[-1]
        elif type(metric) is set:
            metric = list(metric)[-1]
        else:
            raise NotImplementedError
        valid_sets = self.lgbm_kwargs.get("valid_sets")  # type: Optional[VALID_SET_TYPE]

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

        metric = self._metric_with_eval_at(metric)
        val_score = booster.best_score[valid_name][metric]
        return val_score

    def _metric_with_eval_at(self, metric):
        # type: (str) -> str

        if metric != "ndcg" and metric != "map":
            return metric

        eval_at = self.lgbm_params.get("eval_at")
        if eval_at is None:
            eval_at = self.lgbm_params.get("{}_at".format(metric))
        if eval_at is None:
            eval_at = self.lgbm_params.get("{}_eval_at".format(metric))
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

    def higher_is_better(self):
        # type: () -> bool

        metric_name = self.lgbm_params.get("metric", "binary_logloss")
        return metric_name.startswith(("auc", "ndcg", "map"))

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
        step_name,  # type: str
        model_dir,  # type: Optional[str]
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
        self.best_booster_with_trial_number = None  # type: Optional[Tuple[lgb.Booster, int]]
        self.step_name = step_name
        self.model_dir = model_dir

        self._check_target_names_supported()

    def _check_target_names_supported(self):
        # type: () -> None

        supported_param_names = [
            "lambda_l1",
            "lambda_l2",
            "num_leaves",
            "feature_fraction",
            "bagging_fraction",
            "bagging_freq",
            "min_child_samples",
        ]
        for target_param_name in self.target_param_names:
            if target_param_name not in supported_param_names:
                raise NotImplementedError("Parameter `{}` is not supported for tunning.")

    def __call__(self, trial):
        # type: (Trial) -> float

        pbar_fmt = "{}, val_score: {:.6f}"

        if self.pbar is not None:
            self.pbar.set_description(pbar_fmt.format(self.step_name, self.best_score))

        if "lambda_l1" in self.target_param_names:
            self.lgbm_params["lambda_l1"] = trial.suggest_loguniform("lambda_l1", 1e-8, 10.0)
        if "lambda_l2" in self.target_param_names:
            self.lgbm_params["lambda_l2"] = trial.suggest_loguniform("lambda_l2", 1e-8, 10.0)
        if "num_leaves" in self.target_param_names:
            tree_depth = self.lgbm_params.get("max_depth", DEFAULT_TUNER_TREE_DEPTH)
            max_num_leaves = 2 ** tree_depth if tree_depth > 0 else 2 ** DEFAULT_TUNER_TREE_DEPTH
            self.lgbm_params["num_leaves"] = trial.suggest_int("num_leaves", 2, max_num_leaves)
        if "feature_fraction" in self.target_param_names:
            # `GridSampler` is used for sampling feature_fraction value.
            # The value 1.0 for the hyperparameter is always sampled.
            param_value = min(trial.suggest_uniform("feature_fraction", 0.4, 1.0 + EPS), 1.0)
            self.lgbm_params["feature_fraction"] = param_value
        if "bagging_fraction" in self.target_param_names:
            # `TPESampler` is used for sampling bagging_fraction value.
            # The value 1.0 for the hyperparameter might by sampled.
            param_value = min(trial.suggest_uniform("bagging_fraction", 0.4, 1.0 + EPS), 1.0)
            self.lgbm_params["bagging_fraction"] = param_value
        if "bagging_freq" in self.target_param_names:
            self.lgbm_params["bagging_freq"] = trial.suggest_int("bagging_freq", 1, 7)
        if "min_child_samples" in self.target_param_names:
            # `GridSampler` is used for sampling min_child_samples value.
            # The value 1.0 for the hyperparameter is always sampled.
            param_value = int(trial.suggest_uniform("min_child_samples", 5, 100 + EPS))
            self.lgbm_params["min_child_samples"] = param_value

        start_time = time.time()
        booster = lgb.train(self.lgbm_params, self.train_set, **self.lgbm_kwargs)

        val_score = self._get_booster_best_score(booster)
        elapsed_secs = time.time() - start_time
        average_iteration_time = elapsed_secs / booster.current_iteration()

        if self.model_dir is not None:
            path = os.path.join(self.model_dir, "{}.pkl".format(trial.number))
            with open(path, "wb") as fout:
                pickle.dump(booster, fout)
            _logger.info("The booster of trial#{} was saved as {}.".format(trial.number, path))

        if self.compare_validation_metrics(val_score, self.best_score):
            self.best_score = val_score
            self.best_booster_with_trial_number = (booster, trial.number)

        if self.pbar is not None:
            self.pbar.set_description(pbar_fmt.format(self.step_name, self.best_score))
            self.pbar.update(1)

        self.report.append(
            dict(
                # Since v1.2.0, action was concatenation of parameter names. Currently, it is
                # explicitly given to distinguish steps which tune the same parameters.
                action=self.step_name,
                trial=self.trial_count,
                value=str(trial.params),
                val_score=val_score,
                elapsed_secs=elapsed_secs,
                average_iteration_time=average_iteration_time,
            )
        )

        trial.set_system_attr(_ELAPSED_SECS_KEY, elapsed_secs)
        trial.set_system_attr(_AVERAGE_ITERATION_TIME_KEY, average_iteration_time)
        trial.set_system_attr(_STEP_NAME_KEY, self.step_name)
        trial.set_system_attr(_LGBM_PARAMS_KEY, json.dumps(self.lgbm_params))

        self.trial_count += 1

        return val_score


class LightGBMTuner(BaseTuner):
    """Hyperparameter-tuning with Optuna for LightGBM.

    Arguments and keyword arguments for `lightgbm.train()`_ can be passed.
    The arguments that only :class:`~optuna.integration.lightgbm.LightGBMTuner` has are listed
    below:

    Args:
        time_budget:
            A time budget for parameter tuning in seconds.

        best_params:
            A dictionary to store the best parameters.

            .. deprecated:: 1.4.0
                Please use the ``params`` attribute of the best booster, which is obtained by
                :meth:`~optuna.integration.lightgbm.LightGBMTuner.get_best_booster`.

        tuning_history:
            A List to store the history of parameter tuning.

            .. deprecated:: 1.4.0
                Please use the ``study`` argument to access optimization history.

        study:
            A :class:`~optuna.study.Study` instance to store optimization results. The
            :class:`~optuna.trial.Trial` instances in it has the following system attributes:
            ``elapsed_secs`` is the elapsed time since the optimization starts.
            ``average_iteration_time`` is the average time of iteration to train the booster
            model in the trial. ``lgbm_params`` is a JSON-serialized dictionary of LightGBM
            parameters used in the trial.

        optuna_callbacks:
            List of Optuna callback functions that are invoked at the end of each trial.
            Each function must accept two parameters with the following types in this order:
            :class:`~optuna.study.Study` and :class:`~optuna.FrozenTrial`.
            Please note that this is not a ``callbacks`` argument of `lightgbm.train()`_ .

        model_dir:
            A directory to save boosters. By default, it is set to :obj:`None` and no boosters are
            saved. Please set shared directory (e.g., directories on NFS) if you want to access
            :meth:`~optuna.integration.LightGBMTuner.get_best_booster` in distributed environments.
            Otherwise, it may raise :obj:`ValueError`. If the directory does not exist, it will be
            created. The filenames of the boosters will be ``{model_dir}/{trial_number}.pkl``
            (e.g., ``./boosters/0.pkl``).

    .. _lightgbm.train(): https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.train.html
    """

    def __init__(
        self,
        params,  # type: Dict[str, Any]
        train_set,  # type: lgb.Dataset
        num_boost_round=1000,  # type: int
        valid_sets=None,  # type: Optional[VALID_SET_TYPE]
        valid_names=None,  # type: Optional[Any]
        fobj=None,  # type: Optional[Callable[..., Any]]
        feval=None,  # type: Optional[Callable[..., Any]]
        feature_name="auto",  # type: str
        categorical_feature="auto",  # type: str
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
        study=None,  # type: Optional[Study]
        optuna_callbacks=None,  # type: Optional[List[Callable[[Study, FrozenTrial], None]]]
        model_dir=None,  # type: Optional[str]
        verbosity=1,  # type: Optional[int]
    ):
        # type: (...) -> None

        params = copy.deepcopy(params)

        # Handling alias metrics.
        _handling_alias_metrics(params)

        args = [params, train_set]
        kwargs = dict(
            num_boost_round=num_boost_round,
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
            sample_size=sample_size,
        )  # type: Dict[str, Any]
        self._parse_args(*args, **kwargs)
        self._best_booster_with_trial_number = None  # type: Optional[Tuple[lgb.Booster, int]]
        self._start_time = None  # type: Optional[float]
        self._model_dir = model_dir

        if self._model_dir is not None and not os.path.exists(self._model_dir):
            os.mkdir(self._model_dir)

        if best_params is not None:
            warnings.warn(
                "The `best_params` argument is deprecated. "
                "Please get the parameter values via `lightgbm.basic.Booster.params`.",
                DeprecationWarning,
            )

        if tuning_history is not None:
            warnings.warn(
                "The `tuning_history` argument is deprecated. "
                "Please use the `study` argument to access optimization history.",
                DeprecationWarning,
            )

        self._best_params = {} if best_params is None else best_params
        self.tuning_history = [] if tuning_history is None else tuning_history

        # Set default parameters as best.
        self._best_params.update(DEFAULT_LIGHTGBM_PARAMETERS)

        if study is None:
            self.study = optuna.create_study(
                direction="maximize" if self.higher_is_better() else "minimize"
            )
        else:
            self.study = study

        if self.higher_is_better():
            if self.study.direction != optuna.study.StudyDirection.MAXIMIZE:
                metric_name = self.lgbm_params.get("metric", "binary_logloss")
                raise ValueError(
                    "Study direction is inconsistent with the metric {}. "
                    "Please set 'maximize' as the direction.".format(metric_name)
                )
        else:
            if self.study.direction != optuna.study.StudyDirection.MINIMIZE:
                metric_name = self.lgbm_params.get("metric", "binary_logloss")
                raise ValueError(
                    "Study direction is inconsistent with the metric {}. "
                    "Please set 'minimize' as the direction.".format(metric_name)
                )

        if valid_sets is None:
            raise ValueError("`valid_sets` is required.")

        self._optuna_callbacks = optuna_callbacks

    @property
    def best_score(self) -> float:
        """"Return the score of the best booster."""
        try:
            return self.study.best_value
        except ValueError:
            # Return the default score because no trials have completed.
            return -np.inf if self.higher_is_better() else np.inf

    @property
    def best_params(self) -> Dict[str, Any]:
        """Return parameters of the best booster."""
        try:
            return json.loads(self.study.best_trial.system_attrs[_LGBM_PARAMS_KEY])
        except ValueError:
            # Return the default score because no trials have completed.
            params = copy.deepcopy(DEFAULT_LIGHTGBM_PARAMETERS)
            # self.lgbm_params may contain parameters given by users.
            params.update(self.lgbm_params)
            return params

    @property
    def best_booster(self) -> lgb.Booster:
        """Return the best booster.

        .. deprecated:: 1.4.0
            Please get the best booster via
            :class:`~optuna.integration.lightgbm.LightGBMTuner.get_best_booster` instead.
        """
        warnings.warn(
            "The `best_booster` attribute is deprecated. Please use `get_best_booster` instead.",
            DeprecationWarning,
        )

        return self.get_best_booster()

    def get_best_booster(self) -> lgb.Booster:
        """Return the best booster.

        If the best booster cannot be found, :class:`ValueError` will be raised. To prevent the
        errors, please save boosters by specifying the ``model_dir`` arguments of
        :meth:`~optuna.integration.lightgbm.LightGBMTuner.__init__` when you resume tuning
        or you run tuning in parallel.
        """
        if self._best_booster_with_trial_number is not None:
            if self._best_booster_with_trial_number[1] == self.study.best_trial.number:
                return self._best_booster_with_trial_number[0]
        if len(self.study.trials) == 0:
            raise ValueError("The best booster is not available because no trials completed.")

        # The best booster exists, but this instance does not have it.
        # This may be due to resuming or parallelization.
        if self._model_dir is None:
            raise ValueError(
                "The best booster cannot be found. It may be found in the other processes due to "
                "resuming or distributed computing. Please set the `model_dir` argument of "
                "`LightGBMTuner.__init__` and make sure that boosters are shared with all "
                "processes."
            )

        best_trial = self.study.best_trial
        path = os.path.join(self._model_dir, "{}.pkl".format(best_trial.number))
        if not os.path.exists(path):
            raise ValueError(
                "The best booster cannot be found in {}. If you execute `LightGBMTuner` in "
                "distributed environment, please use network file system (e.g., NFS) to share "
                "models with multiple workers.".format(self._model_dir)
            )

        with open(path, "rb") as fin:
            booster = pickle.load(fin)

        return booster

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
                "time_budget",
                "sample_size",
                "best_params",
                "tuning_history",
                "verbosity",
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

    def run(self) -> None:
        """Perform the hyperparameter-tuning with given parameters."""
        # Suppress log messages.
        if self.auto_options["verbosity"] == 0:
            optuna.logging.disable_default_handler()
            self.lgbm_params["verbose"] = -1
            self.lgbm_params["seed"] = 111
            self.lgbm_kwargs["verbose_eval"] = False

        # Handling aliases.
        _handling_alias_parameters(self.lgbm_params)

        # Sampling.
        self.sample_train_set()

        self.tune_feature_fraction()
        self.tune_num_leaves()
        self.tune_bagging()
        self.tune_feature_fraction_stage2()
        self.tune_regularization_factors()
        self.tune_min_data_in_leaf()

    def sample_train_set(self):
        # type: () -> None
        """Make subset of `self.train_set` Dataset object."""

        if self.auto_options["sample_size"] is None:
            return

        self.train_set.construct()
        n_train_instance = self.train_set.get_label().shape[0]
        if n_train_instance > self.auto_options["sample_size"]:
            offset = n_train_instance - self.auto_options["sample_size"]
            idx_list = offset + np.arange(self.auto_options["sample_size"])
            self.train_subset = self.train_set.subset(idx_list)

    def tune_feature_fraction(self, n_trials=7):
        # type: (int) -> None

        param_name = "feature_fraction"
        param_values = np.linspace(0.4, 1.0, n_trials).tolist()

        # TODO(toshihikoyanase): Remove catch_warnings after GridSampler becomes non-experimental.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=optuna.exceptions.ExperimentalWarning)
            sampler = optuna.samplers.GridSampler({param_name: param_values})
        self.tune_params([param_name], len(param_values), sampler, "feature_fraction")

    def tune_num_leaves(self, n_trials=20):
        # type: (int) -> None

        self.tune_params(["num_leaves"], n_trials, optuna.samplers.TPESampler(), "num_leaves")

    def tune_bagging(self, n_trials=10):
        # type: (int) -> None

        self.tune_params(
            ["bagging_fraction", "bagging_freq"], n_trials, optuna.samplers.TPESampler(), "bagging"
        )

    def tune_feature_fraction_stage2(self, n_trials=6):
        # type: (int) -> None

        param_name = "feature_fraction"
        best_feature_fraction = self.best_params[param_name]
        param_values = np.linspace(
            best_feature_fraction - 0.08, best_feature_fraction + 0.08, n_trials
        ).tolist()
        param_values = [val for val in param_values if val >= 0.4 and val <= 1.0]

        # TODO(toshihikoyanase): Remove catch_warnings after GridSampler becomes non-experimental.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=optuna.exceptions.ExperimentalWarning)
            sampler = optuna.samplers.GridSampler({param_name: param_values})
        self.tune_params([param_name], len(param_values), sampler, "feature_fraction_stage2")

    def tune_regularization_factors(self, n_trials=20):
        # type: (int) -> None

        self.tune_params(
            ["lambda_l1", "lambda_l2"],
            n_trials,
            optuna.samplers.TPESampler(),
            "regularization_factors",
        )

    def tune_min_data_in_leaf(self):
        # type: () -> None

        param_name = "min_child_samples"
        param_values = [5, 10, 25, 50, 100]

        # TODO(toshihikoyanase): Remove catch_warnings after GridSampler becomes non-experimental.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=optuna.exceptions.ExperimentalWarning)
            sampler = optuna.samplers.GridSampler({param_name: param_values})
        self.tune_params([param_name], len(param_values), sampler, "min_data_in_leaf")

    def tune_params(self, target_param_names, n_trials, sampler, step_name):
        # type: (List[str], int, optuna.samplers.BaseSampler, str) -> None

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
            step_name=step_name,
            model_dir=self._model_dir,
            pbar=pbar,
        )

        study = self._create_stepwise_study(self.study, step_name)
        study.sampler = sampler

        complete_trials = [
            t
            for t in study.trials
            if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
        ]
        _n_trials = n_trials - len(complete_trials)

        if self._start_time is None:
            self._start_time = time.time()

        if self.auto_options["time_budget"] is not None:
            _timeout = self.auto_options["time_budget"] - (time.time() - self._start_time)
        else:
            _timeout = None
        if _n_trials > 0:
            try:
                study.optimize(
                    objective,
                    n_trials=_n_trials,
                    timeout=_timeout,
                    catch=(),
                    callbacks=self._optuna_callbacks,
                )
            except ValueError:
                # ValueError is raised by GridSampler when all combinations were examined.
                # TODO(toshihikoyanase): Remove this try-except after Study.stop is implemented.
                pass

        pbar.close()
        del pbar

        # Add tuning history.
        self.tuning_history += objective.report

        if objective.best_booster_with_trial_number is not None:
            self._best_booster_with_trial_number = objective.best_booster_with_trial_number
            self._best_params.update(self.best_params)

    def _create_stepwise_study(
        self, study: "optuna.study.Study", step_name: str
    ) -> "optuna.study.Study":

        # This class is assumed to be passed to a sampler and a pruner corresponding to the step.
        class _StepwiseStudy(optuna.study.Study):
            def __init__(self, study, step_name):
                # type: (optuna.study.Study, str) -> None

                super().__init__(
                    study_name=study.study_name,
                    storage=study._storage,
                    sampler=study.sampler,
                    pruner=study.pruner,
                )
                self._step_name = step_name

            def get_trials(self, deepcopy=True):
                # type: (bool) -> List[optuna.trial.FrozenTrial]

                trials = super().get_trials(deepcopy=deepcopy)
                return [t for t in trials if t.system_attrs.get(_STEP_NAME_KEY) == self._step_name]

            @property
            def best_trial(self):
                # type: () -> optuna.trial.FrozenTrial
                """Return the best trial in the study.

                Returns:
                    A :class:`~optuna.trial.FrozenTrial` object of the best trial.
                """

                trials = self.get_trials(deepcopy=False)
                trials = [t for t in trials if t.state is optuna.trial.TrialState.COMPLETE]

                if len(trials) == 0:
                    raise ValueError("No trials are completed yet.")

                if self.direction == optuna.study.StudyDirection.MINIMIZE:
                    best_trial = min(trials, key=lambda t: t.value)
                else:
                    best_trial = max(trials, key=lambda t: t.value)
                return copy.deepcopy(best_trial)

        return _StepwiseStudy(study, step_name)
