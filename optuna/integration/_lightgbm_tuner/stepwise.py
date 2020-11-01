from collections.abc import Sequence
import copy
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generator
from typing import Iterator
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

import optuna
from optuna import stepwise
from optuna._imports import try_import
from optuna.integration._lightgbm_tuner.alias import _handling_alias_parameters
from optuna.integration._lightgbm_tuner.steps import default_lgb_steps
from optuna.study import Study
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import Trial


with try_import() as _imports:
    import lightgbm as lgb
    from sklearn.model_selection import BaseCrossValidator

    ValidSet = Union[List[lgb.Dataset], Tuple[lgb.Dataset, ...], lgb.Dataset]
    _IntPair = Tuple[int, int]
    Folds = Union[Generator[_IntPair, None, None], Iterator[_IntPair], "BaseCrossValidator"]


_BOOSTER_KEY = "__Lgb_BOOSTER__"
_BEST_ITERATION_KEY = "__Lgb_BEST_ITERATION__"

# Default parameter values described in the official webpage.
_DEFAULT_LIGHTGBM_PARAMETERS = {
    "lambda_l1": 0.0,
    "lambda_l2": 0.0,
    "num_leaves": 31,
    "feature_fraction": 1.0,
    "bagging_fraction": 1.0,
    "bagging_freq": 0,
    "min_child_samples": 20,
    "feature_pre_filter": False,
}

_MAXIMIZED_METRICS = (
    "ndcg",
    "lambdarank",
    "rank_xendcg",
    "xendcg",
    "xe_ndcg",
    "xe_ndcg_mart",
    "xendcg_mart",
    "map",
    "mean_average_precision",
    "auc",
)

_AnyCallable = Callable[..., Any]


class _LgbStepObjective:
    def __init__(
        self, direction: StudyDirection, train_set: "lgb.Dataset", train_kwargs: Dict[str, Any]
    ) -> None:
        self.direction = direction
        self.train_set = train_set
        self.train_kwargs = train_kwargs
        self.first_dataset: str = None
        self.first_metric: str = None

    def __call__(self, trial: Trial, params: Dict[str, Any]) -> float:
        train_kwargs = copy.copy(self.train_kwargs)
        train_kwargs = self._add_first_metric_cb(train_kwargs)

        valid_sets = self.train_kwargs.get("valid_sets", None)
        if valid_sets:
            train_set, train_kwargs["valid_sets"] = self._copy_datasets(self.train_set, valid_sets)
        else:
            train_set = copy.copy(self.train_set)

        booster = lgb.train(params, train_set, **train_kwargs)
        self._serialize_booster(booster, trial)
        score = booster.best_score[self.first_dataset][self.first_metric]
        return score

    def _copy_datasets(
        self, train_set: "lgb.Dataset", valid_sets: "ValidSet"
    ) -> Tuple["lgb.Dataset", "ValidSet"]:
        copied_train_set = copy.copy(train_set)

        if isinstance(valid_sets, lgb.Dataset):
            valid_sets = [valid_sets]

        # lgb.train finds train sets in valid_sets with the `is` operator.
        # We need to make sure to keep the same reference for all train sets when we copy.
        copied_valid_sets = []
        for valid_set in valid_sets:
            if valid_set is self.train_set:
                copied_valid_set = copied_train_set
            else:
                copied_valid_set = copy.copy(valid_set)
            copied_valid_sets.append(copied_valid_set)

        return copied_train_set, copied_valid_sets

    def _serialize_booster(self, booster: "lgb.Booster", trial: Trial) -> None:
        serialized_model = booster.model_to_string()
        trial.set_system_attr(_BOOSTER_KEY, serialized_model)
        trial.set_system_attr(_BEST_ITERATION_KEY, booster.best_iteration)

    def _add_first_metric_cb(self, train_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve the first metric of the first dataset, in the order defined by LightGBM.

        Note:
            LightGBM order when 'first_metric_only'=True:
                1. If 'metrics' is in params, take first metric
                2. If 'metrics' is not in params, use a default metric based on the objective
                3. When using a custom evaluation function feval, 'metrics' must be set to the
                   string "None".

            The callback retrieves the first metric from LightGBM internals to avoid
            harcoding the above logic.
        """

        def _callback(env: "lgb.callback.CallbackEnv") -> None:
            dataset_name, eval_name, _, higher_better, *_ = env.evaluation_result_list[0]
            # split is needed for "<dataset type> <metric>" case (e.g. "train l1")
            self.first_metric = eval_name.split(" ")[-1]

            direction = StudyDirection.MAXIMIZE if higher_better else StudyDirection.MINIMIZE
            if direction != self.direction:
                raise ValueError(
                    f"Study direction is inconsistent with the metric '{self.first_metric}'. "
                    + f"Please set '{direction.name.lower()}' as the direction."
                )

            # lgb.train: find first non-training dataset, if none defaults to first dataset
            self.first_dataset = dataset_name
            for dataset_name, eval_name, _, higher_better, *_ in env.evaluation_result_list:
                if dataset_name != getattr(env.model, "_train_data_name", None):
                    self.first_dataset = dataset_name
                    break

        _callback.order = 99  # type: ignore

        callbacks = train_kwargs.pop("callbacks", None) or []
        train_kwargs["callbacks"] = callbacks + [_callback]

        return train_kwargs


class _LgbStepObjectiveCV(_LgbStepObjective):
    def __call__(self, trial: Trial, params: Dict[str, Any]) -> float:
        train_set = copy.copy(self.train_set)
        train_kwargs = copy.copy(self.train_kwargs)
        train_kwargs = self._add_first_metric_cb(train_kwargs)

        cv_results = lgb.cv(params, train_set, **train_kwargs)
        metric_key = f"{self.first_metric}-mean"

        if "return_cvbooster" in self.train_kwargs:
            booster = cv_results["cvbooster"]
            self._serialize_booster(booster, trial)

        return cv_results[metric_key][-1]

    def _serialize_booster(self, booster: "lgb.CVBooster", trial: Trial) -> None:
        serialized_models = [bst.model_to_string() for bst in booster.boosters]
        trial.set_system_attr(_BOOSTER_KEY, serialized_models)
        trial.set_system_attr(_BEST_ITERATION_KEY, booster.best_iteration)


def _infer_direction(params: Dict[str, Any], feval: _AnyCallable = None) -> str:
    metric = "binary_logloss"
    for alias in ("metric", "metrics", "metric_types"):
        if alias in params:
            metric = params[alias]
            break  # keep first alias found (similar to lgb logic)

    if isinstance(metric, Sequence):
        metric = metric[0]  # first_metric_only is mandatory

    if metric in {"None", "na", "null", "custom"} and feval:
        raise ValueError(
            "Direction cannot be automatically inferred from 'feval'. "
            + "Please specify a study with an appropriate direction."
        )

    if metric.startswith(_MAXIMIZED_METRICS) and metric != "mape":
        return "maximize"
    else:
        return "minimize"


def _keep_best_model(study: Study, trial: FrozenTrial) -> None:
    """Avoid bloating memory with non-optimal boosters."""
    if not stepwise.is_better(study.direction, trial.value, study.best_value):
        del trial.system_attrs[_BOOSTER_KEY]
        del trial.system_attrs[_BEST_ITERATION_KEY]


class _BaseLGBTuner(stepwise.StepwiseTuner):
    def __init__(
        self,
        lgb_objective: Type[_LgbStepObjective],
        params: Dict[str, Any],
        train_set: "lgb.Dataset",
        steps: Optional[stepwise.StepListType] = None,
        study: Optional[Study] = None,
        default_params: Dict[str, Any] = None,
        **train_kwargs: Any,
    ) -> None:
        _imports.check()

        if not params.get("first_metric_only", True):
            raise ValueError(
                "Optuna only handles a single metric. Please set 'first_metric_only' to True."
            )

        # Should not alter data since `min_data_in_leaf` is tuned.
        # https://lightgbm.readthedocs.io/en/latest/Parameters.html#feature_pre_filter
        if params.get("feature_pre_filter", False):
            self.logger.warn(
                "feature_pre_filter is given as True but will be set to False. This is required "
                "for the tuner to tune min_data_in_leaf."
            )

        params = copy.deepcopy(params)
        _handling_alias_parameters(params)
        base_params = copy.deepcopy(default_params or _DEFAULT_LIGHTGBM_PARAMETERS)
        base_params.update(params)
        base_params["first_metric_only"] = True
        base_params["feature_pre_filter"] = False

        if not study:
            feval = train_kwargs.get("feval", None)
            direction = _infer_direction(params, feval)
            study = optuna.create_study(direction=direction)

        objective = lgb_objective(study.direction, train_set, train_kwargs)

        if not steps:
            steps = default_lgb_steps()

        super().__init__(objective, steps, base_params, study)

    def _check_best_booster(self) -> None:
        if not self.study.trials:
            raise ValueError("The best booster is not available because no trials completed.")

    def get_best_booster(self) -> "lgb.Booster":
        """Return the best booster."""
        self._check_best_booster()
        serialized_model = self.study.best_trial.system_attrs[_BOOSTER_KEY]
        params = {**self.default_params, **self.best_params}
        booster = lgb.Booster(params=params, model_str=serialized_model, silent=True)

        # LightGBM does not serialize best_iteration and best_score.
        booster.best_iteration = self.study.best_trial.system_attrs[_BEST_ITERATION_KEY]
        booster.best_score = self.study.best_value

        return booster

    def _optimize_step(
        self,
        step_name: str,
        step: stepwise.Step,
        n_trials: Optional[int] = None,
        timeout: Optional[float] = None,
        n_jobs: int = 1,
        catch: Tuple[Type[Exception], ...] = (),
        callbacks: Optional[List[stepwise._OptunaCallback]] = None,
        gc_after_trial: bool = False,
        show_progress_bar: bool = False,
    ) -> None:
        callbacks = callbacks or []
        callbacks.append(_keep_best_model)
        super()._optimize_step(
            step_name,
            step,
            n_trials,
            timeout,
            n_jobs,
            catch,
            callbacks,
            gc_after_trial,
            show_progress_bar,
        )


class StepwiseLightGBMTuner(_BaseLGBTuner):
    """Hyperparameter tuner for LightGBM.

    By default, it optimizes the following hyperparameters in a stepwise manner:
    ``lambda_l1``, ``lambda_l2``, ``num_leaves``, ``feature_fraction``, ``bagging_fraction``,
    ``bagging_freq`` and ``min_child_samples``.

    You can find the details of the algorithm and benchmark results in `this blog article <https:/
    /medium.com/optuna/lightgbm-tuner-new-optuna-integration-for-hyperparameter-optimization-8b709
    5e99258>`_ by `Kohei Ozaki <https://www.kaggle.com/confirm>`_, a Kaggle Grandmaster.

    Any positional and keyword arguments for `lightgbm.train()`_ can be passed.
    The arguments specific to :class:`~.StepwiseLightGBMTuner` are listed below:

    Args:
        objective:
             A callable that implements objective function. The callable must accept
             a class:`~optuna.Trial` object and a dictionary of parameters to optimize.
        steps:
            List of (step_name, :class:`~.Step`) tuples that will be optimized in the
            in the order in which they are listed.
        default_params:
            The parameters that will serve as a baseline for the optimization in order
            to avoid performance regression. If :obj:`None`, default `LightGBM parameters`_
            are used.
        study:
            The study that will hold the trials for the sequence of steps. If :obj:`None`,
            a default study is created.

    Attributes:
        study:
            The study holding the trials.

    Notes:
        * If more than one ``metric`` is supplied in the ``params``, only the first one is used
          for the optimization. Consequently, `first_metric_only`_ must be omitted or set to
          d``True``.
        * If more than one ``valid_sets`` is supplied in the ``params``, only the first one is
          used for the optimization.

    .. _lightgbm.train(): https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.train.html
    .. _LightGBM parameters: https://lightgbm.readthedocs.io/en/latest/Parameters.html
    .. _first_metric_only: https://lightgbm.readthedocs.io/en/latest/Parameters.html#first_metric_only # noqa: E501
    """

    def __init__(
        self,
        params: Dict[str, Any],
        train_set: "lgb.Dataset",
        steps: Optional[stepwise.StepListType] = None,
        default_params: Optional[Dict[str, Any]] = None,
        study: Optional[Study] = None,
        num_boost_round: int = 1000,
        valid_sets: Optional["ValidSet"] = None,
        valid_names: Optional[Any] = None,
        fobj: Optional[_AnyCallable] = None,
        feval: Optional[_AnyCallable] = None,
        init_model: Optional[Union[str, "lgb.Booster"]] = None,
        feature_name: str = "auto",
        categorical_feature: str = "auto",
        early_stopping_rounds: Optional[int] = None,
        evals_result: Optional[Dict[Any, Any]] = None,
        verbose_eval: Optional[Union[bool, int]] = True,
        learning_rates: Optional[List[float]] = None,
        keep_training_booster: bool = False,
        callbacks: Optional[List[_AnyCallable]] = None,
    ) -> None:
        train_kwargs = locals()
        for non_kwarg in ("self", "__class__", "params", "train_set"):
            del train_kwargs[non_kwarg]

        super().__init__(_LgbStepObjective, params, train_set, **train_kwargs)

        if valid_sets is None:
            raise ValueError("'valid_sets' is required.")

        if isinstance(valid_sets, Sequence) and len(valid_sets) > 1:
            self.logger.warning(
                "Detected multiple 'valid_sets', "
                + "the first non-training dataset, if any, will be used for the optimization."
            )


class StepwiseLightGBMTunerCV(_BaseLGBTuner):
    """Hyperparameter tuner for LightGBM with cross-validation.

    It employs the same stepwise approach as :class:`~.StepwiseLightGBMTuner`.
    :class:`~.StepwiseLightGBMTunerCV` invokes `lightgbm.cv()`_ to train and validate boosters while
    :class:`~StepwiseLightGBMTuner` invokes `lightgbm.train()`_.

    Any positional and keyword arguments for `lightgbm.cv()`_ can be passed.
    The arguments specific to :class:`~.StepwiseLightGBMTuner` are listed below:

    Args:
        objective:
             A callable that implements objective function. The callable must accept
             a class:`~optuna.Trial` object and a dictionary of parameters to optimize.
        steps:
            List of (step_name, :class:`~.Step`) tuples that will be optimized in the
            in the order in which they are listed.
        default_params:
            The parameters that will serve as a baseline for the optimization in order
            to avoid performance regression. If :obj:`None`, default `LightGBM parameters`_
            are used.
        study:
            The study that will hold the trials for the sequence of steps. If :obj:`None`,
            a default study is created.

    Attributes:
        study:
            The study holding the trials.

    Notes:
        If more than one ``metric`` is supplied, only the first one is used for the optimization.
        Consequently, `first_metric_only`_ must be omitted or set to ``True`` in the ``params``.

    .. _lightgbm.train(): https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.train.html
    .. _lightgbm.cv(): https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.cv.html
    .. _LightGBM parameters: https://lightgbm.readthedocs.io/en/latest/Parameters.html
    .. _first_metric_only: https://lightgbm.readthedocs.io/en/latest/Parameters.html#first_metric_only # noqa: E501
    """

    def __init__(
        self,
        params: Dict[str, Any],
        train_set: "lgb.Dataset",
        steps: Optional[stepwise.StepListType] = None,
        study: Optional[Study] = None,
        num_boost_round: int = 1000,
        folds: Optional["Folds"] = None,
        nfold: int = 5,
        stratified: bool = True,
        shuffle: bool = True,
        metrics: Optional[Union[str, List[str]]] = None,
        fobj: Optional[_AnyCallable] = None,
        feval: Optional[_AnyCallable] = None,
        init_model: Optional[Union[str, "lgb.Booster"]] = None,
        feature_name: str = "auto",
        categorical_feature: str = "auto",
        early_stopping_rounds: Optional[int] = None,
        fpreproc: Optional[_AnyCallable] = None,
        verbose_eval: Optional[Union[bool, int]] = True,
        show_stdv: bool = True,
        seed: int = 0,
        callbacks: Optional[List[_AnyCallable]] = None,
        eval_train_metric: Optional[bool] = False,
    ) -> None:
        cv_kwargs = locals()
        for non_kwarg in ("self", "__class__", "params", "train_set"):
            del cv_kwargs[non_kwarg]

        if lgb.__version__ >= "3.0.0":
            cv_kwargs["return_cvbooster"] = True

        super().__init__(_LgbStepObjectiveCV, params, train_set, **cv_kwargs)

    def get_best_booster(self) -> "lgb.Booster":
        """Return the best booster. Requires lightgbm >= 3.0.0 (addition of ``CVBooster``)."""
        self._check_best_booster()
        if lgb.__version__ < "3.0.0":
            raise ValueError("'get_best_booster' requires lightgbm >= 3.0.0")

        params = {**self.default_params, **self.best_params}
        boosters = []
        for serialized_model in self.study.best_trial.system_attrs[_BOOSTER_KEY]:
            bst = lgb.Booster(params=params, model_str=serialized_model, silent=True)
            boosters.append(bst)
        booster = lgb.CVBooster()
        booster.boosters = boosters

        # LightGBM does not serialize best_iteration and best_score.
        setattr(booster, "best_iteration", self.study.best_trial.system_attrs[_BEST_ITERATION_KEY])
        setattr(booster, "best_score", self.study.best_value)

        return booster
