import contextlib
from tempfile import TemporaryDirectory
from typing import Any
from typing import Dict
from typing import Generator
from typing import List
from typing import Optional
from unittest import mock
import warnings

import numpy as np
import pytest

import optuna
import optuna.integration.lightgbm as lgb
from optuna.integration.lightgbm_tuner.optimize import BaseTuner
from optuna.integration.lightgbm_tuner.optimize import LightGBMTuner
from optuna.integration.lightgbm_tuner.optimize import LightGBMTunerCV
from optuna.integration.lightgbm_tuner.optimize import OptunaObjective
from optuna.integration.lightgbm_tuner.optimize import OptunaObjectiveCV
from optuna import type_checking

if type_checking.TYPE_CHECKING:
    from typing import Union  # NOQA

    from optuna.study import Study  # NOQA


@contextlib.contextmanager
def turnoff_train(metric: Optional[str] = "binary_logloss") -> Generator[None, None, None]:

    unexpected_value = 0.5
    dummy_num_iterations = 1234

    class DummyBooster(object):
        def __init__(self):
            # type: () -> None

            self.best_score = {
                "valid_0": {metric: unexpected_value},
            }

        def current_iteration(self):
            # type: () -> int

            return dummy_num_iterations

    dummy_booster = DummyBooster()

    with mock.patch("lightgbm.train", return_value=dummy_booster):
        yield


@contextlib.contextmanager
def turnoff_cv(metric: Optional[str] = "binary_logloss") -> Generator[None, None, None]:

    unexpected_value = 0.5
    dummy_results = {"{}-mean".format(metric): [unexpected_value]}

    with mock.patch("lightgbm.cv", return_value=dummy_results):
        yield


class TestOptunaObjective(object):
    def test_init_(self):
        # type: () -> None

        target_param_names = ["learning_rate"]  # Invalid parameter name.

        with pytest.raises(NotImplementedError) as execinfo:
            OptunaObjective(target_param_names, {}, None, {}, 0, "tune_learning_rate", None)

        assert execinfo.type is NotImplementedError

    def test_call(self):
        # type: () -> None

        target_param_names = ["lambda_l1"]
        lgbm_params = {}  # type: Dict[str, Any]
        train_set = lgb.Dataset(None)
        val_set = lgb.Dataset(None)

        lgbm_kwargs = {"valid_sets": val_set}
        best_score = -np.inf

        with turnoff_train():
            objective = OptunaObjective(
                target_param_names,
                lgbm_params,
                train_set,
                lgbm_kwargs,
                best_score,
                "tune_lambda_l1",
                None,
            )
            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=10)

            assert study.best_value == 0.5


class TestOptunaObjectiveCV(object):
    def test_call(self) -> None:
        target_param_names = ["lambda_l1"]
        lgbm_params = {}  # type: Dict[str, Any]
        train_set = lgb.Dataset(None)
        lgbm_kwargs = {}  # type: Dict[str, Any]
        best_score = -np.inf

        with turnoff_cv():
            objective = OptunaObjectiveCV(
                target_param_names,
                lgbm_params,
                train_set,
                lgbm_kwargs,
                best_score,
                "tune_lambda_l1",
                None,
            )
            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=10)

            assert study.best_value == 0.5


class TestBaseTuner(object):
    def test_get_booster_best_score(self):
        # type: () -> None

        expected_value = 1.0

        class DummyBooster(object):
            def __init__(self):
                # type: () -> None

                self.best_score = {"valid_0": {"binary_logloss": expected_value}}

        booster = DummyBooster()
        dummy_dataset = lgb.Dataset(None)

        tuner = BaseTuner(lgbm_kwargs=dict(valid_sets=dummy_dataset))
        val_score = tuner._get_booster_best_score(booster)
        assert val_score == expected_value

    def test_higher_is_better(self):
        # type: () -> None

        for metric in [
            "auc",
            "ndcg",
            "lambdarank",
            "rank_xendcg",
            "xendcg",
            "xe_ndcg",
            "xe_ndcg_mart",
            "xendcg_mart",
            "map",
            "mean_average_precision",
        ]:
            tuner = BaseTuner(lgbm_params={"metric": metric})
            assert tuner.higher_is_better()

        for metric in ["rmsle", "rmse", "binary_logloss"]:
            tuner = BaseTuner(lgbm_params={"metric": metric})
            assert not tuner.higher_is_better()

    def test_get_booster_best_score__using_valid_names_as_str(self):
        # type: () -> None

        expected_value = 1.0

        class DummyBooster(object):
            def __init__(self):
                # type: () -> None

                self.best_score = {"dev": {"binary_logloss": expected_value}}

        booster = DummyBooster()
        dummy_dataset = lgb.Dataset(None)

        tuner = BaseTuner(lgbm_kwargs={"valid_names": "dev", "valid_sets": dummy_dataset,})
        val_score = tuner._get_booster_best_score(booster)
        assert val_score == expected_value

    def test_get_booster_best_score__using_valid_names_as_list(self):
        # type: () -> None

        unexpected_value = 0.5
        expected_value = 1.0

        class DummyBooster(object):
            def __init__(self):
                # type: () -> None

                self.best_score = {
                    "train": {"binary_logloss": unexpected_value},
                    "val": {"binary_logloss": expected_value},
                }

        booster = DummyBooster()
        dummy_train_dataset = lgb.Dataset(None)
        dummy_val_dataset = lgb.Dataset(None)

        tuner = BaseTuner(
            lgbm_kwargs={
                "valid_names": ["train", "val"],
                "valid_sets": [dummy_train_dataset, dummy_val_dataset],
            }
        )
        val_score = tuner._get_booster_best_score(booster)
        assert val_score == expected_value

    def test_compare_validation_metrics(self):
        # type: () -> None

        for metric in [
            "auc",
            "ndcg",
            "lambdarank",
            "rank_xendcg",
            "xendcg",
            "xe_ndcg",
            "xe_ndcg_mart",
            "xendcg_mart",
            "map",
            "mean_average_precision",
        ]:
            tuner = BaseTuner(lgbm_params={"metric": metric})
            assert tuner.compare_validation_metrics(0.5, 0.1)
            assert not tuner.compare_validation_metrics(0.5, 0.5)
            assert not tuner.compare_validation_metrics(0.1, 0.5)

        for metric in ["rmsle", "rmse", "binary_logloss"]:
            tuner = BaseTuner(lgbm_params={"metric": metric})
            assert not tuner.compare_validation_metrics(0.5, 0.1)
            assert not tuner.compare_validation_metrics(0.5, 0.5)
            assert tuner.compare_validation_metrics(0.1, 0.5)

    @pytest.mark.parametrize(
        "metric, eval_at_param, expected",
        [
            ("auc", {"eval_at": 5}, "auc"),
            ("accuracy", {"eval_at": 5}, "accuracy"),
            ("rmsle", {"eval_at": 5}, "rmsle"),
            ("rmse", {"eval_at": 5}, "rmse"),
            ("binary_logloss", {"eval_at": 5}, "binary_logloss"),
            ("ndcg", {"eval_at": 5}, "ndcg@5"),
            ("ndcg", {"ndcg_at": 5}, "ndcg@5"),
            ("ndcg", {"ndcg_eval_at": 5}, "ndcg@5"),
            ("ndcg", {"eval_at": [20]}, "ndcg@20"),
            ("ndcg", {"eval_at": [10, 20]}, "ndcg@10"),
            ("ndcg", {}, "ndcg@1"),
            ("map", {"eval_at": 5}, "map@5"),
            ("map", {"eval_at": [20]}, "map@20"),
            ("map", {"eval_at": [10, 20]}, "map@10"),
            ("map", {}, "map@1"),
        ],
    )
    def test_metric_with_eval_at(self, metric, eval_at_param, expected):
        # type: (str, Dict[str, Union[int, List[int]]], str) -> None

        params = {"metric": metric}  # type: Dict[str, Union[str, int, List[int]]]
        params.update(eval_at_param)
        tuner = BaseTuner(lgbm_params=params)
        assert tuner._metric_with_eval_at(metric) == expected

    def test_metric_with_eval_at_error(self):
        # type: () -> None

        tuner = BaseTuner(lgbm_params={"metric": "ndcg", "eval_at": "1"})
        with pytest.raises(ValueError):
            tuner._metric_with_eval_at("ndcg")


class TestLightGBMTuner(object):
    def _get_tuner_object(self, params={}, train_set=None, kwargs_options={}, study=None):
        # type: (Dict[str, Any], lgb.Dataset, Dict[str, Any], Optional[Study]) -> lgb.LightGBMTuner

        # Required keyword arguments.
        dummy_dataset = lgb.Dataset(None)

        kwargs = dict(
            num_boost_round=5, early_stopping_rounds=2, valid_sets=dummy_dataset, study=study
        )
        kwargs.update(kwargs_options)

        runner = lgb.LightGBMTuner(params, train_set, **kwargs)
        return runner

    def test_no_eval_set_args(self):
        # type: () -> None

        params = {}  # type: Dict[str, Any]
        train_set = lgb.Dataset(None)
        with pytest.raises(ValueError) as excinfo:
            lgb.LightGBMTuner(params, train_set, num_boost_round=5, early_stopping_rounds=2)

        assert excinfo.type == ValueError
        assert str(excinfo.value) == "`valid_sets` is required."

    @pytest.mark.parametrize(
        "best_params, tuning_history", [({}, None), (None, []),],
    )
    def test_deprecated_args(
        self, best_params: Optional[Dict[str, Any]], tuning_history: Optional[List[Dict[str, Any]]]
    ) -> None:
        # Required keyword arguments.
        params = {}  # type: Dict[str, Any]
        train_set = lgb.Dataset(None)
        with pytest.warns(DeprecationWarning):
            lgb.LightGBMTuner(
                params,
                train_set,
                valid_sets=[train_set],
                best_params=best_params,
                tuning_history=tuning_history,
            )

    @pytest.mark.parametrize(
        "metric, study_direction",
        [
            ("auc", "minimize"),
            ("mse", "maximize"),
            (None, "maximize"),  # The default metric is binary_logloss.
        ],
    )
    def test_inconsistent_study_direction(self, metric: str, study_direction: str) -> None:

        params = {}  # type: Dict[str, Any]
        if metric is not None:
            params["metric"] = metric
        train_set = lgb.Dataset(None)
        valid_set = lgb.Dataset(None)
        study = optuna.create_study(direction=study_direction)
        with pytest.raises(ValueError) as excinfo:
            lgb.LightGBMTuner(
                params,
                train_set,
                valid_sets=[train_set, valid_set],
                num_boost_round=5,
                early_stopping_rounds=2,
                study=study,
            )

        assert excinfo.type == ValueError
        assert str(excinfo.value).startswith("Study direction is inconsistent with the metric")

    def test_with_minimum_required_args(self):
        # type: () -> None

        runner = self._get_tuner_object()
        assert "num_boost_round" in runner.lgbm_kwargs
        assert "num_boost_round" not in runner.auto_options
        assert runner.lgbm_kwargs["num_boost_round"] == 5

    def test__parse_args_wrapper_args(self):
        # type: () -> None

        params = {}  # type: Dict[str, Any]
        train_set = lgb.Dataset(None)
        val_set = lgb.Dataset(None)
        kwargs = dict(
            num_boost_round=12,
            early_stopping_rounds=10,
            valid_sets=val_set,
            time_budget=600,
            best_params={},
            sample_size=1000,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            runner = lgb.LightGBMTuner(params, train_set, **kwargs)
        new_args = ["time_budget", "time_budget", "best_params", "sample_size"]
        for new_arg in new_args:
            assert new_arg not in runner.lgbm_kwargs
            assert new_arg in runner.auto_options

    @pytest.mark.parametrize(
        "metric, study_direction, expected",
        [("auc", "maximize", -np.inf), ("mse", "minimize", np.inf),],
    )
    def test_best_score(self, metric: str, study_direction: str, expected: float) -> None:
        with turnoff_train(metric=metric):
            study = optuna.create_study(direction=study_direction)
            runner = self._get_tuner_object(
                params=dict(lambda_l1=0.0, metric=metric), kwargs_options={}, study=study,
            )
            assert runner.best_score == expected
            runner.tune_regularization_factors()
            assert runner.best_score == 0.5

    def test_best_params(self) -> None:
        unexpected_value = 20  # out of scope.

        with turnoff_train():
            study = optuna.create_study()
            runner = self._get_tuner_object(
                params=dict(lambda_l1=unexpected_value,), kwargs_options={}, study=study,
            )
            assert runner.best_params["lambda_l1"] == unexpected_value
            runner.tune_regularization_factors()
            assert runner.best_params["lambda_l1"] != unexpected_value

    def test_sample_train_set(self):
        # type: () -> None

        sample_size = 3

        X_trn = np.random.uniform(10, size=50).reshape((10, 5))
        y_trn = np.random.randint(2, size=10)
        train_dataset = lgb.Dataset(X_trn, label=y_trn)
        runner = self._get_tuner_object(
            train_set=train_dataset, kwargs_options=dict(sample_size=sample_size)
        )
        runner.sample_train_set()

        # Workaround for mypy.
        if not type_checking.TYPE_CHECKING:
            runner.train_subset.construct()  # Cannot get label before construct `lgb.Dataset`.
            assert runner.train_subset.get_label().shape[0] == sample_size

    def test_time_budget(self) -> None:
        unexpected_value = 1.1  # out of scope.

        with turnoff_train():
            runner = self._get_tuner_object(
                params=dict(
                    feature_fraction=unexpected_value,  # set default as unexpected value.
                ),
                kwargs_options=dict(time_budget=0,),
            )
            assert len(runner.study.trials) == 0
            # No trials run because `time_budget` is set to zero.
            runner.tune_feature_fraction()
            assert runner.lgbm_params["feature_fraction"] == unexpected_value
            assert len(runner.study.trials) == 0

    def test_tune_feature_fraction(self):
        # type: () -> None

        unexpected_value = 1.1  # out of scope.

        with turnoff_train():
            tuning_history = []  # type: List[Dict[str, float]]
            best_params = {}  # type: Dict[str, Any]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=DeprecationWarning)
                runner = self._get_tuner_object(
                    params=dict(
                        feature_fraction=unexpected_value,  # set default as unexpected value.
                    ),
                    kwargs_options=dict(tuning_history=tuning_history, best_params=best_params,),
                )
            assert len(tuning_history) == 0
            assert len(runner.study.trials) == 0
            runner.tune_feature_fraction()

            assert runner.lgbm_params["feature_fraction"] != unexpected_value
            assert len(tuning_history) == 7
            assert len(runner.study.trials) == 7

    def test_tune_num_leaves(self):
        # type: () -> None

        unexpected_value = 1  # out of scope.

        with turnoff_train():
            tuning_history = []  # type: List[Dict[str, float]]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=DeprecationWarning)
                runner = self._get_tuner_object(
                    params=dict(num_leaves=unexpected_value,),
                    kwargs_options=dict(tuning_history=tuning_history, best_params={},),
                )
            assert len(tuning_history) == 0
            assert len(runner.study.trials) == 0
            runner.tune_num_leaves()

            assert runner.lgbm_params["num_leaves"] != unexpected_value
            assert len(tuning_history) == 20
            assert len(runner.study.trials) == 20

    def test_tune_num_leaves_negative_max_depth(self):
        # type: () -> None

        params = {
            "metric": "binary_logloss",
            "max_depth": -1,
        }  # type: Dict[str, Any]
        X_trn = np.random.uniform(10, size=(10, 5))
        y_trn = np.random.randint(2, size=10)
        train_dataset = lgb.Dataset(X_trn, label=y_trn)
        valid_dataset = lgb.Dataset(X_trn, label=y_trn)

        tuning_history = []  # type: List[Dict[str, float]]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            runner = lgb.LightGBMTuner(
                params,
                train_dataset,
                num_boost_round=3,
                early_stopping_rounds=2,
                valid_sets=valid_dataset,
                tuning_history=tuning_history,
            )
        runner.tune_num_leaves()
        assert len(tuning_history) == 20
        assert len(runner.study.trials) == 20

    def test_tune_bagging(self):
        # type: () -> None

        unexpected_value = 1  # out of scope.

        with turnoff_train():
            tuning_history = []  # type: List[Dict[str, float]]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=DeprecationWarning)
                runner = self._get_tuner_object(
                    params=dict(bagging_fraction=unexpected_value,),
                    kwargs_options=dict(tuning_history=tuning_history, best_params={},),
                )
            assert len(tuning_history) == 0
            assert len(runner.study.trials) == 0
            runner.tune_bagging()

            assert runner.lgbm_params["bagging_fraction"] != unexpected_value
            assert len(tuning_history) == 10
            assert len(runner.study.trials) == 10

    def test_tune_feature_fraction_stage2(self):
        # type: () -> None

        unexpected_value = 0.5

        with turnoff_train():
            tuning_history = []  # type: List[Dict[str, float]]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=DeprecationWarning)
                runner = self._get_tuner_object(
                    params=dict(feature_fraction=unexpected_value,),
                    kwargs_options=dict(tuning_history=tuning_history, best_params={},),
                )
            assert len(tuning_history) == 0
            assert len(runner.study.trials) == 0
            runner.tune_feature_fraction_stage2()

            assert runner.lgbm_params["feature_fraction"] != unexpected_value
            assert len(tuning_history) == 6
            assert len(runner.study.trials) == 6

    def test_tune_regularization_factors(self):
        # type: () -> None

        unexpected_value = 20  # out of scope.

        with turnoff_train():
            tuning_history = []  # type: List[Dict[str, float]]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=DeprecationWarning)
                runner = self._get_tuner_object(
                    params=dict(lambda_l1=unexpected_value,),  # set default as unexpected value.
                    kwargs_options=dict(tuning_history=tuning_history, best_params={},),
                )
            assert len(tuning_history) == 0
            assert len(runner.study.trials) == 0
            runner.tune_regularization_factors()

            assert runner.lgbm_params["lambda_l1"] != unexpected_value
            assert len(tuning_history) == 20
            assert len(runner.study.trials) == 20

    def test_tune_min_data_in_leaf(self):
        # type: () -> None

        unexpected_value = 1  # out of scope.

        with turnoff_train():
            tuning_history = []  # type: List[Dict[str, float]]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=DeprecationWarning)
                runner = self._get_tuner_object(
                    params=dict(
                        min_child_samples=unexpected_value,  # set default as unexpected value.
                    ),
                    kwargs_options=dict(tuning_history=tuning_history, best_params={},),
                )
            assert len(tuning_history) == 0
            assert len(runner.study.trials) == 0
            runner.tune_min_data_in_leaf()

            assert runner.lgbm_params["min_child_samples"] != unexpected_value
            assert len(tuning_history) == 5
            assert len(runner.study.trials) == 5

    def test_when_a_step_does_not_improve_best_score(self):
        # type: () -> None

        params = {}  # type: Dict
        valid_data = np.zeros((10, 10))
        valid_sets = lgb.Dataset(valid_data)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            tuner = LightGBMTuner(params, None, valid_sets=valid_sets)
        assert not tuner.higher_is_better()

        with mock.patch("lightgbm.train"), mock.patch.object(
            BaseTuner, "_get_booster_best_score", return_value=0.9
        ):
            tuner.tune_feature_fraction()

        assert "feature_fraction" in tuner.best_params
        assert tuner.best_score == 0.9

        # Assume that tuning `num_leaves` doesn't improve the `best_score`.
        with mock.patch("lightgbm.train"), mock.patch.object(
            BaseTuner, "_get_booster_best_score", return_value=1.1
        ):
            tuner.tune_num_leaves()

    def test_resume_run(self) -> None:
        params = {"verbose": -1}  # type: Dict
        dataset = lgb.Dataset(np.zeros((10, 10)))

        study = optuna.create_study()
        tuner = LightGBMTuner(params, dataset, valid_sets=dataset, study=study)

        with mock.patch.object(BaseTuner, "_get_booster_best_score", return_value=1.0):
            tuner.tune_regularization_factors()

        n_trials = len(study.trials)
        assert n_trials == len(study.trials)

        tuner2 = LightGBMTuner(params, dataset, valid_sets=dataset, study=study)
        with mock.patch.object(BaseTuner, "_get_booster_best_score", return_value=1.0):
            tuner2.tune_regularization_factors()
        assert n_trials == len(study.trials)

    def test_get_best_booster(self) -> None:
        unexpected_value = 20  # out of scope.

        params = {"verbose": -1, "lambda_l1": unexpected_value}  # type: Dict
        dataset = lgb.Dataset(np.zeros((10, 10)))

        study = optuna.create_study()
        tuner = LightGBMTuner(params, dataset, valid_sets=dataset, study=study)

        with pytest.raises(ValueError):
            tuner.get_best_booster()

        with mock.patch.object(BaseTuner, "_get_booster_best_score", return_value=0.0):
            tuner.tune_regularization_factors()

        best_booster = tuner.get_best_booster()
        assert best_booster.params["lambda_l1"] != unexpected_value

        # TODO(toshihikoyanase): Remove this check when LightGBMTuner.best_booster is removed.
        with pytest.warns(DeprecationWarning):
            tuner.best_booster

        tuner2 = LightGBMTuner(params, dataset, valid_sets=dataset, study=study)

        # Resumed study does not have the best booster.
        with pytest.raises(ValueError):
            tuner2.get_best_booster()

    def test_best_booster_with_model_dir(self) -> None:
        params = {"verbose": -1}  # type: Dict
        dataset = lgb.Dataset(np.zeros((10, 10)))

        study = optuna.create_study()
        with TemporaryDirectory() as tmpdir:
            tuner = LightGBMTuner(
                params, dataset, valid_sets=dataset, study=study, model_dir=tmpdir
            )

            with mock.patch.object(BaseTuner, "_get_booster_best_score", return_value=0.0):
                tuner.tune_regularization_factors()

            best_booster = tuner.get_best_booster()

            tuner2 = LightGBMTuner(
                params, dataset, valid_sets=dataset, study=study, model_dir=tmpdir
            )
            best_booster2 = tuner2.get_best_booster()

            assert best_booster.params == best_booster2.params

    @pytest.mark.parametrize("direction, overall_best", [("minimize", 1), ("maximize", 2),])
    def test_create_stepwise_study(self, direction: str, overall_best: int) -> None:

        tuner = LightGBMTuner({}, None, valid_sets=lgb.Dataset(np.zeros((10, 10))))

        def objective(trial: optuna.trial.Trial, value: float) -> float:

            trial.set_system_attr(
                optuna.integration.lightgbm_tuner.optimize._STEP_NAME_KEY,
                "step{:.0f}".format(value),
            )
            return trial.suggest_uniform("x", value, value)

        study = optuna.create_study(direction=direction)
        study_step1 = tuner._create_stepwise_study(study, "step1")

        with pytest.raises(ValueError):
            study_step1.best_trial

        study_step1.optimize(lambda t: objective(t, 1), n_trials=1)

        study_step2 = tuner._create_stepwise_study(study, "step2")

        # `study` has a trial, but `study_step2` has no trials.
        with pytest.raises(ValueError):
            study_step2.best_trial

        study_step2.optimize(lambda t: objective(t, 2), n_trials=2)

        assert len(study_step1.trials) == 1
        assert len(study_step2.trials) == 2
        assert len(study.trials) == 3

        assert study_step1.best_trial.value == 1
        assert study_step2.best_trial.value == 2
        assert study.best_trial.value == overall_best

    def test_optuna_callback(self) -> None:
        params = {"verbose": -1}  # type: Dict[str, Any]
        dataset = lgb.Dataset(np.zeros((10, 10)))

        callback_mock = mock.MagicMock()

        study = optuna.create_study()
        tuner = LightGBMTuner(
            params, dataset, valid_sets=dataset, study=study, optuna_callbacks=[callback_mock],
        )

        with mock.patch.object(BaseTuner, "_get_booster_best_score", return_value=1.0):
            tuner.tune_params(["num_leaves"], 10, optuna.samplers.TPESampler(), "num_leaves")

        assert callback_mock.call_count == 10


class TestLightGBMTunerCV(object):
    def _get_tunercv_object(
        self,
        params: Dict[str, Any] = {},
        train_set: lgb.Dataset = None,
        kwargs_options: Dict[str, Any] = {},
        study: Optional[optuna.study.Study] = None,
    ) -> LightGBMTunerCV:

        # Required keyword arguments.
        kwargs = dict(
            num_boost_round=5, early_stopping_rounds=2, study=study
        )  # type: Dict[str, Any]
        kwargs.update(kwargs_options)

        runner = LightGBMTunerCV(params, train_set, **kwargs)
        return runner

    @pytest.mark.parametrize(
        "metric, study_direction",
        [
            ("auc", "minimize"),
            ("mse", "maximize"),
            (None, "maximize"),  # The default metric is binary_logloss.
        ],
    )
    def test_inconsistent_study_direction(self, metric: str, study_direction: str) -> None:

        params = {}  # type: Dict[str, Any]
        if metric is not None:
            params["metric"] = metric
        train_set = lgb.Dataset(None)
        study = optuna.create_study(direction=study_direction)
        with pytest.raises(ValueError) as excinfo:
            LightGBMTunerCV(
                params, train_set, num_boost_round=5, early_stopping_rounds=2, study=study,
            )

        assert excinfo.type == ValueError
        assert str(excinfo.value).startswith("Study direction is inconsistent with the metric")

    def test_with_minimum_required_args(self) -> None:

        runner = self._get_tunercv_object()
        assert "num_boost_round" in runner.lgbm_kwargs
        assert "num_boost_round" not in runner.auto_options
        assert runner.lgbm_kwargs["num_boost_round"] == 5

    def test_tune_feature_fraction(self) -> None:
        unexpected_value = 1.1  # out of scope.

        with turnoff_cv():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=DeprecationWarning)
                runner = self._get_tunercv_object(
                    params=dict(
                        feature_fraction=unexpected_value,  # set default as unexpected value.
                    ),
                )
            assert len(runner.study.trials) == 0
            runner.tune_feature_fraction()

            assert runner.lgbm_params["feature_fraction"] != unexpected_value
            assert len(runner.study.trials) == 7

    def test_tune_num_leaves(self) -> None:
        unexpected_value = 1  # out of scope.

        with turnoff_cv():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=DeprecationWarning)
                runner = self._get_tunercv_object(params=dict(num_leaves=unexpected_value,),)
            assert len(runner.study.trials) == 0
            runner.tune_num_leaves()

            assert runner.lgbm_params["num_leaves"] != unexpected_value
            assert len(runner.study.trials) == 20

    def test_tune_bagging(self) -> None:
        unexpected_value = 1  # out of scope.

        with turnoff_cv():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=DeprecationWarning)
                runner = self._get_tunercv_object(params=dict(bagging_fraction=unexpected_value,),)
            assert len(runner.study.trials) == 0
            runner.tune_bagging()

            assert runner.lgbm_params["bagging_fraction"] != unexpected_value
            assert len(runner.study.trials) == 10

    def test_tune_feature_fraction_stage2(self) -> None:
        unexpected_value = 0.5

        with turnoff_cv():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=DeprecationWarning)
                runner = self._get_tunercv_object(params=dict(feature_fraction=unexpected_value,),)
            assert len(runner.study.trials) == 0
            runner.tune_feature_fraction_stage2()

            assert runner.lgbm_params["feature_fraction"] != unexpected_value
            assert len(runner.study.trials) == 6

    def test_tune_regularization_factors(self) -> None:
        unexpected_value = 20  # out of scope.

        with turnoff_cv():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=DeprecationWarning)
                runner = self._get_tunercv_object(
                    params=dict(lambda_l1=unexpected_value,),  # set default as unexpected value.
                )
            assert len(runner.study.trials) == 0
            runner.tune_regularization_factors()

            assert runner.lgbm_params["lambda_l1"] != unexpected_value
            assert len(runner.study.trials) == 20

    def test_tune_min_data_in_leaf(self) -> None:
        unexpected_value = 1  # out of scope.

        with turnoff_cv():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=DeprecationWarning)
                runner = self._get_tunercv_object(
                    params=dict(
                        min_child_samples=unexpected_value,  # set default as unexpected value.
                    ),
                )
            assert len(runner.study.trials) == 0
            runner.tune_min_data_in_leaf()

            assert runner.lgbm_params["min_child_samples"] != unexpected_value
            assert len(runner.study.trials) == 5

    def test_resume_run(self) -> None:
        params = {"verbose": -1}  # type: Dict
        dataset = lgb.Dataset(np.zeros((10, 10)))

        study = optuna.create_study()
        tuner = LightGBMTunerCV(params, dataset, study=study)

        with mock.patch.object(OptunaObjectiveCV, "_get_cv_scores", return_value=[1.0]):
            tuner.tune_regularization_factors()

        n_trials = len(study.trials)
        assert n_trials == len(study.trials)

        tuner2 = LightGBMTuner(params, dataset, valid_sets=dataset, study=study)
        with mock.patch.object(OptunaObjectiveCV, "_get_cv_scores", return_value=[1.0]):
            tuner2.tune_regularization_factors()
        assert n_trials == len(study.trials)

    def test_optuna_callback(self) -> None:
        params = {"verbose": -1}  # type: Dict[str, Any]
        dataset = lgb.Dataset(np.zeros((10, 10)))

        callback_mock = mock.MagicMock()

        study = optuna.create_study()
        tuner = LightGBMTunerCV(params, dataset, study=study, optuna_callbacks=[callback_mock],)

        with mock.patch.object(OptunaObjectiveCV, "_get_cv_scores", return_value=[1.0]):
            tuner.tune_params(["num_leaves"], 10, optuna.samplers.TPESampler(), "num_leaves")

        assert callback_mock.call_count == 10
