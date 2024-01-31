from __future__ import annotations

from collections.abc import Generator
import contextlib
from tempfile import TemporaryDirectory
from typing import Any
from typing import TYPE_CHECKING
from unittest import mock
import warnings

import numpy as np
import pytest

import optuna
from optuna._imports import try_import
from optuna.integration._lightgbm_tuner.optimize import _BaseTuner
from optuna.integration._lightgbm_tuner.optimize import _OptunaObjective
from optuna.integration._lightgbm_tuner.optimize import _OptunaObjectiveCV
from optuna.integration._lightgbm_tuner.optimize import LightGBMTuner
from optuna.integration._lightgbm_tuner.optimize import LightGBMTunerCV
import optuna.integration.lightgbm as lgb
from optuna.study import Study


with try_import():
    from lightgbm import early_stopping
    from lightgbm import log_evaluation
    import sklearn.datasets
    from sklearn.model_selection import KFold
    from sklearn.model_selection import train_test_split

pytestmark = pytest.mark.integration


@contextlib.contextmanager
def turnoff_train(metric: str = "binary_logloss") -> Generator[None, None, None]:
    unexpected_value = 0.5
    dummy_num_iterations = 1234

    class DummyBooster:
        def __init__(self) -> None:
            self.best_score = {
                "valid_0": {metric: unexpected_value},
            }

        def current_iteration(self) -> int:
            return dummy_num_iterations

    dummy_booster = DummyBooster()

    with mock.patch("lightgbm.train", return_value=dummy_booster):
        yield


@contextlib.contextmanager
def turnoff_cv(metric: str = "binary_logloss") -> Generator[None, None, None]:
    unexpected_value = 0.5
    dummy_results = {"valid {}-mean".format(metric): [unexpected_value]}

    with mock.patch("lightgbm.cv", return_value=dummy_results):
        yield


class TestOptunaObjective:
    def test_init_(self) -> None:
        target_param_names = ["learning_rate"]  # Invalid parameter name.

        with pytest.raises(NotImplementedError):
            dataset = mock.MagicMock(spec="lgb.Dataset")
            _OptunaObjective(target_param_names, {}, dataset, {}, 0, "tune_learning_rate", None)

    def test_call(self) -> None:
        target_param_names = ["lambda_l1"]
        lgbm_params: dict[str, Any] = {}
        train_set = lgb.Dataset(None)
        val_set = lgb.Dataset(None)

        lgbm_kwargs = {"valid_sets": val_set}
        best_score = -np.inf

        with turnoff_train():
            objective = _OptunaObjective(
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


class TestOptunaObjectiveCV:
    def test_call(self) -> None:
        target_param_names = ["lambda_l1"]
        lgbm_params: dict[str, Any] = {}
        train_set = lgb.Dataset(None)
        lgbm_kwargs: dict[str, Any] = {}
        best_score = -np.inf

        with turnoff_cv():
            objective = _OptunaObjectiveCV(
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


class TestBaseTuner:
    def test_get_booster_best_score(self) -> None:
        expected_value = 1.0

        booster = mock.MagicMock(
            spec="lgb.Booster", best_score={"valid_0": {"binary_logloss": expected_value}}
        )
        dummy_dataset = lgb.Dataset(None)

        tuner = _BaseTuner(lgbm_kwargs=dict(valid_sets=dummy_dataset))
        val_score = tuner._get_booster_best_score(booster)
        assert val_score == expected_value

    def test_higher_is_better(self) -> None:
        for metric in [
            "auc",
            "auc_mu",
            "ndcg",
            "lambdarank",
            "rank_xendcg",
            "xendcg",
            "xe_ndcg",
            "xe_ndcg_mart",
            "xendcg_mart",
            "map",
            "mean_average_precision",
            "average_precision",
        ]:
            tuner = _BaseTuner(lgbm_params={"metric": metric})
            assert tuner.higher_is_better()

        for metric in [
            "mae",
            "rmse",
            "quantile",
            "mape",
            "binary_logloss",
            "multi_logloss",
            "cross_entropy",
        ]:
            tuner = _BaseTuner(lgbm_params={"metric": metric})
            assert not tuner.higher_is_better()

    def test_get_booster_best_score_using_valid_names_as_str(self) -> None:
        expected_value = 1.0

        booster = mock.MagicMock(
            spec="lgb.Booster", best_score={"dev": {"binary_logloss": expected_value}}
        )
        dummy_dataset = lgb.Dataset(None)

        tuner = _BaseTuner(lgbm_kwargs={"valid_names": "dev", "valid_sets": dummy_dataset})
        val_score = tuner._get_booster_best_score(booster)
        assert val_score == expected_value

    def test_get_booster_best_score_using_valid_names_as_list(self) -> None:
        unexpected_value = 0.5
        expected_value = 1.0

        booster = mock.MagicMock(
            spec="lgb.Booster",
            best_score={
                "train": {"binary_logloss": unexpected_value},
                "val": {"binary_logloss": expected_value},
            },
        )
        dummy_train_dataset = lgb.Dataset(None)
        dummy_val_dataset = lgb.Dataset(None)

        tuner = _BaseTuner(
            lgbm_kwargs={
                "valid_names": ["train", "val"],
                "valid_sets": [dummy_train_dataset, dummy_val_dataset],
            }
        )
        val_score = tuner._get_booster_best_score(booster)
        assert val_score == expected_value

    def test_compare_validation_metrics(self) -> None:
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
            tuner = _BaseTuner(lgbm_params={"metric": metric})
            assert tuner.compare_validation_metrics(0.5, 0.1)
            assert not tuner.compare_validation_metrics(0.5, 0.5)
            assert not tuner.compare_validation_metrics(0.1, 0.5)

        for metric in ["rmsle", "rmse", "binary_logloss"]:
            tuner = _BaseTuner(lgbm_params={"metric": metric})
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
    def test_metric_with_eval_at(
        self, metric: str, eval_at_param: dict[str, int | list[int]], expected: str
    ) -> None:
        params: dict[str, str | int | list[int]] = {"metric": metric}
        params.update(eval_at_param)
        tuner = _BaseTuner(lgbm_params=params)
        assert tuner._metric_with_eval_at(metric) == expected

    def test_metric_with_eval_at_error(self) -> None:
        tuner = _BaseTuner(lgbm_params={"metric": "ndcg", "eval_at": "1"})
        with pytest.raises(ValueError):
            tuner._metric_with_eval_at("ndcg")


class TestLightGBMTuner:
    def _get_tuner_object(
        self,
        params: dict[str, Any] = {},
        train_set: "lgb.Dataset" | None = None,
        kwargs_options: dict[str, Any] = {},
        study: Study | None = None,
    ) -> lgb.LightGBMTuner:
        # Required keyword arguments.
        dummy_dataset = lgb.Dataset(None)
        train_set = train_set or mock.MagicMock(spec="lgb.Dataset")

        runner = lgb.LightGBMTuner(
            params,
            train_set,
            num_boost_round=5,
            valid_sets=dummy_dataset,
            callbacks=[early_stopping(stopping_rounds=2)],
            study=study,
            **kwargs_options,
        )
        return runner

    def test_deprecated_args(self) -> None:
        dummy_dataset = lgb.Dataset(None)

        with pytest.warns(FutureWarning):
            LightGBMTuner({}, dummy_dataset, valid_sets=[dummy_dataset], verbosity=1)

    def test_no_eval_set_args(self) -> None:
        params: dict[str, Any] = {}
        train_set = lgb.Dataset(None)
        with pytest.raises(ValueError) as excinfo:
            lgb.LightGBMTuner(
                params,
                train_set,
                num_boost_round=5,
                callbacks=[early_stopping(stopping_rounds=2)],
            )

        assert excinfo.type == ValueError
        assert str(excinfo.value) == "`valid_sets` is required."

    @pytest.mark.parametrize(
        "metric, study_direction",
        [
            ("auc", "minimize"),
            ("mse", "maximize"),
            (None, "maximize"),  # The default metric is binary_logloss.
        ],
    )
    def test_inconsistent_study_direction(self, metric: str, study_direction: str) -> None:
        params: dict[str, Any] = {}
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
                callbacks=[early_stopping(stopping_rounds=2)],
                study=study,
            )

        assert excinfo.type == ValueError
        assert str(excinfo.value).startswith("Study direction is inconsistent with the metric")

    def test_with_minimum_required_args(self) -> None:
        runner = self._get_tuner_object()
        assert "num_boost_round" in runner.lgbm_kwargs
        assert "num_boost_round" not in runner.auto_options
        assert runner.lgbm_kwargs["num_boost_round"] == 5

    def test_parse_args_wrapper_args(self) -> None:
        params: dict[str, Any] = {}
        train_set = lgb.Dataset(None)
        val_set = lgb.Dataset(None)
        runner = lgb.LightGBMTuner(
            params,
            train_set,
            num_boost_round=12,
            callbacks=[early_stopping(stopping_rounds=10)],
            valid_sets=val_set,
            time_budget=600,
            sample_size=1000,
        )
        new_args = ["time_budget", "time_budget", "sample_size"]
        for new_arg in new_args:
            assert new_arg not in runner.lgbm_kwargs
            assert new_arg in runner.auto_options

    @pytest.mark.parametrize(
        "metric, study_direction, expected",
        [("auc", "maximize", -np.inf), ("l2", "minimize", np.inf)],
    )
    def test_best_score(self, metric: str, study_direction: str, expected: float) -> None:
        with turnoff_train(metric=metric):
            study = optuna.create_study(direction=study_direction)
            runner = self._get_tuner_object(
                params=dict(lambda_l1=0.0, metric=metric), kwargs_options={}, study=study
            )
            assert runner.best_score == expected
            runner.tune_regularization_factors()
            assert runner.best_score == 0.5

    def test_best_params(self) -> None:
        unexpected_value = 20  # out of scope.

        with turnoff_train():
            study = optuna.create_study()
            runner = self._get_tuner_object(
                params=dict(lambda_l1=unexpected_value), kwargs_options={}, study=study
            )
            assert runner.best_params["lambda_l1"] == unexpected_value
            runner.tune_regularization_factors()
            assert runner.best_params["lambda_l1"] != unexpected_value

    def test_sample_train_set(self) -> None:
        sample_size = 3

        X_trn = np.random.uniform(10, size=50).reshape((10, 5))
        y_trn = np.random.randint(2, size=10)
        train_dataset = lgb.Dataset(X_trn, label=y_trn)
        runner = self._get_tuner_object(
            train_set=train_dataset, kwargs_options=dict(sample_size=sample_size)
        )
        runner.sample_train_set()

        # Workaround for mypy.
        if not TYPE_CHECKING:
            runner.train_subset.construct()  # Cannot get label before construct `lgb.Dataset`.
            assert runner.train_subset.get_label().shape[0] == sample_size

    def test_time_budget(self) -> None:
        unexpected_value = 1.1  # out of scope.

        with turnoff_train():
            runner = self._get_tuner_object(
                params=dict(
                    feature_fraction=unexpected_value,  # set default as unexpected value.
                ),
                kwargs_options=dict(time_budget=0),
            )
            assert len(runner.study.trials) == 0
            # No trials run because `time_budget` is set to zero.
            runner.tune_feature_fraction()
            assert runner.lgbm_params["feature_fraction"] == unexpected_value
            assert len(runner.study.trials) == 0

    def test_tune_feature_fraction(self) -> None:
        unexpected_value = 1.1  # out of scope.

        with turnoff_train():
            runner = self._get_tuner_object(
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

        with turnoff_train():
            runner = self._get_tuner_object(params=dict(num_leaves=unexpected_value))
            assert len(runner.study.trials) == 0
            runner.tune_num_leaves()

            assert runner.lgbm_params["num_leaves"] != unexpected_value
            assert len(runner.study.trials) == 20

    def test_tune_num_leaves_negative_max_depth(self) -> None:
        params: dict[str, Any] = {"metric": "binary_logloss", "max_depth": -1, "verbose": -1}
        X_trn = np.random.uniform(10, size=(10, 5))
        y_trn = np.random.randint(2, size=10)
        train_dataset = lgb.Dataset(X_trn, label=y_trn)
        valid_dataset = lgb.Dataset(X_trn, label=y_trn)

        runner = lgb.LightGBMTuner(
            params,
            train_dataset,
            num_boost_round=3,
            valid_sets=valid_dataset,
            callbacks=[early_stopping(stopping_rounds=2), log_evaluation(-1)],
        )
        runner.tune_num_leaves()
        assert len(runner.study.trials) == 20

    def test_tune_bagging(self) -> None:
        unexpected_value = 1  # out of scope.

        with turnoff_train():
            runner = self._get_tuner_object(params=dict(bagging_fraction=unexpected_value))
            assert len(runner.study.trials) == 0
            runner.tune_bagging()

            assert runner.lgbm_params["bagging_fraction"] != unexpected_value
            assert len(runner.study.trials) == 10

    def test_tune_feature_fraction_stage2(self) -> None:
        unexpected_value = 0.5

        with turnoff_train():
            runner = self._get_tuner_object(params=dict(feature_fraction=unexpected_value))
            assert len(runner.study.trials) == 0
            runner.tune_feature_fraction_stage2()

            assert runner.lgbm_params["feature_fraction"] != unexpected_value
            assert len(runner.study.trials) == 6

    def test_tune_regularization_factors(self) -> None:
        unexpected_value = 20  # out of scope.

        with turnoff_train():
            runner = self._get_tuner_object(
                params=dict(lambda_l1=unexpected_value)  # set default as unexpected value.
            )
            assert len(runner.study.trials) == 0
            runner.tune_regularization_factors()

            assert runner.lgbm_params["lambda_l1"] != unexpected_value
            assert len(runner.study.trials) == 20

    def test_tune_min_data_in_leaf(self) -> None:
        unexpected_value = 1  # out of scope.

        with turnoff_train():
            runner = self._get_tuner_object(
                params=dict(
                    min_child_samples=unexpected_value,  # set default as unexpected value.
                ),
            )
            assert len(runner.study.trials) == 0
            runner.tune_min_data_in_leaf()

            assert runner.lgbm_params["min_child_samples"] != unexpected_value
            assert len(runner.study.trials) == 5

    def test_when_a_step_does_not_improve_best_score(self) -> None:
        params: dict = {}
        valid_data = np.zeros((10, 10))
        valid_sets = lgb.Dataset(valid_data)

        dataset = mock.MagicMock(spec="lgb.Dataset")
        tuner = LightGBMTuner(params, dataset, valid_sets=valid_sets)
        assert not tuner.higher_is_better()

        with mock.patch("lightgbm.train"), mock.patch.object(
            _BaseTuner, "_get_booster_best_score", return_value=0.9
        ):
            tuner.tune_feature_fraction()

        assert "feature_fraction" in tuner.best_params
        assert tuner.best_score == 0.9

        # Assume that tuning `num_leaves` doesn't improve the `best_score`.
        with mock.patch("lightgbm.train"), mock.patch.object(
            _BaseTuner, "_get_booster_best_score", return_value=1.1
        ):
            tuner.tune_num_leaves()

    def test_resume_run(self) -> None:
        params: dict = {"verbose": -1}
        dataset = lgb.Dataset(np.zeros((10, 10)))

        study = optuna.create_study()
        tuner = LightGBMTuner(
            params, dataset, valid_sets=dataset, study=study, callbacks=[log_evaluation(-1)]
        )

        with mock.patch.object(_BaseTuner, "_get_booster_best_score", return_value=1.0):
            tuner.tune_regularization_factors()

        n_trials = len(study.trials)
        assert n_trials == len(study.trials)

        tuner2 = LightGBMTuner(params, dataset, valid_sets=dataset, study=study)
        with mock.patch.object(_BaseTuner, "_get_booster_best_score", return_value=1.0):
            tuner2.tune_regularization_factors()
        assert n_trials == len(study.trials)

    @pytest.mark.parametrize(
        "verbosity, level",
        [
            (None, optuna.logging.INFO),
            (-2, optuna.logging.CRITICAL),
            (-1, optuna.logging.CRITICAL),
            (0, optuna.logging.WARNING),
            (1, optuna.logging.INFO),
            (2, optuna.logging.DEBUG),
        ],
    )
    def test_run_verbosity(self, verbosity: int, level: int) -> None:
        # We need to reconstruct our default handler to properly capture stderr.
        optuna.logging._reset_library_root_logger()
        optuna.logging.set_verbosity(optuna.logging.INFO)

        params: dict = {"verbose": -1}
        dataset = lgb.Dataset(np.zeros((10, 10)))

        study = optuna.create_study()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            tuner = LightGBMTuner(
                params,
                dataset,
                valid_sets=dataset,
                study=study,
                verbosity=verbosity,
                callbacks=[log_evaluation(-1)],
                time_budget=1,
            )

        with mock.patch.object(_BaseTuner, "_get_booster_best_score", return_value=1.0):
            tuner.run()

        assert optuna.logging.get_verbosity() == level
        assert tuner.lgbm_params["verbose"] == -1

    @pytest.mark.parametrize("show_progress_bar, expected", [(True, 6), (False, 0)])
    def test_run_show_progress_bar(self, show_progress_bar: bool, expected: int) -> None:
        params: dict = {"verbose": -1}
        dataset = lgb.Dataset(np.zeros((10, 10)))

        study = optuna.create_study()
        tuner = LightGBMTuner(
            params,
            dataset,
            valid_sets=dataset,
            study=study,
            callbacks=[log_evaluation(-1)],
            time_budget=1,
            show_progress_bar=show_progress_bar,
        )

        with mock.patch.object(
            _BaseTuner, "_get_booster_best_score", return_value=1.0
        ), mock.patch("tqdm.tqdm") as mock_tqdm:
            tuner.run()

        assert mock_tqdm.call_count == expected

    def test_get_best_booster(self) -> None:
        unexpected_value = 20  # out of scope.

        params: dict = {"verbose": -1, "lambda_l1": unexpected_value}
        dataset = lgb.Dataset(np.zeros((10, 10)))

        study = optuna.create_study()
        tuner = LightGBMTuner(
            params, dataset, valid_sets=dataset, study=study, callbacks=[log_evaluation(-1)]
        )

        with pytest.raises(ValueError):
            tuner.get_best_booster()

        with mock.patch.object(_BaseTuner, "_get_booster_best_score", return_value=0.0):
            tuner.tune_regularization_factors()

        best_booster = tuner.get_best_booster()
        assert isinstance(best_booster.params, dict)
        assert best_booster.params["lambda_l1"] != unexpected_value

        tuner2 = LightGBMTuner(params, dataset, valid_sets=dataset, study=study)

        # Resumed study does not have the best booster.
        with pytest.raises(ValueError):
            tuner2.get_best_booster()

    @pytest.mark.parametrize("dir_exists, expected", [(False, True), (True, False)])
    def test_model_dir(self, dir_exists: bool, expected: bool) -> None:
        params: dict = {"verbose": -1}
        dataset = lgb.Dataset(np.zeros((10, 10)))

        with mock.patch("optuna.integration._lightgbm_tuner.optimize.os.mkdir") as m:
            with mock.patch("os.path.exists", return_value=dir_exists):
                LightGBMTuner(params, dataset, valid_sets=dataset, model_dir="./booster")
                assert m.called == expected

    def test_best_booster_with_model_dir(self) -> None:
        params: dict = {"verbose": -1}
        dataset = lgb.Dataset(np.zeros((10, 10)))

        study = optuna.create_study()
        with TemporaryDirectory() as tmpdir:
            tuner = LightGBMTuner(
                params,
                dataset,
                valid_sets=dataset,
                study=study,
                model_dir=tmpdir,
                callbacks=[log_evaluation(-1)],
            )

            with mock.patch.object(_BaseTuner, "_get_booster_best_score", return_value=0.0):
                tuner.tune_regularization_factors()

            best_booster = tuner.get_best_booster()

            tuner2 = LightGBMTuner(
                params, dataset, valid_sets=dataset, study=study, model_dir=tmpdir
            )
            best_booster2 = tuner2.get_best_booster()

            assert best_booster.params == best_booster2.params

    @pytest.mark.parametrize("direction, overall_best", [("minimize", 1), ("maximize", 2)])
    def test_create_stepwise_study(self, direction: str, overall_best: int) -> None:
        dataset = mock.MagicMock(spec="lgb.Dataset")
        tuner = LightGBMTuner({}, dataset, valid_sets=lgb.Dataset(np.zeros((10, 10))))

        def objective(trial: optuna.trial.Trial, value: float) -> float:
            trial.storage.set_trial_system_attr(
                trial._trial_id,
                optuna.integration._lightgbm_tuner.optimize._STEP_NAME_KEY,
                "step{:.0f}".format(value),
            )
            return trial.suggest_float("x", value, value)

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
        params: dict[str, Any] = {"verbose": -1}
        dataset = lgb.Dataset(np.zeros((10, 10)))

        callback_mock = mock.MagicMock()

        study = optuna.create_study()
        tuner = LightGBMTuner(
            params,
            dataset,
            valid_sets=dataset,
            study=study,
            callbacks=[log_evaluation(-1)],
            optuna_callbacks=[callback_mock],
        )

        with mock.patch.object(_BaseTuner, "_get_booster_best_score", return_value=1.0):
            tuner._tune_params(["num_leaves"], 10, optuna.samplers.TPESampler(), "num_leaves")

        assert callback_mock.call_count == 10

    @pytest.mark.skip(reason="Fail since 28 Jan 2024. TODO(nabenabe0928): Fix here.")
    def test_tune_best_score_reproducibility(self) -> None:
        iris = sklearn.datasets.load_iris()
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            iris.data, iris.target, random_state=0
        )

        train = lgb.Dataset(X_trainval, y_trainval)
        valid = lgb.Dataset(X_test, y_test)
        params = {
            "objective": "regression",
            "metric": "rmse",
            "random_seed": 0,
            "deterministic": True,
            "force_col_wise": True,
            "verbosity": -1,
        }

        tuner_first_try = lgb.LightGBMTuner(
            params,
            train,
            valid_sets=valid,
            callbacks=[early_stopping(stopping_rounds=3), log_evaluation(-1)],
            optuna_seed=10,
        )
        tuner_first_try.run()
        best_score_first_try = tuner_first_try.best_score

        tuner_second_try = lgb.LightGBMTuner(
            params,
            train,
            valid_sets=valid,
            callbacks=[early_stopping(stopping_rounds=3), log_evaluation(-1)],
            optuna_seed=10,
        )
        tuner_second_try.run()
        best_score_second_try = tuner_second_try.best_score

        assert best_score_second_try == best_score_first_try

        first_try_trials = tuner_first_try.study.trials
        second_try_trials = tuner_second_try.study.trials
        assert len(first_try_trials) == len(second_try_trials)
        for first_trial, second_trial in zip(first_try_trials, second_try_trials):
            assert first_trial.value == second_trial.value
            assert first_trial.params == second_trial.params


class TestLightGBMTunerCV:
    def _get_tunercv_object(
        self,
        params: dict[str, Any] = {},
        train_set: lgb.Dataset | None = None,
        kwargs_options: dict[str, Any] = {},
        study: optuna.study.Study | None = None,
    ) -> LightGBMTunerCV:
        # Required keyword arguments.
        kwargs: dict[str, Any] = dict(num_boost_round=5, study=study)
        kwargs.update(kwargs_options)

        train_set = train_set or mock.MagicMock(spec="lgb.Dataset")
        runner = LightGBMTunerCV(
            params, train_set, callbacks=[early_stopping(stopping_rounds=2)], **kwargs
        )
        return runner

    def test_deprecated_args(self) -> None:
        dummy_dataset = lgb.Dataset(None)

        with pytest.warns(FutureWarning):
            LightGBMTunerCV({}, dummy_dataset, verbosity=1)

    @pytest.mark.parametrize(
        "metric, study_direction",
        [
            ("auc", "minimize"),
            ("mse", "maximize"),
            (None, "maximize"),  # The default metric is binary_logloss.
        ],
    )
    def test_inconsistent_study_direction(self, metric: str, study_direction: str) -> None:
        params: dict[str, Any] = {}
        if metric is not None:
            params["metric"] = metric
        train_set = lgb.Dataset(None)
        study = optuna.create_study(direction=study_direction)
        with pytest.raises(ValueError) as excinfo:
            LightGBMTunerCV(
                params,
                train_set,
                num_boost_round=5,
                callbacks=[early_stopping(stopping_rounds=2)],
                study=study,
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
            runner = self._get_tunercv_object(params=dict(num_leaves=unexpected_value))
            assert len(runner.study.trials) == 0
            runner.tune_num_leaves()

            assert runner.lgbm_params["num_leaves"] != unexpected_value
            assert len(runner.study.trials) == 20

    def test_tune_bagging(self) -> None:
        unexpected_value = 1  # out of scope.

        with turnoff_cv():
            runner = self._get_tunercv_object(params=dict(bagging_fraction=unexpected_value))
            assert len(runner.study.trials) == 0
            runner.tune_bagging()

            assert runner.lgbm_params["bagging_fraction"] != unexpected_value
            assert len(runner.study.trials) == 10

    def test_tune_feature_fraction_stage2(self) -> None:
        unexpected_value = 0.5

        with turnoff_cv():
            runner = self._get_tunercv_object(params=dict(feature_fraction=unexpected_value))
            assert len(runner.study.trials) == 0
            runner.tune_feature_fraction_stage2()

            assert runner.lgbm_params["feature_fraction"] != unexpected_value
            assert len(runner.study.trials) == 6

    def test_tune_regularization_factors(self) -> None:
        unexpected_value = 20  # out of scope.

        with turnoff_cv():
            runner = self._get_tunercv_object(
                params=dict(lambda_l1=unexpected_value)  # set default as unexpected value.
            )
            assert len(runner.study.trials) == 0
            runner.tune_regularization_factors()

            assert runner.lgbm_params["lambda_l1"] != unexpected_value
            assert len(runner.study.trials) == 20

    def test_tune_min_data_in_leaf(self) -> None:
        unexpected_value = 1  # out of scope.

        with turnoff_cv():
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
        params: dict = {"verbose": -1}
        dataset = lgb.Dataset(np.zeros((10, 10)))

        study = optuna.create_study()
        tuner = LightGBMTunerCV(params, dataset, study=study)

        with mock.patch.object(_OptunaObjectiveCV, "_get_cv_scores", return_value=[1.0]):
            tuner.tune_regularization_factors()

        n_trials = len(study.trials)
        assert n_trials == len(study.trials)

        tuner2 = LightGBMTuner(params, dataset, valid_sets=dataset, study=study)
        with mock.patch.object(_OptunaObjectiveCV, "_get_cv_scores", return_value=[1.0]):
            tuner2.tune_regularization_factors()
        assert n_trials == len(study.trials)

    @pytest.mark.parametrize(
        "verbosity, level",
        [
            (None, optuna.logging.INFO),
            (-2, optuna.logging.CRITICAL),
            (-1, optuna.logging.CRITICAL),
            (0, optuna.logging.WARNING),
            (1, optuna.logging.INFO),
            (2, optuna.logging.DEBUG),
        ],
    )
    def test_run_verbosity(self, verbosity: int, level: int) -> None:
        # We need to reconstruct our default handler to properly capture stderr.
        optuna.logging._reset_library_root_logger()
        optuna.logging.set_verbosity(optuna.logging.INFO)

        params: dict = {"verbose": -1}
        dataset = lgb.Dataset(np.zeros((10, 10)))

        study = optuna.create_study()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            tuner = LightGBMTunerCV(
                params, dataset, study=study, verbosity=verbosity, time_budget=1
            )

        with mock.patch.object(_OptunaObjectiveCV, "_get_cv_scores", return_value=[1.0]):
            tuner.run()

        assert optuna.logging.get_verbosity() == level
        assert tuner.lgbm_params["verbose"] == -1

    @pytest.mark.parametrize("show_progress_bar, expected", [(True, 6), (False, 0)])
    def test_run_show_progress_bar(self, show_progress_bar: bool, expected: int) -> None:
        params: dict = {"verbose": -1}
        dataset = lgb.Dataset(np.zeros((10, 10)))

        study = optuna.create_study()
        tuner = LightGBMTunerCV(
            params, dataset, study=study, time_budget=1, show_progress_bar=show_progress_bar
        )

        with mock.patch.object(
            _OptunaObjectiveCV, "_get_cv_scores", return_value=[1.0]
        ), mock.patch("tqdm.tqdm") as mock_tqdm:
            tuner.run()

        assert mock_tqdm.call_count == expected

    def test_optuna_callback(self) -> None:
        params: dict[str, Any] = {"verbose": -1}
        dataset = lgb.Dataset(np.zeros((10, 10)))

        callback_mock = mock.MagicMock()

        study = optuna.create_study()
        tuner = LightGBMTunerCV(params, dataset, study=study, optuna_callbacks=[callback_mock])

        with mock.patch.object(_OptunaObjectiveCV, "_get_cv_scores", return_value=[1.0]):
            tuner._tune_params(["num_leaves"], 10, optuna.samplers.TPESampler(), "num_leaves")

        assert callback_mock.call_count == 10

    @pytest.mark.parametrize("dir_exists, expected", [(False, True), (True, False)])
    def test_model_dir(self, dir_exists: bool, expected: bool) -> None:
        unexpected_value = 20  # out of scope.

        params: dict = {"verbose": -1, "lambda_l1": unexpected_value}
        dataset = lgb.Dataset(np.zeros((10, 10)))

        with mock.patch("os.mkdir") as m:
            with mock.patch("os.path.exists", return_value=dir_exists):
                LightGBMTunerCV(params, dataset, model_dir="./booster")
                assert m.called == expected

    def test_get_best_booster(self) -> None:
        unexpected_value = 20  # out of scope.

        params: dict = {"verbose": -1, "lambda_l1": unexpected_value}
        dataset = lgb.Dataset(np.zeros((10, 10)))
        study = optuna.create_study()

        with TemporaryDirectory() as tmpdir:
            tuner = LightGBMTunerCV(
                params, dataset, study=study, model_dir=tmpdir, return_cvbooster=True
            )

            with pytest.raises(ValueError):
                tuner.get_best_booster()

            with mock.patch.object(_OptunaObjectiveCV, "_get_cv_scores", return_value=[1.0]):
                tuner.tune_regularization_factors()

            best_boosters = tuner.get_best_booster().boosters
            for booster in best_boosters:
                assert booster.params["lambda_l1"] != unexpected_value

            tuner2 = LightGBMTunerCV(
                params, dataset, study=study, model_dir=tmpdir, return_cvbooster=True
            )
            best_boosters2 = tuner2.get_best_booster().boosters
            for booster, booster2 in zip(best_boosters, best_boosters2):
                assert booster.params == booster2.params

    def test_get_best_booster_with_error(self) -> None:
        params: dict = {"verbose": -1}
        dataset = lgb.Dataset(np.zeros((10, 10)))
        study = optuna.create_study()

        tuner = LightGBMTunerCV(
            params, dataset, study=study, model_dir=None, return_cvbooster=True
        )
        # No trial is completed yet.
        with pytest.raises(ValueError):
            tuner.get_best_booster()

        with mock.patch.object(_OptunaObjectiveCV, "_get_cv_scores", return_value=[1.0]):
            tuner.tune_regularization_factors()

        tuner2 = LightGBMTunerCV(
            params, dataset, study=study, model_dir=None, return_cvbooster=True
        )
        # Resumed the study does not have the best booster.
        with pytest.raises(ValueError):
            tuner2.get_best_booster()

        with TemporaryDirectory() as tmpdir:
            tuner3 = LightGBMTunerCV(
                params, dataset, study=study, model_dir=tmpdir, return_cvbooster=True
            )
            # The booster was not saved hence not found in the `model_dir`.
            with pytest.raises(ValueError):
                tuner3.get_best_booster()

    @pytest.mark.skip(reason="Fail since 28 Jan 2024. TODO(nabenabe0928): Fix here.")
    def test_tune_best_score_reproducibility(self) -> None:
        iris = sklearn.datasets.load_iris()
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            iris.data, iris.target, random_state=0
        )

        train = lgb.Dataset(X_trainval, y_trainval)
        params = {
            "objective": "regression",
            "metric": "rmse",
            "random_seed": 0,
            "deterministic": True,
            "force_col_wise": True,
            "verbosity": -1,
        }

        tuner_first_try = lgb.LightGBMTunerCV(
            params,
            train,
            callbacks=[early_stopping(stopping_rounds=3)],
            folds=KFold(n_splits=3),
            optuna_seed=10,
        )
        tuner_first_try.run()
        best_score_first_try = tuner_first_try.best_score

        tuner_second_try = lgb.LightGBMTunerCV(
            params,
            train,
            callbacks=[early_stopping(stopping_rounds=3)],
            folds=KFold(n_splits=3),
            optuna_seed=10,
        )
        tuner_second_try.run()
        best_score_second_try = tuner_second_try.best_score

        assert best_score_second_try == best_score_first_try

        first_try_trials = tuner_first_try.study.trials
        second_try_trials = tuner_second_try.study.trials
        assert len(first_try_trials) == len(second_try_trials)
        for first_trial, second_trial in zip(first_try_trials, second_try_trials):
            assert first_trial.value == second_trial.value
            assert first_trial.params == second_trial.params
