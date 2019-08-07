"""
reminder:

* add `_`-prefix for all private functions.
"""
import sys
import contextlib

import pytest
import mock
import numpy as np

import optuna
import optuna.integration.lightgbm as lgb
from optuna.integration.lightgbm_autotune.optimize import (
    _TimeKeeper,
    _timer,
    BaseTuner,
    OptunaObjective,
)


if sys.version_info.major == 3:
    from contextlib import ExitStack
else:
    from contextlib import nested


@contextlib.contextmanager
def turnoff_autotune():
    fqn_prefix = 'optuna.integration.lightgbm_autotune.LGBMAutoTune'
    mock_pairs = [
        (fqn_prefix + '.__init__', None),
        (fqn_prefix + '.run', True),
        (fqn_prefix + '._parse_args', None),
        (fqn_prefix + '.get_params', {}),
    ]

    if sys.version_info.major == 3:
        with ExitStack() as stack:
            for fqn, return_value in mock_pairs:
                stack.enter_context(mock.patch(fqn, return_value=return_value))
            yield

    else:
        mocks = [
            mock.patch(fqn, return_value=return_value)
            for fqn, return_value in mock_pairs
        ]
        with nested(*mocks):
            yield


@contextlib.contextmanager
def turnoff_train():
    unexpected_value = 0.5
    dummy_num_iterations = 1234

    class DummyBooster:
        def __init__(self):
            self.best_score = {
                'valid_0': {'binary_logloss': unexpected_value},
            }

        def current_iteration(self):
            return dummy_num_iterations

    dummy_booster = DummyBooster()

    with mock.patch('lightgbm.train', return_value=dummy_booster):
        yield


class TestOptunaObjective:

    def test_init_(self):
        target_param_names = ['learning_rate']  # Invalid parameter name

        with pytest.raises(NotImplementedError) as execinfo:
            OptunaObjective(target_param_names, {}, None, {}, 0)

        assert execinfo.type is NotImplementedError

    @turnoff_train()
    def test_call(self):
        target_param_names = ['lambda_l1']
        lgbm_params = {}
        train_set = lgb.Dataset(None)
        val_set = lgb.Dataset(None)

        lgbm_kwargs = {'valid_sets': val_set}
        best_score = -np.inf

        objective = OptunaObjective(
            target_param_names,
            lgbm_params,
            train_set,
            lgbm_kwargs,
            best_score,
        )
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=10)

        assert study.best_value == 0.5


class TestLGBMModel:

    def _generate_dataset(self):
        X_trn = np.random.uniform(10, size=20).reshape((4, 5))
        y_trn = np.random.randint(2, size=4)
        X_val = np.random.uniform(10, size=20).reshape((4, 5))
        y_val = np.random.randint(2, size=4)
        return X_trn, y_trn, X_val, y_val

    def test_fit(self):
        X_trn, y_trn, X_val, y_val = self._generate_dataset()

        with turnoff_autotune():
            clf = lgb.LGBMModel(objective='binary', n_estimators=5)
            clf.fit(X_trn, y_trn, eval_set=(X_val, y_val), early_stopping_rounds=2)

        y_pred = clf.predict(X_val)
        assert isinstance(y_pred, np.ndarray)

    def test_tune__required_parameters(self):
        X_trn, y_trn, X_val, y_val = self._generate_dataset()

        with turnoff_autotune():
            # Case1
            clf = lgb.LGBMModel(objective='binary', n_estimators=5)
            with pytest.raises(ValueError) as excinfo:
                clf.tune(X_trn, y_trn, early_stopping_rounds=2)
            assert excinfo.type is ValueError

            # Case2
            clf = lgb.LGBMModel(objective='binary', n_estimators=5)
            with pytest.raises(ValueError) as excinfo:
                clf.tune(X_trn, y_trn, eval_set=(X_val, y_val))
            assert excinfo.type is ValueError

            # Case3
            clf = lgb.LGBMModel(objective='binary', n_estimators=5)
            with pytest.raises(ValueError) as excinfo:
                clf.tune(X_trn, y_trn, eval_set=(X_val, y_val))
            assert excinfo.type is ValueError

    def test_tune(self):
        X_trn, y_trn, X_val, y_val = self._generate_dataset()

        with turnoff_autotune():
            clf = lgb.LGBMModel(objective='binary', n_estimators=5)
            clf.tune(X_trn, y_trn, eval_set=[(X_val, y_val)], early_stopping_rounds=2)
        assert clf.is_tuned

    def test_tune__warning_before_tuned(self):
        X_trn, y_trn, X_val, y_val = self._generate_dataset()

        with turnoff_autotune():
            clf = lgb.LGBMModel(objective='binary', n_estimators=5)

            with pytest.warns(UserWarning) as record:
                clf.fit(X_trn, y_trn,
                        eval_set=(X_val, y_val),
                        early_stopping_rounds=2)
            assert len(record) == 1
            assert record[0].message.args[0] == (
                'Not tuned hyperparameter yet. Call `tune()` before `fit()`.')

            y_pred = clf.predict(X_val)
            assert isinstance(y_pred, np.ndarray)

            assert 'is_tuned' not in clf.__dict__
            clf.tune(X_trn, y_trn,
                     eval_set=[(X_val, y_val)],
                     early_stopping_rounds=2)
            assert clf.is_tuned

            clf = lgb.LGBMModel(objective='binary', n_estimators=5)
            assert 'is_tuned' not in clf.__dict__
            clf.tune(X_trn, y_trn,
                     eval_set=[(X_val, y_val)],
                     early_stopping_rounds=2)
            assert clf.is_tuned
            with pytest.warns(None) as record:
                clf.fit(X_trn, y_trn,
                        eval_set=(X_val, y_val),
                        early_stopping_rounds=2)
            assert len(record) == 0


class Test_TimeKeeper:
    def test__timer_elapsed_secs(self):
        with mock.patch('time.time', return_value=1):
            tk = _TimeKeeper()
            with mock.patch('time.time', return_value=10):
                assert tk.elapsed_secs() == 9


def test__timer_context():
    with mock.patch('time.time', return_value=1):
        with _timer() as t:
            with mock.patch('time.time', return_value=10):
                assert t.elapsed_secs() == 9


class TestBaseTuner:
    def test_get_booster_best_score(self):
        expected_value = 1.0

        class DummyBooster:
            def __init__(self):
                self.best_score = {
                    'valid_0': {'binary_logloss': expected_value}
                }

        booster = DummyBooster()
        dummy_dataset = lgb.Dataset(None)

        tuner = BaseTuner(lgbm_kwargs=dict(valid_sets=dummy_dataset))
        val_score = tuner.get_booster_best_score(booster)
        assert val_score == expected_value

    def test_higher_is_better(self):
        for metric in ['auc', 'accuracy']:
            tuner = BaseTuner(lgbm_params={'metric': metric})
            assert tuner.higher_is_better()

        for metric in ['rmsle', 'rmse', 'binary_logloss']:
            tuner = BaseTuner(lgbm_params={'metric': metric})
            assert not tuner.higher_is_better()

    def test_get_booster_best_score__using_valid_names_as_str(self):
        expected_value = 1.0

        class DummyBooster:
            def __init__(self):
                self.best_score = {
                    'dev': {'binary_logloss': expected_value}
                }

        booster = DummyBooster()
        dummy_dataset = lgb.Dataset(None)

        tuner = BaseTuner(lgbm_kwargs={
            'valid_names': 'dev',
            'valid_sets': dummy_dataset,
        })
        val_score = tuner.get_booster_best_score(booster)
        assert val_score == expected_value

    def test_get_booster_best_score__using_valid_names_as_list(self):
        unexpected_value = 0.5
        expected_value = 1.0

        class DummyBooster:
            def __init__(self):
                self.best_score = {
                    'train': {'binary_logloss': unexpected_value},
                    'val': {'binary_logloss': expected_value}
                }

        booster = DummyBooster()
        dummy_train_dataset = lgb.Dataset(None)
        dummy_val_dataset = lgb.Dataset(None)

        tuner = BaseTuner(lgbm_kwargs={
            'valid_names': ['train', 'val'],
            'valid_sets': [dummy_train_dataset, dummy_val_dataset],
        })
        val_score = tuner.get_booster_best_score(booster)
        assert val_score == expected_value

    def test_compare_validation_metrics(self):
        for metric in ['auc', 'accuracy']:
            tuner = BaseTuner(lgbm_params={'metric': metric})
            assert tuner.compare_validation_metrics(0.5, 0.1)
            assert not tuner.compare_validation_metrics(0.5, 0.5)
            assert not tuner.compare_validation_metrics(0.1, 0.5)

        for metric in ['rmsle', 'rmse', 'binary_logloss']:
            tuner = BaseTuner(lgbm_params={'metric': metric})
            assert not tuner.compare_validation_metrics(0.5, 0.1)
            assert not tuner.compare_validation_metrics(0.5, 0.5)
            assert tuner.compare_validation_metrics(0.1, 0.5)


class TestLGBMAutoTune:

    def _helper_get_minimum_runner(self, params={}, train_set=None, kwargs_options={}):
        # Required keyword arguments
        dummy_dataset = lgb.Dataset(None)

        kwargs = dict(
            num_boost_round=5,
            early_stopping_rounds=2,
            valid_sets=dummy_dataset,
        )
        kwargs.update(kwargs_options)

        runner = lgb.LGBMAutoTune(params, train_set, **kwargs)
        return runner

    def test_no_eval_set_args(self):
        params = {}
        train_set = lgb.Dataset(None)
        with pytest.raises(ValueError) as excinfo:
            lgb.LGBMAutoTune(params,
                             train_set,
                             num_boost_round=5,
                             early_stopping_rounds=2)

        assert excinfo.type == ValueError
        assert str(excinfo.value) == '`valid_sets` is required.'

    def test_with_minimum_required_args(self):
        runner = self._helper_get_minimum_runner()
        assert 'num_boost_round' in runner.lgbm_kwargs
        assert 'num_boost_round' not in runner.auto_options
        assert runner.lgbm_kwargs['num_boost_round'] == 5

    def test__parse_args_wrapper_args(self):
        params = {}
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
        runner = lgb.LGBMAutoTune(params, train_set, **kwargs)
        new_args = ['time_budget', 'time_budget', 'best_params', 'sample_size']
        for new_arg in new_args:
            assert new_arg not in runner.lgbm_kwargs
            assert new_arg in runner.auto_options

    def test_sampling_train_set(self):
        sample_size = 3

        X_trn = np.random.uniform(10, size=50).reshape((10, 5))
        y_trn = np.random.randint(2, size=10)
        train_dataset = lgb.Dataset(X_trn, label=y_trn)
        runner = self._helper_get_minimum_runner(train_set=train_dataset,
                                                 kwargs_options=dict(sample_size=sample_size))
        runner.sampling_train_set()
        runner.train_subset.construct()  # Cannt get label before construct Dataset
        assert runner.train_subset.get_label().shape[0] == sample_size

    def test_tune_feature_fraction(self):
        unexpected_value = 1.1  # out of scope

        with turnoff_train():
            runner = self._helper_get_minimum_runner(params=dict(
                feature_fraction=unexpected_value,  # set default as unexpected value
            ), kwargs_options=dict(
                tuning_history=[],
                best_params={},
            ))
            assert len(runner.tuning_history) == 0
            runner.tune_feature_fraction()

            assert runner.lgbm_params['feature_fraction'] != unexpected_value
            assert len(runner.tuning_history) > 0

    def test_tune_num_leaves(self):
        unexpected_value = 1  # out of scope

        with turnoff_train():
            runner = self._helper_get_minimum_runner(params=dict(
                num_leaves=unexpected_value,
            ), kwargs_options=dict(
                tuning_history=[],
                best_params={},
            ))
            assert len(runner.tuning_history) == 0
            runner.tune_num_leaves()

            assert runner.lgbm_params['num_leaves'] != unexpected_value
            assert len(runner.tuning_history) > 0

    def test_tune_bagging_fraction(self):
        unexpected_value = 1  # out of scope

        with turnoff_train():
            runner = self._helper_get_minimum_runner(params=dict(
                bagging_fraction=unexpected_value,
            ), kwargs_options=dict(
                tuning_history=[],
                best_params={},
            ))
            assert len(runner.tuning_history) == 0
            runner.tune_bagging_fraction()

            assert runner.lgbm_params['bagging_fraction'] != unexpected_value
            assert len(runner.tuning_history) > 0

    def test_tune_feature_fraction_stage2(self):
        unexpected_value = 1.1  # out of scope

        with turnoff_train():
            runner = self._helper_get_minimum_runner(params=dict(
                feature_fraction=unexpected_value,  # set default as unexpected value
            ), kwargs_options=dict(
                tuning_history=[],
                best_params={},
            ))
            assert len(runner.tuning_history) == 0
            runner.tune_feature_fraction()

            assert len(runner.tuning_history) > 0

    def test_tune_regularization_factors(self):
        unexpected_value = 20  # out of scope

        with turnoff_train():
            runner = self._helper_get_minimum_runner(params=dict(
                lambda_l1=unexpected_value,  # set default as unexpected value
            ), kwargs_options=dict(
                tuning_history=[],
                best_params={},
            ))
            assert len(runner.tuning_history) == 0
            runner.tune_regularization_factors()

            assert runner.lgbm_params['lambda_l1'] != unexpected_value
            assert len(runner.tuning_history) > 0

    def test_tune_min_data_in_leaf(self):
        unexpected_value = 1  # out of scope

        with turnoff_train():
            runner = self._helper_get_minimum_runner(params=dict(
                min_child_samples=unexpected_value,  # set default as unexpected value
            ), kwargs_options=dict(
                tuning_history=[],
                best_params={},
            ))
            assert len(runner.tuning_history) == 0
            runner.tune_min_data_in_leaf()

            assert runner.lgbm_params['min_child_samples'] != unexpected_value
            assert len(runner.tuning_history) > 0

    def test_tune_learning_rate(self):
        pass
