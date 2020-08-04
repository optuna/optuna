from collections import OrderedDict
import math
from unittest.mock import patch

import numpy as np
import tensorflow as tf

import optuna
from optuna.integration import TensorFlowPruningHook
from optuna.testing.integration import DeterministicPruner
from optuna import type_checking

if type_checking.TYPE_CHECKING:
    import typing  # NOQA


def fixed_value_input_fn():
    # type: () -> typing.Tuple[typing.Dict[str, tf.Tensor], tf.Tensor]

    x_train = np.zeros([16, 20])
    y_train = np.zeros(16)
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset = dataset.repeat().batch(8)
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    features, labels = iterator.get_next()
    return {"x": features}, labels


def test_tensorflow_pruning_hook():
    # type: () -> None

    def objective(trial):
        # type: (optuna.trial.Trial) -> float

        clf = tf.estimator.DNNClassifier(
            hidden_units=[],
            feature_columns=[tf.feature_column.numeric_column(key="x", shape=[20])],
            model_dir=None,
            n_classes=2,
            config=tf.estimator.RunConfig(save_summary_steps=10, save_checkpoints_steps=10),
        )
        hook = TensorFlowPruningHook(
            trial=trial, estimator=clf, metric="accuracy", run_every_steps=5,
        )
        train_spec = tf.estimator.TrainSpec(
            input_fn=fixed_value_input_fn, max_steps=100, hooks=[hook]
        )
        eval_spec = tf.estimator.EvalSpec(input_fn=fixed_value_input_fn, steps=1, hooks=[])
        tf.estimator.train_and_evaluate(estimator=clf, train_spec=train_spec, eval_spec=eval_spec)
        return 1.0

    study = optuna.create_study(pruner=DeterministicPruner(True), direction="maximize")
    study.optimize(objective, n_trials=1)
    assert study.trials[0].state == optuna.trial.TrialState.PRUNED

    study = optuna.create_study(pruner=DeterministicPruner(False), direction="maximize")
    study.optimize(objective, n_trials=1)
    assert study.trials[0].state == optuna.trial.TrialState.COMPLETE
    assert study.trials[0].value == 1.0

    # Check if eval_metrics returns the None value.
    value = OrderedDict([(10, {"accuracy": None})])
    with patch("optuna.integration.tensorflow.read_eval_metrics", return_value=value) as mock_obj:
        study = optuna.create_study(pruner=DeterministicPruner(True), direction="maximize")
        study.optimize(objective, n_trials=1)
        assert mock_obj.call_count == 1
        assert math.isnan(study.trials[0].intermediate_values[10])
        assert study.trials[0].state == optuna.trial.TrialState.PRUNED
