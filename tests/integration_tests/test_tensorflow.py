import numpy as np
import pytest
import tensorflow as tf

import optuna
from optuna.integration import TensorFlowPruningHook
from optuna.testing.integration import DeterministicPruner
from optuna import types

if types.TYPE_CHECKING:
    import typing  # NOQA


def fixed_value_input_fn():
    # type: () -> typing.Tuple[typing.Dict[str, tf.Tensor], tf.Tensor]

    x_train = np.zeros([16, 20])
    y_train = np.zeros(16)
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset = dataset.repeat().batch(8)
    iterator = dataset.make_one_shot_iterator()
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
            trial=trial,
            estimator=clf,
            metric="accuracy",
            run_every_steps=5,
        )
        train_spec = tf.estimator.TrainSpec(
            input_fn=fixed_value_input_fn, max_steps=100, hooks=[hook])
        eval_spec = tf.estimator.EvalSpec(input_fn=fixed_value_input_fn, steps=1, hooks=[])
        tf.estimator.train_and_evaluate(estimator=clf, train_spec=train_spec, eval_spec=eval_spec)
        return 1.0

    study = optuna.create_study(pruner=DeterministicPruner(True), direction='maximize')
    study.optimize(objective, n_trials=1)
    assert study.trials[0].state == optuna.structs.TrialState.PRUNED

    study = optuna.create_study(pruner=DeterministicPruner(False), direction='maximize')
    study.optimize(objective, n_trials=1)
    assert study.trials[0].state == optuna.structs.TrialState.COMPLETE
    assert study.trials[0].value == 1.0


@pytest.mark.parametrize('is_higher_better', [True, False])
def test_init_with_is_higher_better(is_higher_better):
    # type: (bool) -> None

    clf = tf.estimator.DNNClassifier(
        hidden_units=[],
        feature_columns=[tf.feature_column.numeric_column(key="x", shape=[20])],
        model_dir=None,
        n_classes=2,
        config=tf.estimator.RunConfig(save_summary_steps=10, save_checkpoints_steps=10),
    )

    study = optuna.create_study()
    trial_id = study.storage.create_new_trial_id(study.study_id)

    with pytest.raises(ValueError):
        TensorFlowPruningHook(
            trial=optuna.trial.Trial(study, trial_id),
            estimator=clf,
            metric="accuracy",
            run_every_steps=5,
            is_higher_better=is_higher_better)
