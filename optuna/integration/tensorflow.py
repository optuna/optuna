from __future__ import absolute_import

import optuna


try:
    import tensorflow as tf
    from tensorflow.train import SessionRunHook
    _available = True
except ImportError as e:
    _import_error = e
    # PruningHook is disabled because TensorFlow is not available.
    _available = False
    SessionRunHook = object


class TensorFlowPruningHook(SessionRunHook):
    """TensorFlow SessionRunHook to prune umpromising trials.

    Example:

        Add a pruning SessionRunHook which observes validation scores to training of a TensorFlow's Estimator.

        .. code::

                optuna_pruning_hook = OptunaPruningHook(
                    trial=trial,
                    estimator=clf,
                    metric="accuracy",
                    is_higher_better=True,
                    run_every_steps=10,
                )
                tf.estimator.train_and_evaluate(
                    clf,
                    tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=500, hooks=hooks),
                    eval_spec
                )
    Args:
        trial:
            A :class:`~optuna.trial.Trial` corresponding to the current evaluation of
            the objective function.
        estimator:
            A :estimator which you will use.
        metric:
            An evaluation metric for pruning, e.g., ``accuracy`` and ``loss``.
        is_higher_better:
           It should be True if you use a metric to be maximize such as ``loss``.
        run_every_steps:
           An interval to watch the summary file.
    """
    def __init__(self, trial, estimator, metric, is_higher_better, run_every_steps):
        self.trial = trial
        self.estimator = estimator
        self.current_summary_step = -1
        self.metric = metric
        self.is_higher_better = is_higher_better
        self._global_step_tensor = None
        self._timer = tf.train.SecondOrStepTimer(every_secs=None, every_steps=run_every_steps)

    def begin(self):
        self._global_step_tensor = tf.train.get_global_step()

    def before_run(self, run_context):
        del run_context
        return tf.train.SessionRunArgs(self._global_step_tensor)

    def after_run(self, run_context, run_value):
        global_step = run_value.results
        # Get eval metrics every n steps
        if self._timer.should_trigger_for_step(global_step):
            eval_metrics = tf.contrib.estimator.read_eval_metrics(self.estimator.eval_dir())
        else:
            eval_metrics = None
        if eval_metrics:
            summary_step = next(reversed(eval_metrics))
            latest_eval_metrics = eval_metrics[summary_step]
            # If there exists a new evaluation summary
            if summary_step > self.current_summary_step:
                if self.is_higher_better:
                    current_score = 1.0 - latest_eval_metrics[self.metric]
                else:
                    current_score = latest_eval_metrics[self.metric]
                self.trial.report(current_score, step=summary_step)
                self.current_summary_step = summary_step
            if self.trial.should_prune(self.current_summary_step):
                message = "Trial was pruned at iteration {}.".format(self.current_summary_step)
                raise optuna.structs.TrialPruned(message)


def _check_tensorflow_availability():
    # type: () -> None

    if not _available:
        raise ImportError(
            'TensorFlow is not available. Please install TensorFlow to use this feature. '
            'TensorFlow can be installed by executing `$ pip install tensorflow`. '
            'For further information, please refer to the installation guide of TensorFlow. '
            '(The actual import error is as follows: ' + str(_import_error) + ')')
