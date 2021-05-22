import optuna
from optuna._imports import try_import


with try_import() as _imports:
    import tensorflow as tf
    from tensorflow.estimator import SessionRunHook
    from tensorflow_estimator.python.estimator.early_stopping import read_eval_metrics

if not _imports.is_successful():
    SessionRunHook = object  # NOQA


class TensorFlowPruningHook(SessionRunHook):
    """TensorFlow SessionRunHook to prune unpromising trials.

    See `the example <https://github.com/optuna/optuna-examples/tree/main/
    tensorflow/tensorflow_estimator_integration.py>`_
    if you want to add a pruning hook to TensorFlow's estimator.

    Args:
        trial:
            A :class:`~optuna.trial.Trial` corresponding to the current evaluation of
            the objective function.
        estimator:
            An estimator which you will use.
        metric:
            An evaluation metric for pruning, e.g., ``accuracy`` and ``loss``.
        run_every_steps:
           An interval to watch the summary file.
    """

    def __init__(
        self,
        trial: optuna.trial.Trial,
        estimator: "tf.estimator.Estimator",
        metric: str,
        run_every_steps: int,
    ) -> None:

        _imports.check()

        self._trial = trial
        self._estimator = estimator
        self._current_summary_step = -1
        self._metric = metric
        self._global_step_tensor = None
        self._timer = tf.estimator.SecondOrStepTimer(every_secs=None, every_steps=run_every_steps)

    def begin(self) -> None:

        self._global_step_tensor = tf.compat.v1.train.get_global_step()

    def before_run(
        self, run_context: "tf.estimator.SessionRunContext"
    ) -> "tf.estimator.SessionRunArgs":

        del run_context
        return tf.estimator.SessionRunArgs(self._global_step_tensor)

    def after_run(
        self,
        run_context: "tf.estimator.SessionRunContext",
        run_values: "tf.estimator.SessionRunValues",
    ) -> None:

        global_step = run_values.results
        # Get eval metrics every n steps.
        if self._timer.should_trigger_for_step(global_step):
            self._timer.update_last_triggered_step(global_step)
            eval_metrics = read_eval_metrics(self._estimator.eval_dir())
        else:
            eval_metrics = None
        if eval_metrics:
            summary_step = next(reversed(eval_metrics))
            latest_eval_metrics = eval_metrics[summary_step]
            # If there exists a new evaluation summary.
            if summary_step > self._current_summary_step:
                current_score = latest_eval_metrics[self._metric]
                if current_score is None:
                    current_score = float("nan")
                self._trial.report(float(current_score), step=summary_step)
                self._current_summary_step = summary_step
            if self._trial.should_prune():
                message = "Trial was pruned at iteration {}.".format(self._current_summary_step)
                raise optuna.TrialPruned(message)
