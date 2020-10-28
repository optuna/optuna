"""
.. _ud_pruner:

User-Defined Pruner
===================

This tutorial walks you through how :class:`~optuna.pruners.ThresholdPruner` is implemented to give you
a big picture of how you can implement your own pruners.

As you can see in the :class:`~optuna.pruner.BasePruner`,
what you need to implement is :func:`~optuna.pruner.BasePruner.prune`
which takes :class:`~optuna.study.Study` and currently being evaluated
:class:`~optuna.trial.FrozenTrial`.
This means that you can have the access to the annals of :class:`~optuna.trial.FrozenTrial`\\'s.
:class:`~optuna.pruners.SuccessiveHalvingPruner` utilizes this feature.

So, for the illustration purpose, I walk through you the implementation of :class:`~optuna.pruners.ThresholdPruner`\\'s :func:`~optuna.pruners.ThresholdPruner.prune`.

.. code:: python

    class ThresholdProuner(BasePruner):

        ...

        def pruner(
            self,
            study: optuna.study.Study,
            trial: optuna.trial.FrozenTrial
        ) -> bool:

            # `step` generally represents the iteration or epoch.
            step = trail.last_step

            # ``False`` means not pruned.
            if step is None:
                return False

            # Check whether the ``trial`` has run the enough ``steps``.
            if not _is_first_in_interval_step(
                step, trial.intermediate_values.keys(), n_warmup_steps, self._interval_steps
            ):
                return False

            latest_value = trial.intermediate_values[step]
            if math.isnan(latest_value):
                return True

            if latest_value < self._lower:
                return True

            if latest_value > self._upper:
                return True

            return False


"""
