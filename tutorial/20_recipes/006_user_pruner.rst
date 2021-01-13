.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_tutorial_009_user_pruner.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr_tutorial_009_user_pruner.py:


.. _user_pruner:

User-Defined Pruner
====================

In :ref:`sphx_glr_tutorial_007_pruning.py`, we described how an objective function can optionally include calls to a pruning feature which allows Optuna to terminate an optimization trial when intermediate results do not appear promising. In this document, we describe how to implement your own pruner, i.e., a custom strategy for determining when to stop a trial.

Overview of Pruner
-------------------

The :func:`~optuna.study.create_study` constructor takes, as an optional argument, a pruner inheriting from :class:`~optuna.pruners.BasePruner`. The pruner should implement the abstract method :meth:`~optuna.pruners.BaseSampler.prune`, which takes arguments for the study and the trial and returns a boolean value: `True` if the trial should be pruned and `False` otherwise. With this call signature, you are free to make the boolean determination using a combination of custom logic and and data from the study and/or trial objects. In particular, you can access all other trials through the study.get_trial() method and, and from a trial, its reported intermediate values through the trial.intermediate_values attribute.

You can refer to the source code of the built-in Optuna pruners as templates for building your own. In this document, for quick illustration, we describe the construction and usage of a simple (but aggressive) pruner which prunes trials that are in last place compared to completed trials at the same step.

Note
Please refer to the documentation of BasePruner or, for example, ThresholdPruner or PercentilePruner for more robust examples of pruner implementation, including error checking and complex pruner-internal logic. (todo)


An Example: Implementing LastPlacePruner
--------------------------------------------------

We aim to optimize the `loss` and `alpha` hyperparameters for a stochastic gradient descent classifier (SGDClassifier) run on the sklearn iris dataset. We implement a pruner which terminates a trial at a certain step if it is in last place compared to completed trials at the same step. We begin considering pruning after 1 training step and 5 completed trials. For demonstration purposes, we print() a diagnostic message from prune() when it is about to return False. 

It may be important to note that the SGDClassifier score, as it is evaluated on a holdout set, decreases with enough training steps due to overfitting. Optuna will register the intermediate value last reported as the value of a trial when it is pruned.

.. code-block:: python

  import numpy as np
  import optuna
  from optuna.pruners import BasePruner

  class LastPlacePruner(BasePruner):
    def __init__(self, warmup_steps, warmup_trials):
      self._warmup_steps = warmup_steps
      self._warmup_trials = warmup_trials

    def prune(self, study: "optuna.study.Study", trial: "optuna.trial.FrozenTrial") -> bool:
      # Get the latest score reported from this trial
      step = trial.last_step

      if step: # trial.last_step == None when no scores have been reported yet
        this_score = trial.intermediate_values[step]

        # Get scores from other trials in the study reported at the same step
        all_trials = study.get_trials(deepcopy=False)
        completed_trials = [t for t in all_trials if t.state == TrialState.COMPLETE]
        other_scores = [t.intermediate_values[step] for t in completed_trials
          if step in t.intermediate_values]
        other_scores = sorted(other_scores)

        # Prune if this trial at this step has a lower value than all completed trials
        # at the same step
        if step >= self._warmup_steps and len(other_scores) > self._warmup_trials:
          if this_score < other_scores[0]:
            print(f'prune() True: Trial {trial.number}, Step {step}, Score {this_score}')
            return True

      return False

You can use ``LastPlacePruner`` as you would a built-in pruner, passing it into `create_study()`

.. code-block:: python

  from sklearn.datasets import load_iris
  from sklearn.model_selection import train_test_split
  from sklearn.linear_model import SGDClassifier

def objective(trial):
  iris = load_iris()
  classes = np.unique(iris.target)
  X_train, X_valid, y_train, y_valid = train_test_split(
      iris.data, iris.target, train_size=100, test_size=50, random_state=0)

  loss = trial.suggest_categorical('loss', ['hinge', 'log', 'perceptron'])
  alpha = trial.suggest_loguniform('alpha', 0.00001, 0.001)
  clf = SGDClassifier(loss=loss, alpha=alpha, random_state=0)
  score = 0

  for step in range(0, 5):
    clf.partial_fit(X_train, y_train, classes=classes)
    score = clf.score(X_valid, y_valid)

    trial.report(score, step)

    if trial.should_prune():
      raise optuna.TrialPruned()

  return score

pruner = LastPlacePruner(warmup_steps=1, warmup_trials=5)
study = optuna.create_study(direction='maximize', pruner=pruner)
study.optimize(objective, n_trials=50)

Output:

.. code-block:: none
  A new study created in memory with name: no-name-41af23a8-d08d-49a8-9716-5ceaf882dd10
  Trial 0 finished with value: 0.9 and parameters: {'loss': 'log', 'alpha': 1.3556336320990736e-05}. Best is trial 0 with value: 0.9.
  Trial 1 finished with value: 0.7 and parameters: {'loss': 'log', 'alpha': 0.000389725611243791}. Best is trial 0 with value: 0.9.
  Trial 2 finished with value: 0.84 and parameters: {'loss': 'perceptron', 'alpha': 1.2865038874137284e-05}. Best is trial 0 with value: 0.9.
  Trial 3 finished with value: 0.7 and parameters: {'loss': 'hinge', 'alpha': 0.00035042913013081864}. Best is trial 0 with value: 0.9.
  Trial 4 finished with value: 0.86 and parameters: {'loss': 'log', 'alpha': 1.2879243641858533e-05}. Best is trial 0 with value: 0.9.
  Trial 5 finished with value: 0.72 and parameters: {'loss': 'log', 'alpha': 3.556020490950068e-05}. Best is trial 0 with value: 0.9.

  ...

  prune() True: Trial 23, Step 2, Score 0.58
  Trial 23 pruned. 
  prune() True: Trial 24, Step 2, Score 0.42
  Trial 24 pruned. 
  prune() True: Trial 25, Step 4, Score 0.68

  ...

  Trial 47 finished with value: 0.72 and parameters: {'loss': 'log', 'alpha': 0.00012269474870283797}. Best is trial 13 with value: 0.94.
  
.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  3.000 seconds)


.. _sphx_glr_download_tutorial_009_user_pruner.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example

  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: 009_user_pruner.py <009_user_pruner.py>`

.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
