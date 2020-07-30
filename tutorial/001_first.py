"""
.. _first:

First Optimization
==================

Quadratic Function Example
--------------------------

Usually, Optuna is used to optimize hyper-parameters, but as an example,
let us directly optimize a quadratic function in an IPython shell.
"""


import optuna

###################################################################################################
# The objective function is what will be optimized.


def objective(trial):
    x = trial.suggest_uniform('x', -10, 10)
    return (x - 2) ** 2


###################################################################################################
# This function returns the value of :math:`(x - 2)^2`. Our goal is to find the value of ``x``
# that minimizes the output of the ``objective`` function. This is the "optimization."
# During the optimization, Optuna repeatedly calls and evaluates the objective function with
# different values of ``x``.
#
# A :class:`~optuna.trial.Trial` object corresponds to a single execution of the objective
# function and is internally instantiated upon each invocation of the function.
#
# The `suggest` APIs (for example, :func:`~optuna.trial.Trial.suggest_float`) are called
# inside the objective function to obtain parameters for a trial.
# :func:`~optuna.trial.Trial.suggest_float` selects parameters uniformly within the range
# provided. In our example, from :math:`-10` to :math:`10`.
#
# To start the optimization, we create a study object and pass the objective function to method
# :func:`~optuna.study.Study.optimize` as follows.

study = optuna.create_study()
study.optimize(objective, n_trials=100)


###################################################################################################
# You can get the best parameter as follows.

print(study.best_params)

###################################################################################################
# We can see that the ``x`` value found by Optuna is close to the optimal value of ``2``.

###################################################################################################
# .. note::
#     When used to search for hyper-parameters in machine learning,
#     usually the objective function would return the loss or accuracy
#     of the model.


###################################################################################################
# Study Object
# ------------
#
# Let us clarify the terminology in Optuna as follows:
#
# * **Trial**: A single call of the objective function
# * **Study**: An optimization session, which is a set of trials
# * **Parameter**: A variable whose value is to be optimized, such as ``x`` in the above example
#
# In Optuna, we use the study object to manage optimization.
# Method :func:`~optuna.study.create_study` returns a study object.
# A study object has useful properties for analyzing the optimization outcome.

###################################################################################################
# To get the best parameter:


study.best_params

###################################################################################################
# To get the best value:

study.best_value


###################################################################################################
# To get the best trial:

study.best_trial


###################################################################################################
# To get all trials:

study.trials


###################################################################################################
# To get the number of trials:

len(study.trials)


###################################################################################################
# By executing :func:`~optuna.study.Study.optimize` again, we can continue the optimization.

study.optimize(objective, n_trials=100)


###################################################################################################
# To get the updated number of trials:

len(study.trials)
